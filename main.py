# main.py (refactored, adds "place to side" step)
from time import sleep
import sys
import traceback

import cv2
import numpy as np
import open3d as o3d

from camera import DepthCamera
from detector import YoloEDetector
from mask_pointcloud_processor import MaskPointCloudProcessor
from pointcloud_utils import get_rotation_matrix_from_two_vectors
from Six_Robot_Control import Blinx_Six_Robot_Control
from calibration import transform_to_robot_frame, R_cam_to_robot
from visualizer import Visualizer

VISUALIZE = False            # 是否显示 Open3D 可视化（阻塞或占用线程），默认 False
SIDE_OFFSET_MM = np.array([200.0, 0.0, 0.0])  # 在机械臂坐标系中放置偏移（毫米）
SAFE_Z_ABOVE = 30.0          # 抓取上方安全高度（mm）
LIFT_AFTER_PICK_MM = 50.0    # 抓取后抬升高度（mm）
PLACE_Z_CLEAR_MM = 100.0     # 到达侧放位置时的安全高度（mm）

def compute_euler_from_normal(normal_robot):
    """给定机械臂坐标系下的法向量，生成姿态欧拉角（degrees）"""
    z_axis = normal_robot / np.linalg.norm(normal_robot)
    x_axis = np.array([1, 0, 0], dtype=float)
    if np.allclose(np.abs(np.dot(z_axis, x_axis)), 1.0):
        x_axis = np.array([0, 1, 0], dtype=float)
    y_axis = np.cross(z_axis, x_axis)
    x_axis = np.cross(y_axis, z_axis)
    R_mat = np.vstack([x_axis, y_axis, z_axis]).T
    import scipy.spatial.transform
    euler_deg = scipy.spatial.transform.Rotation.from_matrix(R_mat).as_euler('xyz', degrees=True)
    return euler_deg.tolist()

def place_object_to_side(robot, center_robot_mm, euler_deg):
    """
    将已抓取的物体移动到侧放位置并释放。
    center_robot_mm: numpy array (3,) 单位 mm
    euler_deg: [rx, ry, rz] 角度列表
    """
    try:
        # 抬升到安全高度
        up_pos = center_robot_mm + np.array([0.0, 0.0, LIFT_AFTER_PICK_MM])
        print(f"[ACTION] 抬升到安全高度 (mm): {up_pos}")
        robot.blinx_move_coordinate_all(
            up_pos[0], up_pos[1], up_pos[2],
            euler_deg[0], euler_deg[1], euler_deg[2],
            speed=30
        )
        sleep(0.5)

        # 移动到侧放位置（在XY平面偏移 SIDE_OFFSET_MM，Z 到 PLACE_Z_CLEAR_MM 高度以避免碰撞）
        side_pos = center_robot_mm + SIDE_OFFSET_MM
        side_pos_safe = np.array([side_pos[0], side_pos[1], PLACE_Z_CLEAR_MM])
        print(f"[ACTION] 移动到侧放安全位置 (mm): {side_pos_safe}")
        robot.blinx_move_coordinate_all(
            side_pos_safe[0], side_pos_safe[1], side_pos_safe[2],
            euler_deg[0], euler_deg[1], euler_deg[2],
            speed=40
        )
        sleep(0.5)

        # 下降到放置高度（可调整为 center_robot_mm 的 Z 或更低）
        place_pos = np.array([side_pos[0], side_pos[1], center_robot_mm[2]])
        print(f"[ACTION] 下降到放置位置 (mm): {place_pos}")
        robot.blinx_move_coordinate_all(
            place_pos[0], place_pos[1], place_pos[2],
            euler_deg[0], euler_deg[1], euler_deg[2],
            speed=20
        )
        sleep(0.5)

        # 关闭吸泵（释放物体）
        robot.blinx_pump_off()
        sleep(0.5)

        # 抬升到安全高度（完成）
        post_up = np.array([side_pos[0], side_pos[1], side_pos[2] + LIFT_AFTER_PICK_MM])
        print(f"[ACTION] 放置后抬升 (mm): {post_up}")
        robot.blinx_move_coordinate_all(
            post_up[0], post_up[1], post_up[2],
            180, 0, 0,  # 这里可以改为固定姿态或保持原姿态
            speed=30
        )
        sleep(0.5)
        print("[ACTION] 放置完成。")
    except Exception:
        print("[ERROR] place_object_to_side 出现异常：")
        traceback.print_exc()

def handle_detection(robot, processor, res, rgb_image, depth_image, intrinsics):
    """
    处理单个检测结果：计算中心、法向量、点云、控制机械臂抓取并返回相机坐标系中心与法向量
    返回 (center_camera, normal_camera) 或 (None, None) 表示失败
    """
    try:
        mask = res['mask']
        center, normal, masked_pcd = processor.process_mask_with_pcd(mask, rgb_image, depth_image, intrinsics)
        if center is None or normal is None or masked_pcd is None:
            print("[WARN] 处理掩膜失败或未生成点云")
            return None, None

        # 可视化处理（如果需要）
        if VISUALIZE:
            try:
                masked_pcd.paint_uniform_color([0.0, 1.0, 0.0])
                arrow = o3d.geometry.TriangleMesh.create_arrow(
                    cylinder_radius=0.002, cone_radius=0.004, cylinder_height=0.02, cone_height=0.01
                )
                arrow.paint_uniform_color([1.0, 0.0, 0.0])
                arrow.translate(center)
                arrow_direction = normal / np.linalg.norm(normal)
                default_direction = np.array([0, 0, 1])
                R = get_rotation_matrix_from_two_vectors(default_direction, arrow_direction)
                arrow.rotate(R, center=center)
                sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.005)
                sphere.translate(center)
                sphere.paint_uniform_color([0.0, 0.0, 1.0])
                o3d.visualization.draw_geometries([masked_pcd, arrow, sphere])
            except Exception:
                print("[WARN] 可视化失败，跳过。")
                traceback.print_exc()

        # 将相机坐标系转换为机械臂坐标系
        center_robot = transform_to_robot_frame(center)           # 单位 m （假设）
        normal_robot = R_cam_to_robot @ normal                    # 方向向量
        center_robot_mm = center_robot * 1000.0                   # 转为 mm

        print("========== 检测到一个物体 ==========")
        print(f"[相机坐标系] 中心坐标: {center}")
        print(f"[相机坐标系] 法向量: {normal}")
        print(f"[机械臂坐标系] 中心坐标 (mm): {center_robot_mm}")
        print(f"[机械臂坐标系] 法向量: {normal_robot}")
        print("==================================")

        # 计算姿态角
        euler_deg = compute_euler_from_normal(normal_robot)

        # 机械臂动作：移动到物体上方 -> 下降 -> 吸取 -> 抬起
        try:
            robot.blinx_move_angle(1, 45, 0)
            sleep(0.5)

            # 移动到上方（+ SAFE_Z_ABOVE mm）
            above_pos = center_robot_mm + np.array([0.0, 0.0, SAFE_Z_ABOVE])
            print(f"[ACTION] 移动到物体上方：{above_pos}, 姿态(°): {euler_deg}")
            robot.blinx_move_coordinate_all(
                above_pos[0], above_pos[1], above_pos[2],
                euler_deg[0], euler_deg[1], euler_deg[2],
                speed=30
            )
            sleep(0.5)

            # 下降到目标位置
            print(f"[ACTION] 下降到物体位置：{center_robot_mm}, 姿态(°): {euler_deg}")
            robot.blinx_move_coordinate_all(
                center_robot_mm[0], center_robot_mm[1], center_robot_mm[2],
                euler_deg[0], euler_deg[1], euler_deg[2],
                speed=20
            )
            sleep(0.5)

            # 吸取
            robot.blinx_pump_on()
            sleep(0.5)

            # 抬起
            lifted = center_robot_mm + np.array([0.0, 0.0, LIFT_AFTER_PICK_MM])
            print(f"[ACTION] 抬起到：{lifted}")
            robot.blinx_move_coordinate_all(
                lifted[0], lifted[1], lifted[2],
                180, 0, 0,
                speed=20
            )
            sleep(0.5)
        except Exception:
            print("[ERROR] 抓取过程出现异常：")
            traceback.print_exc()
            return None, None

        # 返回抓取后位置与姿态（用于侧放）
        return center_robot_mm, euler_deg
    except Exception:
        print("[ERROR] handle_detection 出现异常：")
        traceback.print_exc()
        return None, None

def process_frame(camera, robot, detector, processor, visualizer):
    """处理一帧：检测、处理每个目标并执行抓取+侧放"""
    rgb_image, depth_image, _ = camera.get_aligned_frames()
    if rgb_image is None or depth_image is None:
        print("[WARNING] 获取图像失败")
        return

    # YOLOE 推理
    results = detector.detect(rgb_image)
    if not results:
        print("[INFO] 当前帧未检测到物体。")
        return

    # 逐个处理检测到的物体
    for res in results:
        center_robot_mm, euler_deg = handle_detection(robot, processor, res, rgb_image, depth_image, camera.intrinsics)
        if center_robot_mm is None:
            continue

        # 侧放到一侧
        place_object_to_side(robot, center_robot_mm, euler_deg)

def main():
    camera = DepthCamera()
    robot = Blinx_Six_Robot_Control()
    detector = YoloEDetector(model_path="yoloe-11m-seg.pt")
    processor = MaskPointCloudProcessor()
    visualizer = Visualizer()

    print("[INFO] 系统启动中...")
    robot.blinx_home()
    sleep(0.5)
    robot.blinx_move_angle(1, 45, -90)
    sleep(0.5)

    try:
        while True:
            process_frame(camera, robot, detector, processor, visualizer)

            # cv2.imshow("RGB", rgb_image)
            # cv2.imshow("Depth", (depth_image / np.max(depth_image + 1)).astype(np.float32))
            # if cv2.waitKey(1) == 27:
            #     break

    except KeyboardInterrupt:
        print("[INFO] 主循环被中断，准备退出...")
    except Exception:
        print("[ERROR] main 异常：")
        traceback.print_exc()
    finally:
        camera.stop()
        robot.blinx_close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
