# main.py
from time import sleep

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

def main():
    # 初始化各模块
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

    while True:
        # 采集图像和点云
        rgb_image, depth_image, point_cloud = camera.get_aligned_frames()
        if rgb_image is None or depth_image is None:
            print("[WARNING] 获取图像失败")
            continue

        # YOLOE 推理：获取检测结果
        results = detector.detect(rgb_image)
        print(results)

        # 掩膜图像处理（以第一个检测目标为例）
        visualized_image = rgb_image.copy()
        if len(results) > 0:
            masks = []
            for res in results:
                mask = res['mask']
                masks.append(mask)

                # 点云掩膜提取与姿态估计
                center, normal = processor.process_mask(mask, rgb_image, depth_image, camera.intrinsics)
                visualized_image = visualizer.draw_mask(visualized_image, [mask])
                visualized_image = visualizer.draw_center_and_angle(visualized_image, center, normal, camera.intrinsics)
        else:
            print("[INFO] 当前帧未检测到物体。")

        # 显示结果（使用 Open3D）
        if len(results) > 0:
            # 创建一个空的点云列表，用于叠加显示
            all_geometry = []

            for res in results:
                mask = res['mask']
                # label = res['label']

                center, normal, masked_pcd = processor.process_mask_with_pcd(mask, rgb_image, depth_image, camera.intrinsics)

                # 掩膜点云颜色设为统一亮色，或保持 RGB 原色
                masked_pcd.paint_uniform_color([0.0, 1.0, 0.0])  # 绿色

                # 创建法向量箭头
                arrow = o3d.geometry.TriangleMesh.create_arrow(
                    cylinder_radius=0.002,
                    cone_radius=0.004,
                    cylinder_height=0.02,
                    cone_height=0.01
                )
                arrow.paint_uniform_color([1.0, 0.0, 0.0])  # 红色箭头
                arrow.translate(center)
                arrow.rotate(arrow.get_rotation_matrix_from_xyz((0, 0, 0)), center=center)
                arrow_direction = normal / np.linalg.norm(normal)
                default_direction = np.array([0, 0, 1])  # 原始箭头朝向 z
                R = get_rotation_matrix_from_two_vectors(default_direction, arrow_direction)
                arrow.rotate(R, center=center)

                # 创建中心点球体
                sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.005)
                sphere.translate(center)
                sphere.paint_uniform_color([0.0, 0.0, 1.0])  # 蓝色

                all_geometry.extend([masked_pcd, arrow, sphere])

                print("========== 检测到一个物体 ==========")
                print(f"[相机坐标系] 中心坐标: {center}")
                print(f"[相机坐标系] 法向量: {normal}")

                # 将相机坐标系转换为机械臂坐标系
                center_robot = transform_to_robot_frame(center)
                normal_robot = R_cam_to_robot @ normal  # 只变换方向向量
                center_robot_mm = center_robot * 1000.0

                print(f"[机械臂坐标系] 中心坐标: {center_robot_mm}")
                print(f"[机械臂坐标系] 法向量: {normal_robot}")
                print("==================================")

                # 计算姿态角（以法向量为Z轴）
                z_axis = normal_robot / np.linalg.norm(normal_robot)
                x_axis = np.array([1, 0, 0])
                if np.allclose(np.abs(np.dot(z_axis, x_axis)), 1.0):
                    x_axis = np.array([0, 1, 0])  # 避免共线
                y_axis = np.cross(z_axis, x_axis)
                x_axis = np.cross(y_axis, z_axis)
                R_mat = np.vstack([x_axis, y_axis, z_axis]).T

                # 从旋转矩阵计算欧拉角（XYZ顺序）
                import scipy.spatial.transform
                euler_deg = scipy.spatial.transform.Rotation.from_matrix(R_mat).as_euler('xyz', degrees=True)
                euler_deg = euler_deg.tolist()

                # 调用机械臂移动到目标点上方 3cm，然后下移吸取

                robot.blinx_move_angle(1, 45, 0)
                sleep(0.5)

                print(f"[ACTION] 移动到物体上方：位置(mm): {center_robot_mm + np.array([0, 0, 30])}, 姿态(°): {[euler_deg[0], euler_deg[1], euler_deg[2]]}")
                robot.blinx_move_coordinate_all(
                    center_robot_mm[0], center_robot_mm[1], center_robot_mm[2] + 30,
                    euler_deg[0], euler_deg[1], euler_deg[2],
                    speed=30
                )
                sleep(0.5)

                print(f"[ACTION] 下降到物体位置：位置(mm): {center_robot_mm}, 姿态(°): {[euler_deg[0], euler_deg[1], euler_deg[2]]}")
                robot.blinx_move_coordinate_all(
                    center_robot_mm[0], center_robot_mm[1], center_robot_mm[2],
                    euler_deg[0], euler_deg[1], euler_deg[2],
                    speed=20
                )
                sleep(0.5)

                robot.blinx_pump_on()
                sleep(0.5)

                print(f"[ACTION] 吸取完成抬升：位置(mm): {center_robot_mm + np.array([0, 0, 50])}, 姿态(°): [180, 0, 0]")
                robot.blinx_move_coordinate_all(
                    center_robot_mm[0], center_robot_mm[1], center_robot_mm[2] + 50,
                    180, 0, 0,
                    speed=20
                )
                sleep(0.5)
            # 使用 Open3D 可视化
            o3d.visualization.draw_geometries(all_geometry)

    # 释放资源
    camera.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
