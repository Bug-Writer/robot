# main.py (refactored, use logging instead of print)
from time import sleep
import logging

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

# ===================== Logging 配置 =====================
logging.basicConfig(
    level=logging.DEBUG,  # 日志等级
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)
# =========================================================

VISUALIZE = False
SAFE_Z_ABOVE = 30.0
LIFT_AFTER_PICK_MM = 50.0
PLACE_Z_CLEAR_MM = 100.0


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
    """将已抓取的物体移动到侧放位置并释放"""
    try:
        up_pos = center_robot_mm + np.array([0.0, 0.0, LIFT_AFTER_PICK_MM])
        logger.info(f"抬升到安全高度 (mm): {up_pos}")
        robot.blinx_move_coordinate_all(*up_pos, *euler_deg, speed=30)
        sleep(0.5)

        logger.info("移动到侧放安全位置")
        robot.blinx_move_angle(1, 45, -90)
        sleep(0.5)

        pos = robot.blinx_positive_solution()
        logger.debug(f"当前机械臂位置: {pos}")
        place_pos = np.array([pos[0], pos[1], center_robot_mm[2]])
        logger.info(f"下降到放置位置: {place_pos}")
        robot.blinx_move_coordinate_all(*place_pos, *euler_deg, speed=20)
        sleep(0.5)

        robot.blinx_pump_off()
        sleep(0.5)

        logger.info("放置后抬升")
        robot.blinx_move_angle_all(-90, 0, 0, 0, 0, 0, speed=30)
        sleep(0.5)

        logger.info("放置完成，终止机械臂通信")
        robot.blinx_close()
    except Exception:
        logger.exception("place_object_to_side 出现异常")


def handle_detection(robot, processor, res, rgb_image, depth_image, intrinsics):
    """处理单个检测结果"""
    try:
        mask = res['mask']
        center, normal, masked_pcd = processor.process_mask_with_pcd(mask, rgb_image, depth_image, intrinsics)
        if center is None or normal is None or masked_pcd is None:
            logger.warning("处理掩膜失败或未生成点云")
            return None, None

        if VISUALIZE:
            try:
                masked_pcd.paint_uniform_color([0.0, 1.0, 0.0])
                arrow = o3d.geometry.TriangleMesh.create_arrow(
                    cylinder_radius=0.002, cone_radius=0.004,
                    cylinder_height=0.02, cone_height=0.01
                )
                arrow.paint_uniform_color([1.0, 0.0, 0.0])
                arrow.translate(center)
                R = get_rotation_matrix_from_two_vectors(np.array([0, 0, 1]), normal / np.linalg.norm(normal))
                arrow.rotate(R, center=center)
                sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.005)
                sphere.translate(center)
                sphere.paint_uniform_color([0.0, 0.0, 1.0])
                o3d.visualization.draw_geometries([masked_pcd, arrow, sphere])
            except Exception:
                logger.exception("可视化失败，跳过")

        center_robot = transform_to_robot_frame(center)
        normal_robot = R_cam_to_robot @ normal
        center_robot_mm = center_robot * 1000.0

        logger.info("========== 检测到一个物体 ==========")
        logger.info(f"[相机坐标系] 中心坐标: {center}")
        logger.info(f"[相机坐标系] 法向量: {normal}")
        logger.info(f"[机械臂坐标系] 中心坐标: {center_robot_mm}")
        logger.info(f"[机械臂坐标系] 法向量: {normal_robot}")
        logger.info("==================================")

        euler_deg = compute_euler_from_normal(normal_robot)

        try:
            robot.blinx_move_angle(1, 45, 0)
            sleep(0.5)

            above_pos = center_robot_mm + np.array([0.0, 0.0, SAFE_Z_ABOVE])
            above_pos = np.around(above_pos, decimals=3)  # 对numpy数组保留3位小数

            # 对欧拉角保留3位小数精度（假设euler_deg是列表或元组）
            euler_deg = [round(angle, 3) for angle in euler_deg]  # 对每个角度保留3位小数

            # 日志输出和机器人移动均使用处理后的高精度值
            logger.info(f"移动到物体上方：{above_pos}, 姿态(°): {euler_deg}")
            robot.blinx_move_coordinate_all(
                above_pos[0], above_pos[1], above_pos[2],
                euler_deg[0], euler_deg[1], euler_deg[2],
                speed=30
            )
            sleep(0.5)

            logger.info(f"下降到物体位置：{center_robot_mm}, 姿态(°): {euler_deg}")
            robot.blinx_move_coordinate_all(*center_robot_mm, *euler_deg, speed=20)
            sleep(0.5)

            robot.blinx_pump_on()
            sleep(0.5)

            lifted = center_robot_mm + np.array([0.0, 0.0, LIFT_AFTER_PICK_MM])
            logger.info(f"抬起到：{lifted}")
            robot.blinx_move_coordinate_all(*lifted, 180, 0, 0, speed=20)
            sleep(0.5)

        except Exception:
            logger.exception("抓取过程出现异常")
            return None, None

        return center_robot_mm, euler_deg
    except Exception:
        logger.exception("handle_detection 出现异常")
        return None, None


def process_frame(camera, robot, detector, processor, visualizer):
    """处理一帧"""
    rgb_image, depth_image, _ = camera.get_aligned_frames()
    if rgb_image is None or depth_image is None:
        logger.warning("获取图像失败")
        return

    results = detector.detect(rgb_image)
    if not results:
        logger.info("当前帧未检测到物体")
        return

    for res in results:
        center_robot_mm, euler_deg = handle_detection(robot, processor, res, rgb_image, depth_image, camera.intrinsics)
        if center_robot_mm is None:
            continue
        place_object_to_side(robot, center_robot_mm, euler_deg)


def main():
    camera = DepthCamera()
    robot = Blinx_Six_Robot_Control()
    detector = YoloEDetector(model_path="yoloe-11m-seg.pt")
    processor = MaskPointCloudProcessor()
    visualizer = Visualizer()

    logger.info("系统启动中...")
    robot.blinx_home()
    sleep(0.5)
    robot.blinx_move_angle(1, 45, -90)
    sleep(0.5)

    try:
        while True:
            process_frame(camera, robot, detector, processor, visualizer)
    except KeyboardInterrupt:
        logger.info("主循环被中断，准备退出...")
    except Exception:
        logger.exception("main 异常")
    finally:
        camera.stop()
        robot.blinx_close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()