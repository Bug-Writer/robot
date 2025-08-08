# calibration.py
import numpy as np

# 相机到机械臂的变换矩阵（假设你已经标定得到了）
# R 是 3x3 旋转矩阵，T 是 3x1 平移向量（单位米）

R_cam_to_robot = np.array([
    [0, -1, 0],  # X轴 -> -Y轴
    [-1, 0, 0],  # Y轴 -> -X轴
    [0,  0,-1]   # Z轴 -> -Z轴
])

T_cam_to_robot = np.array([0.25, 0.0, 0.44])  # 示例：相机在机械臂坐标系中位置是(30cm, 0cm, 20cm)

def transform_to_robot_frame(point_cam):
    """
    将相机坐标系下的点变换到机械臂坐标系
    :param point_cam: numpy 数组 [x, y, z]
    :return: numpy 数组 [x, y, z] in robot frame
    """
    return R_cam_to_robot @ point_cam + T_cam_to_robot