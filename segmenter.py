# segmenter.py
import numpy as np
import open3d as o3d
import cv2

class PointCloudSegmenter:
    def __init__(self, camera_intrinsics):
        self.intrinsics = camera_intrinsics  # RealSense 相机内参 (fx, fy, cx, cy)

    def mask_to_pointcloud(self, mask, depth_image, color_image):
        """
        将掩膜中的点转为点云
        """
        fx, fy, cx, cy = self.intrinsics
        mask = mask.astype(np.uint8)
        indices = np.where(mask > 0)

        if len(indices[0]) == 0:
            return None

        z = depth_image[indices[0], indices[1]] / 1000.0  # mm → m
        x = (indices[1] - cx) * z / fx
        y = (indices[0] - cy) * z / fy

        points = np.vstack((x, y, z)).T
        colors = color_image[indices[0], indices[1]] / 255.0

        # 构建 Open3D 点云
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)

        return pcd

    def estimate_pose(self, pcd):
        """
        估计点云中心和表面法向量（用于吸取角度）
        """
        if len(pcd.points) < 10:
            return None, None

        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30))
        pcd.orient_normals_consistent_tangent_plane(50)

        # 计算几何中心
        center = np.mean(np.asarray(pcd.points), axis=0)
        normals = np.asarray(pcd.normals)

        # 选取中心附近点的平均法向量
        distances = np.linalg.norm(np.asarray(pcd.points) - center, axis=1)
        close_normals = normals[distances < 0.02]  # 2cm 内的法向量
        if len(close_normals) == 0:
            return center, None

        avg_normal = np.mean(close_normals, axis=0)
        avg_normal /= np.linalg.norm(avg_normal)

        return center, avg_normal
