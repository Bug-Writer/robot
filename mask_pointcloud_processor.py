# mask_pointcloud_processor.py

import numpy as np
import open3d as o3d
from pointcloud_utils import mask_to_pointcloud, estimate_normal

class MaskPointCloudProcessor:
    def __init__(self):
        pass

    def process_mask(self, mask, rgb_img, depth_img, intrinsics):
        # 掩膜 → 点云
        masked_pcd = mask_to_pointcloud(mask, depth_img, rgb_img, intrinsics)
        if len(masked_pcd.points) == 0:
            print("[WARNING] 掩膜区域内无有效点云")
            return None, None

        # 点云中心（几何中心）
        center = np.mean(np.asarray(masked_pcd.points), axis=0)

        # 法向量估计
        normal = estimate_normal(masked_pcd, center)

        return center, normal

    def process_mask_with_pcd(self, mask, rgb_img, depth_img, intrinsics):
        masked_pcd = mask_to_pointcloud(mask, depth_img, rgb_img, intrinsics)
        center = np.mean(np.asarray(masked_pcd.points), axis=0)
        normal = estimate_normal(masked_pcd, center)
        return center, normal, masked_pcd