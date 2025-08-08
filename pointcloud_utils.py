# utils/pointcloud_utils.py

import numpy as np
import open3d as o3d

def rgbd_to_pointcloud(color_img, depth_img, intrinsics):
    """
    使用 RGB 图像和深度图生成完整点云
    """
    rgb = o3d.geometry.Image(color_img.astype(np.uint8))
    depth = o3d.geometry.Image(depth_img.astype(np.uint16))

    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color=rgb,
        depth=depth,
        convert_rgb_to_intensity=False,
        depth_scale=1000.0,
        depth_trunc=2.0
    )

    intrinsic = o3d.camera.PinholeCameraIntrinsic()
    intrinsic.set_intrinsics(
        width=color_img.shape[1],
        height=color_img.shape[0],
        fx=intrinsics['fx'],
        fy=intrinsics['fy'],
        cx=intrinsics['cx'],
        cy=intrinsics['cy']
    )

    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image,
        intrinsic
    )

    return pcd

def mask_to_pointcloud(mask, depth_img, rgb_img, intrinsics):
    """
    将掩膜区域转换为点云对象
    """
    assert depth_img.shape == mask.shape, "掩膜与深度图尺寸不匹配"

    # 获取掩膜对应的像素位置
    mask_indices = np.where(mask)

    z = depth_img[mask_indices] / 1000.0  # 深度值转换为米，形状为 (N,)
    x = (mask_indices[1] - intrinsics['cx']) * z / intrinsics['fx']
    y = (mask_indices[0] - intrinsics['cy']) * z / intrinsics['fy']

    points = np.stack((x, y, z), axis=-1)  # (N, 3)

    # 获取颜色
    colors = rgb_img[mask_indices].astype(np.float32) / 255.0  # (N, 3)

    # 构建 open3d 点云
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    return pcd


def estimate_normal(pcd, center, radius=0.02):
    """
    给定点云和中心点，估计中心点处的法向量
    """
    # 估计法线
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
        radius=radius, max_nn=30
    ))

    # 如果你想返回中心最近点的法向量：
    kdtree = o3d.geometry.KDTreeFlann(pcd)
    [_, idx, _] = kdtree.search_knn_vector_3d(center, 1)
    normal = np.asarray(pcd.normals)[idx[0]]
    return normal

def get_rotation_matrix_from_two_vectors(vec1, vec2):
    """
    返回一个旋转矩阵，使得 vec1 旋转到 vec2
    使用 Rodrigues 公式计算
    """
    v1 = vec1 / np.linalg.norm(vec1)
    v2 = vec2 / np.linalg.norm(vec2)
    v = np.cross(v1, v2)
    c = np.dot(v1, v2)
    s = np.linalg.norm(v)

    # 若向量相同
    if s == 0:
        return np.eye(3) if c > 0 else -np.eye(3)

    vx = np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])

    R = np.eye(3) + vx + (vx @ vx) * ((1 - c) / (s ** 2))
    return R