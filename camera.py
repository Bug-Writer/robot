import pyrealsense2 as rs
import numpy as np
import open3d as o3d
import cv2

class DepthCamera:
    def __init__(self):
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

        self.profile = self.pipeline.start(config)

        # 对齐深度到彩色图像
        self.align = rs.align(rs.stream.color)

        # ====== 添加 RealSense 滤波器 ======
        self.spatial = rs.spatial_filter()       # 空间平滑
        self.spatial.set_option(rs.option.holes_fill, 3)  # 填补小空洞（0~5）

        self.temporal = rs.temporal_filter()     # 时间平滑

        self.hole_filling = rs.hole_filling_filter()  # 空洞填补

        # 获取内参
        color_stream = self.profile.get_stream(rs.stream.color)
        intrinsics = color_stream.as_video_stream_profile().get_intrinsics()
        self.intrinsics = {
            'fx': intrinsics.fx,
            'fy': intrinsics.fy,
            'cx': intrinsics.ppx,
            'cy': intrinsics.ppy
        }

    def get_aligned_frames(self):
        """
        获取对齐后的 RGB、深度图 和点云（深度平滑后）
        """
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)

        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()

        if not color_frame or not depth_frame:
            return None, None, None

        # ====== 在这里应用滤波器 ======
        depth_frame = self.spatial.process(depth_frame)
        depth_frame = self.temporal.process(depth_frame)
        depth_frame = self.hole_filling.process(depth_frame)

        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

        # 转换为点云
        rgb = o3d.geometry.Image(color_image)
        depth = o3d.geometry.Image(depth_image)

        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color=rgb,
            depth=depth,
            convert_rgb_to_intensity=False,
            depth_scale=1000.0,
            depth_trunc=2.0
        )

        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            width=color_image.shape[1],
            height=color_image.shape[0],
            fx=self.intrinsics['fx'],
            fy=self.intrinsics['fy'],
            cx=self.intrinsics['cx'],
            cy=self.intrinsics['cy']
        )

        point_cloud = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd,
            intrinsic
        )

        return color_image, depth_image, point_cloud

    def get_frames(self):
        """
        获取未对齐的 RGB 和深度图（深度平滑后）
        """
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        if not color_frame or not depth_frame:
            return None, None

        # ====== 滤波 ======
        depth_frame = self.spatial.process(depth_frame)
        depth_frame = self.temporal.process(depth_frame)
        depth_frame = self.hole_filling.process(depth_frame)

        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())
        return color_image, depth_image

    def stop(self):
        self.pipeline.stop()
