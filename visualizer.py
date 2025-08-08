# visualizer.py
import cv2
import numpy as np

class Visualizer:
    def __init__(self):
        pass

    def draw_mask(self, image, masks, alpha=0.4):
        """
        在图像上叠加半透明掩膜
        """
        overlay = image.copy()
        for mask in masks:
            color = np.random.randint(0, 255, (1, 3), dtype=np.uint8).tolist()[0]
            color_mask = np.zeros_like(image)
            color_mask[mask > 0] = color
            overlay = cv2.addWeighted(overlay, 1, color_mask, alpha, 0)
        return overlay

    def project_point_to_image(self, point_3d, intrinsics):
        """
        将相机坐标系下的 3D 点投影为 2D 图像坐标
        :param point_3d: (x, y, z)
        :param intrinsics: 相机内参字典，含 fx, fy, cx, cy
        :return: (u, v) 像素坐标
        """
        x, y, z = point_3d
        if z == 0:
            return None
        u = int((x * intrinsics['fx']) / z + intrinsics['cx'])
        v = int((y * intrinsics['fy']) / z + intrinsics['cy'])
        return (u, v)

    def draw_center_and_angle(self, image, center_3d, normal_3d, intrinsics=None, scale=50):
        """
        在图像上绘制中心点与法向量方向箭头（需要投影）
        :param image: 原始 RGB 图像
        :param center_3d: 相机坐标系下中心点 (x, y, z)
        :param normal_3d: 相机坐标系下法向量 (x, y, z)
        :param intrinsics: 相机内参字典 {fx, fy, cx, cy}
        :param scale: 箭头长度缩放
        :return: 带绘图的图像
        """
        img = image.copy()

        if center_3d is None or normal_3d is None:
            return img

        if intrinsics is None:
            print("[WARN] draw_center_and_angle 缺少相机内参，将跳过绘图")
            return img

        # 投影中心点
        center_px = self.project_point_to_image(center_3d, intrinsics)
        if center_px is None:
            return img

        cv2.circle(img, center_px, 5, (0, 255, 0), -1)

        # 计算箭头终点的 3D 坐标（沿法向量方向偏移）
        end_3d = center_3d + np.array(normal_3d) * 0.05  # 0.05 米偏移（可调）
        end_px = self.project_point_to_image(end_3d, intrinsics)

        if end_px:
            cv2.arrowedLine(img, center_px, end_px, (0, 0, 255), 2, tipLength=0.3)

        return img

    def draw_boxes(self, image, boxes, labels=None, scores=None):
        """
        绘制检测框
        """
        img = image.copy()
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 0), 2)
            if labels is not None:
                label = labels[i]
                score = scores[i] if scores is not None else None
                text = f"{label} {score:.2f}" if score else str(label)
                cv2.putText(img, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        return img
