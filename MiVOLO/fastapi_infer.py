import os
import cv2
import torch
import numpy as np
from mivolo.predictor import Predictor


class MivoloInference:
    def __init__(self,
                 checkpoint_path: str = "models/mivolo_imbd.pth.tar",
                 detector_weights_path: str = "models/yolov8x_person_face.pt",
                 device: str = 'cpu',
                 with_persons: bool = True,
                 disable_faces: bool = False):

        # 模拟 argparse 参数
        class Config:
            def __init__(self):
                self.output = None
                self.detector_weights = detector_weights_path
                self.checkpoint = checkpoint_path
                self.with_persons = with_persons
                self.disable_faces = disable_faces
                self.draw = False
                self.device = device


        args = Config()
        self.age_mapping = ["男性", "女性"]

        # 加速设置
        if torch.cuda.is_available() and device == 'cuda':
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.benchmark = True

        print(f"Loading MiVOLO models on {device}...")
        self.predictor = Predictor(args, verbose=False)
        print("Models loaded successfully.")

    def extract_frames(self, video_path: str, save_path: str) -> str:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return "无法打开视频，无法预测年龄"
        frame_count = 0
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_count += 1
                if frame_count % 15 != 0:
                    continue
                age_result = self.predict(frame, save_path)
                if age_result is not None:
                    return f"{self.age_mapping[age_result[0]]}年龄是{age_result[1]}"
        finally:
            cap.release()
        return "未检测到清晰人脸，无法预测年龄"


    def predict(self, bgr_image: np.ndarray, save_path: str = None) -> tuple[int, float]:
        """
        推理函数
        :param bgr_image: 输入的 numpy BGR 图像 (H, W, 3)
        :param save_path: (可选) 结果图片的保存路径，如果为 None 则不保存
        :return: (gender_int, age_float)
                 gender_int: 0=Male, 1=Female
                 age_float: 预测年龄
        """


        # 执行推理
        detected_objects, _ = self.predictor.recognize(bgr_image)

        if not detected_objects or detected_objects.n_objects == 0:
            return None

        yolo_boxes = detected_objects.yolo_results.boxes
        names = detected_objects.yolo_results.names

        target_index = -1
        max_area = 0

        # 1. 寻找面积最大的人体 (Person)
        for i, box in enumerate(yolo_boxes):
            cls_id = int(box.cls)
            if names[cls_id] == 'person':
                xyxy = box.xyxy.squeeze().cpu().numpy()
                area = (xyxy[2] - xyxy[0]) * (xyxy[3] - xyxy[1])
                if area > max_area:
                    max_area = area
                    target_index = i

        # 2. 如果没找到人，找面积最大的脸 (Face)
        if target_index == -1:
            for i, box in enumerate(yolo_boxes):
                cls_id = int(box.cls)
                if names[cls_id] == 'face':
                    xyxy = box.xyxy.squeeze().cpu().numpy()
                    area = (xyxy[2] - xyxy[0]) * (xyxy[3] - xyxy[1])
                    if area > max_area:
                        max_area = area
                        target_index = i

        if target_index != -1:
            # 获取数据
            gender_output = detected_objects.genders[target_index]
            age_output = detected_objects.ages[target_index]

            if gender_output is None or age_output is None:
                return None

            # 处理性别 (0=Male, 1=Female)
            if isinstance(gender_output, str):
                is_male = gender_output.lower().startswith('m')
                gender_int = 0 if is_male else 1
            else:
                # 假设模型输出 0:Male, 1:Female (具体视训练配置而定，通常mivolo是这样)
                gender_int = int(gender_output)
                is_male = (gender_int == 0)

            # ---------------------------------------------------------
            # 绘图逻辑 (如果提供了 save_path)
            # ---------------------------------------------------------
            if save_path:
                try:
                    # 获取框坐标并转为整数
                    box = yolo_boxes[target_index]
                    x1, y1, x2, y2 = map(int, box.xyxy.squeeze().tolist())

                    # 准备标签文字
                    gender_str = "Male" if gender_int == 0 else "Female"
                    label = f"{gender_str} {age_output:.1f}"

                    # 颜色定义 (BGR)
                    color = (0, 255, 0)  # 绿框
                    txt_color = (255, 255, 255)  # 白字
                    bg_color = (0, 0, 0)  # 黑底 (用于文字背景)

                    font_scale = 0.8
                    thickness = 2

                    # 1. 画人脸/人体框
                    cv2.rectangle(bgr_image, (x1, y1), (x2, y2), color, 2)

                    # 2. 计算文字大小
                    (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)

                    # 3. 确定标签位置 (默认在框左上角上方，防止遮挡人脸)
                    # 如果上方空间不足，则画在框内左上角
                    if y1 - text_h - 10 > 0:
                        text_origin_y = y1 - 10
                        # 背景矩形坐标
                        bg_p1 = (x1, y1 - text_h - 15)
                        bg_p2 = (x1 + text_w, y1)
                    else:
                        text_origin_y = y1 + text_h + 10
                        bg_p1 = (x1, y1)
                        bg_p2 = (x1 + text_w, y1 + text_h + 15)

                    # 4. 画文字背景 (实心矩形)
                    cv2.rectangle(bgr_image, bg_p1, bg_p2, color, -1)  # 使用框的颜色作为背景，或者改成 bg_color

                    # 5. 写字
                    cv2.putText(bgr_image, label, (x1, text_origin_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                                txt_color if color == (0, 0, 0) else (0, 0, 0), thickness)

                    # 6. 保存图片
                    # 确保目录存在
                    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
                    cv2.imwrite(save_path, bgr_image)
                    # print(f"Result saved to {save_path}")

                except Exception as e:
                    print(f"Error saving image: {e}")

            return gender_int, float(age_output)

        return None


# ==========================================
# 使用示例
# ==========================================
if __name__ == "__main__":
    # 1. 实例化模型 (只做一次)
    # 请替换为你实际的权重文件路径
    ckpt_path = "MiVOLO/models/mivolo_imbd.pth.tar"
    det_path = "MiVOLO/models/yolov8x_person_face.pt"
    # det_path = "models/yolov8s-face-lindevs.pt"
    import time

    # 检查文件是否存在，防止演示报错
    # if os.path.exists(ckpt_path) and os.path.exists(det_path):
    model = MivoloInference(checkpoint_path=ckpt_path,
                            detector_weights_path=det_path
                            )

    # 2. 准备一张 RGB 图片 (模拟输入)
    # 这里用 OpenCV 读取是 BGR，所以演示需转为 RGB 以符合你的输入要求
    img_bgr = cv2.imread("MiVOLO/test_image.jpg")
    if img_bgr is not None:

        start = time.time()
        # 3. 调用推理
        result = model.predict(img_bgr, save_path="result_labeled.jpg")
        print(time.time() - start)
        if result:
            gender, age = result
            print(f"Result -> Gender: {gender}, Age: {age:.2f}")
        else:
            print("No person/face detected.")
    # else:
    #     print("请在示例代码中填入正确的权重文件路径。")