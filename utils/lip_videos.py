import cv2
import mediapipe as mp
import numpy as np


class LipVideo:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
                    static_image_mode=False,
                    max_num_faces=1,
                    refine_landmarks=True,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5)
        self.lips = [61, 291, 17]
        self.desired_pos=np.float32([[10, 20], [90, 20], [45, 50]])
        self.out_size=(90, 50)
        self.lip_index_list_outer = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146]
        self.lip_index_list_inner = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
        self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    def get_matrix(self, face_landmarks, shape):
        lip_series = []

        for idx in self.lip_index_list_outer:
            x = face_landmarks.landmark[idx].x
            y = face_landmarks.landmark[idx].y
            relative_x = int(x * shape[1])
            relative_y = int(y * shape[0])
            lip_series.append((relative_x, relative_y))

        for idx in self.lip_index_list_inner:
            x = face_landmarks.landmark[idx].x
            y = face_landmarks.landmark[idx].y
            relative_x = int(x * shape[1])
            relative_y = int(y * shape[0])
            lip_series.append((relative_x, relative_y))

        return lip_series

    def save_video(self, frame_list, save_path, fps, h, w):
        out = cv2.VideoWriter(save_path, self.fourcc, fps, (w, h))
        for img in frame_list:
            out.write(img)
        out.release()

    def infer(self, file_path, save_path):
        try:
            cap = cv2.VideoCapture(file_path)
            shape = (int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            lip_frame_list = []
            h, w = shape
            while True:
                ret, frame = cap.read()
                if ret is False:
                    break
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.face_mesh.process(frame_rgb)
                if results.multi_face_landmarks:
                    lip_series = self.get_matrix(results.multi_face_landmarks[0], shape)
                    matrix = frame.copy()
                    for i in range(len(lip_series)):
                        if i <= len(self.lip_index_list_outer):
                            cv2.circle(matrix, lip_series[i], 5, (255, 0, 0), -1)
                            cv2.line(matrix, lip_series[i], lip_series[(i + 1) % len(self.lip_index_list_outer)],
                                     (255, 0, 0), 5)
                        else:
                            cv2.circle(matrix, lip_series[i], 5, (0, 255, 0), -1)
                            cv2.line(matrix, lip_series[i], lip_series[(i + 1) % len(self.lip_index_list_inner)],
                                     (0, 255, 0), 5)
                    lip_frame_list.append(matrix)
            self.save_video(lip_frame_list, save_path, fps, h, w)
        except Exception as e:
            print(f"lip_video.infer 错误: {e}")
