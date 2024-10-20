from time import time
import numpy as np
from ultralytics import YOLO
import cv2
import torch
import math


class DistanceEstimationDetector:
    def __init__(self, video_path, model_path):
        """
        :param video_path: 要处理的视频
        :param model_path:
        """
        self.video_path = video_path
        self.model = self.load_model(model_path)
        self.classes = self.model.names
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Using Device: ", self.device)

    def get_video_capture(self):
        """
        :return:
        """
        return cv2.VideoCapture(self.video_path)

    def load_model(self, model_path):
        """
        Model
        :param model_path:
        :return:
        """
        model = YOLO(model_path)
        # model = yolov5.load(model_path)
        # model.conf = 0.40  # confidence 阈值
        # model.iou = 0.45  # NMS IoU threshold
        # model.max_det = 1000  # 最大检测数（每帧）
        model.classes = [0, 2]  # 对象的 class number  people & car
        return model

    def get_model_results(self, frame):
        """
        返回预测和预测结果
        :param frame: 视频帧
        :return: 输入帧的结果
        """
        self.model.to(self.device)
        frame = [frame]
        # results = self.model(frame, size=640)  # yolov5
        results = self.model(frame)

        # YOLOv5
        # predictions = results.xyxyn[0]
        # cords, scores, labels = predictions[:, :4], predictions[:, 4], predictions[:, 5]
        # YOLOv8
        boxes = results[0].boxes
        cords, scores, labels = boxes.xyxyn, boxes.conf, boxes.cls

        return cords, scores, labels

    def draw_rect(self, results, frame):
        """
        绘制框
        :param results: 从模型返回的对象检测结果
        :param frame:
        :return:
        """
        cord, scores, labels = results
        x_shape, y_shape = frame.shape[1], frame.shape[0]
        n = len(labels)  # 检测到的对象（实例）数

        for i in range(n):
            row = cord[i]
            x1, y1, x2, y2 = int(row[0] * x_shape), int(row[1] * y_shape), int(row[2] * x_shape), int(row[3] * y_shape)
            green_bgr = (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), green_bgr, 1)
            cls = labels[i]
            cls = int(cls)
            global cls_name
            if cls == 2:
                cls_name = 'car'
            if cls == 0:
                cls_name = 'person'
            cv2.putText(frame, cls_name, (x1 + 35, y1), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 0, 0), 1)

        return frame

    def calc_distances(self, results, frame):
        """
        计算距离
            :param results: 通过检测对象获得的结果值
        :param frame: 要处理的帧
        :return: 执行距离计算并处理的图像
        """
        cord, scores, labels = results
        x_shape, y_shape = frame.shape[1], frame.shape[0]
        points = []

        for car in cord:
            x1, y1, x2, y2 = int(car[0] * x_shape), int(car[1] * y_shape), int(car[2] * x_shape), int(
                car[3] * y_shape)  # 位置
            x_mid_rect, y_mid_rect = (x1 + x2) / 2, (y1 + y2) / 2  # 轴的中点
            y_line_length, x_line_length = abs(y1 - y2), abs(x1 - x2)  # 轴的长度
            #  cv2.circle(frame, center=(int(x_mid_rect), int(y_mid_rect)), radius=1, color=(0, 0, 255), thickness=5)
            points.append([x1, y1, x2, y2, int(x_mid_rect), int(y_mid_rect), int(x_line_length), int(y_line_length)])

        x_shape_mid = int(x_shape / 2)
        start_x, start_y = x_shape_mid, y_shape
        start_point = (start_x, start_y)

        heigth_in_rf = 121
        measured_distance = 275  # inch = 700cm
        real_heigth = 60  # inch = 150 cm
        focal_length = (heigth_in_rf * measured_distance) / real_heigth

        pixel_per_cm = float(2200 / x_shape) * 2.54

        for i in range(0, len(points)):
            end_x1, end_y1, end_x2, end_y2, end_x_mid_rect, end_y_mid_rect, end_x_line_length, end_y_line_length = \
            points[i]
            if end_x2 < x_shape_mid:  # 如果在左侧
                end_point = (end_x2, end_y2)  # 右下角选择
            elif end_x1 > x_shape_mid:  # 如果在右侧
                end_point = (end_x1, end_y2)  # 左下角选择
            else:  #如果在中间
                end_point = (end_x_mid_rect, end_y2)  # 选择底部中间

            dif_x, dif_y = abs(start_point[0] - end_point[0]), abs(start_point[1] - end_point[1])
            pixel_count = math.sqrt(math.pow(dif_x, 2) + math.pow(dif_y, 2))
            global distance
            distance = float(pixel_count * pixel_per_cm / end_y_line_length)

            # distance = real_heigth * focal_length / abs(end_y1 - end_y2);
            # distance = distance * 2.54 / 100
            #  print(distance)
            #cv2.line(frame, start_point, end_point, color=(0, 0, 255), thickness=1)
            cv2.putText(frame, str(round(distance, 2)) + " m", (int(end_x1), int(end_y2)), cv2.FONT_HERSHEY_DUPLEX,
                        0.5, (255, 255, 255), 2)
            cv2.putText(frame, str(int(scores[i] * 100)) + "%", (int(end_x1), int(end_y1)), cv2.FONT_HERSHEY_DUPLEX,
                        0.5, (255, 255, 0), 2)
        return frame

    def __call__(self):
        cap = self.get_video_capture()
        assert cap.isOpened()

        while True:
            ret, frame = cap.read()
            if not ret:
                break  # 如果无法读取视频帧，则结束循环q
            cap_h, cap_w, _ = frame.shape
            start_time = time()
            results = self.get_model_results(frame)
            frame = self.draw_rect(results, frame)
            frame = self.calc_distances(results, frame)
            end_time = time()
            fps = 1 / np.round(end_time - start_time, 2)
            print('{0}x{1}, {2}, Inference : {3}ms, Distance : {4}m'.format(cap_w, cap_h, cls_name,
                                                                            round(fps / 1000 * 100, 3),
                                                                            round(distance, 2)))
            cv2.putText(frame, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
            cv2.imshow('YOLOv5 Distance Estimation', frame)
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    # dir_path = r'ultralytics\assets'
    # img_list = os.listdir(dir_path)
    # for img_name in img_list:
    #     model.predict(os.path.join(dir_path, img_name), save=True, show=False)

    detector = DistanceEstimationDetector(video_path='input/car_input1.mp4', model_path='yolov8n.pt')
    detector()

