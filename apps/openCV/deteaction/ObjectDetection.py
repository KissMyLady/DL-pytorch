# coding:utf-8
# Author:mylady
# Datetime:2023/3/26 13:46
import cv2
import numpy as np


class ObjectDetection:

    def __init__(self, nmsThreshold=0.4, confThreshold=0.5,
                 weights_path="../data/weight/dnn_model/yolov4.weights",
                 cfg_path="../data/weight/dnn_model/yolov4.cfg"):
        print("Loading Object Detection")
        print("Running opencv dnn with YOLOv4")
        self.nmsThreshold = nmsThreshold
        self.confThreshold = confThreshold
        self.image_size = 608

        # Load Network
        net = cv2.dnn.readNet(weights_path, cfg_path)

        # Enable GPU CUDA
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        self.model = cv2.dnn_DetectionModel(net)

        self.classes = []
        self.load_class_names()
        self.colors = np.random.uniform(0, 255, size=(80, 3))

        self.model.setInputParams(size=(self.image_size, self.image_size), scale=1 / 255)

    def load_class_names(self, classes_path="../data/weight/dnn_model/classes.txt"):
        with open(classes_path, "r") as file_object:
            for class_name in file_object.readlines():
                class_name = class_name.strip()
                self.classes.append(class_name)

        self.colors = np.random.uniform(0, 255, size=(80, 3))
        return self.classes

    def getObjectName(self, objId):
        return self.classes[objId]

    def detect(self, frame):
        """
        :param frame: 每帧的图像
        Threshold: 临界值
        confThreshold: 置信度
        """
        return self.model.detect(frame, nmsThreshold=self.nmsThreshold, confThreshold=self.confThreshold)
