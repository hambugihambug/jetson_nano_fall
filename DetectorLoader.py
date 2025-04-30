import time
import torch
import numpy as np
import torchvision.transforms as transforms
import os
import warnings

from queue import Queue
from threading import Thread

from Detection.Models import Darknet
from Detection.Utils import non_max_suppression, ResizePadding


class TinyYOLOv3_onecls(object):
    """Load trained Tiny-YOLOv3 one class (person) detection model.
    Args:
        input_size: (int) Size of input image must be divisible by 32. Default: 416,
        config_file: (str) Path to Yolo model structure config file.,
        weight_file: (str) Path to trained weights file.,
        nms: (float) Non-Maximum Suppression overlap threshold.,
        conf_thres: (float) Minimum Confidence threshold of predicted bboxs to cut off.,
        device: (str) Device to load the model on 'cpu' or 'cuda'.
    """
    def __init__(self,
                 input_size=416,
                 config_file='Models/yolo-tiny-onecls/yolov3-tiny-onecls.cfg',
                 weight_file='Models/yolo-tiny-onecls/best-model.pth',
                 nms=0.2,
                 conf_thres=0.45,
                 device='cpu'):
        # 경고 메시지 필터링
        warnings.filterwarnings("ignore", message="Unexpected key")
        warnings.filterwarnings("ignore", message="Missing key")
        
        self.input_size = input_size
        self.device = device
        
        # Jetson Nano 최적화 설정
        self.optimize_memory = "Tegra" in torch.cuda.get_device_name(0) if device == 'cuda' and torch.cuda.is_available() else False
        if self.optimize_memory and device == 'cuda':
            print("YOLO: Jetson Nano 메모리 최적화 활성화")
            torch.backends.cudnn.benchmark = True
        
        self.model = Darknet(config_file).to(device)
        
        # 변환된 모델 파일 확인
        converted_weight_file = weight_file.replace('.pth', '-converted.pth')
        if os.path.exists(converted_weight_file):
            print(f"변환된 객체 감지 모델 사용: {converted_weight_file}")
            weights_file = converted_weight_file
        else:
            print(f"경고: 변환된 모델 파일({converted_weight_file})이 없습니다.")
            print("python3 convert_model.py --detect 명령어로 모델을 변환해주세요.")
            weights_file = weight_file
            
        try:
            # strict=False로 설정하여, 키 불일치 오류 무시
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.model.load_state_dict(torch.load(weights_file, map_location=device), strict=False)
            print("객체 감지 모델을 로드했습니다.")
        except Exception as e:
            print(f"객체 감지 모델 로딩 오류: {str(e).split(':', 1)[0]}")
            raise RuntimeError("객체 감지 모델을 로드할 수 없습니다. 모델을 변환해주세요.")
            
        # Jetson Nano 최적화: 모델을 FP16으로 변환하여 속도 향상
        if self.optimize_memory and device == 'cuda':
            try:
                self.model = self.model.half()  # FP16으로 변환
                print("YOLO: 모델을 FP16으로 변환하여 속도 최적화")
            except Exception as e:
                print(f"FP16 변환 실패: {e}")
            
        self.model.eval()
        
        # GPU 메모리 최적화
        if self.optimize_memory and device == 'cuda':
            torch.cuda.empty_cache()
            
        # YOLO 모델은 동적 계산 그래프를 사용하므로 트레이싱 비활성화
        self.use_traced_model = False
        print("YOLO: 동적 계산 그래프 사용으로 트레이싱 비활성화")

        self.nms = nms
        self.conf_thres = conf_thres

        self.resize_fn = ResizePadding(input_size, input_size)
        self.transf_fn = transforms.ToTensor()

    def detect(self, image, need_resize=True, expand_bb=5):
        """Feed forward to the model.
        Args:
            image: (numpy array) Single RGB image to detect.,
            need_resize: (bool) Resize to input_size before feed and will return bboxs
                with scale to image original size.,
            expand_bb: (int) Expand boundary of the boxs.
        Returns:
            (torch.float32) Of each detected object contain a
                [top, left, bottom, right, bbox_score, class_score, class]
            return `None` if no detected.
        """
        image_size = (self.input_size, self.input_size)
        if need_resize:
            image_size = image.shape[:2]
            image = self.resize_fn(image)

        image_tensor = self.transf_fn(image)[None, ...]
        
        # Jetson Nano 최적화 적용
        if self.optimize_memory and self.device == 'cuda':
            image_tensor = image_tensor.half()  # FP16으로 변환
        
        # 배치 처리
        scf = torch.min(self.input_size / torch.FloatTensor([image_size]), 1)[0]
        
        with torch.no_grad():  # 추론 시 그래디언트 계산 비활성화로 메모리 사용량 감소
            # 비동기 데이터 전송으로 속도 향상
            if self.optimize_memory:
                image_tensor = image_tensor.to(self.device, non_blocking=True)
                detected = self.model(image_tensor)
            else:
                detected = self.model(image_tensor.to(self.device))
                
        detected = non_max_suppression(detected, self.conf_thres, self.nms)[0]
        
        if detected is not None:
            detected[:, [0, 2]] -= (self.input_size - scf * image_size[1]) / 2
            detected[:, [1, 3]] -= (self.input_size - scf * image_size[0]) / 2
            detected[:, 0:4] /= scf

            detected[:, 0:2] = np.maximum(0, detected[:, 0:2] - expand_bb)
            detected[:, 2:4] = np.minimum(image_size[::-1], detected[:, 2:4] + expand_bb)
            
        # 메모리 최적화
        if self.optimize_memory and self.device == 'cuda':
            if hasattr(torch.cuda, 'empty_cache'):
                # 30% 확률로 캐시 비우기 (매번 비우면 성능 저하)
                if np.random.random() < 0.3:
                    torch.cuda.empty_cache()

        return detected


class ThreadDetection(object):
    def __init__(self,
                 dataloader,
                 model,
                 queue_size=256):
        self.model = model

        self.dataloader = dataloader
        self.stopped = False
        self.Q = Queue(maxsize=queue_size)

    def start(self):
        t = Thread(target=self.update, args=(), daemon=True).start()
        return self

    def update(self):
        while True:
            if self.stopped:
                return

            images = self.dataloader.getitem()

            outputs = self.model.detect(images)

            if self.Q.full():
                time.sleep(2)
            self.Q.put((images, outputs))

    def getitem(self):
        return self.Q.get()

    def stop(self):
        self.stopped = True

    def __len__(self):
        return self.Q.qsize()







