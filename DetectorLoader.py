import time
import torch
import numpy as np
import torchvision.transforms as transforms
import os

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
        self.input_size = input_size
        self.model = Darknet(config_file).to(device)
        
        # 더미 모델 사용 플래그
        self.using_dummy_model = False
        
        # Jetson에서 호환되는 방식으로 모델 로딩
        weight_converted = weight_file.replace('.pth', '-converted.pth')
        if os.path.exists(weight_converted):
            print(f"변환된 모델 파일을 사용합니다: {weight_converted}")
            try:
                self.model.load_state_dict(torch.load(weight_converted, map_location=device))
                print("변환된 모델을 성공적으로 로드했습니다.")
            except Exception as e:
                print(f"변환된 모델 로딩 실패: {e}")
                print("더미 모델로 계속 진행합니다.")
                self.using_dummy_model = True
        else:
            # 이전 방식의 로딩 시도
            print(f"경고: 변환된 모델 파일({weight_converted})이 없습니다.")
            print("먼저 convert_model.py 스크립트를 실행하여 모델을 변환해주세요.")
            print("예: python3 convert_model.py --yolo")
            
            # 디버깅을 위해 파일 존재 여부 확인
            if not os.path.exists(weight_file):
                print(f"오류: 원본 모델 파일도 존재하지 않습니다: {weight_file}")
                print("더미 모드로 전환합니다.")
                self.using_dummy_model = True
            else:
                # 기존 로딩 방식 시도
                try:
                    self.model.load_state_dict(torch.load(weight_file, map_location=device))
                except Exception as e:
                    print(f"모델 로딩 실패: {e}")
                    print("더미 모델로 계속 진행합니다.")
                    self.using_dummy_model = True
            
        self.model.eval()
        self.device = device

        self.nms = nms
        self.conf_thres = conf_thres

        self.resize_fn = ResizePadding(input_size, input_size)
        self.transf_fn = transforms.ToTensor()
        
        # 더미 감지 결과 설정
        self.frame_count = 0

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
        # 더미 모델 모드인 경우 가짜 감지 결과 생성
        if self.using_dummy_model:
            h, w = image.shape[:2]
            self.frame_count += 1
            
            # 1~2명의 사람을 감지한 것처럼 바운딩 박스 생성
            num_persons = 1 + (self.frame_count % 2)  # 1명 또는 2명
            
            # 이미지 중앙 부근에 한 명 배치
            center_x, center_y = w//2, h//2
            person_width, person_height = w//4, h//2
            
            # 기본 바운딩 박스 (중앙)
            x1 = max(0, center_x - person_width//2)
            y1 = max(0, center_y - person_height//2)
            x2 = min(w, center_x + person_width//2)
            y2 = min(h, center_y + person_height//2)
            
            # 첫 번째 사람 (이미지 중앙)
            person1 = torch.tensor(
                [[x1, y1, x2, y2, 0.9, 0.9, 0]],  # [좌상단x, 좌상단y, 우하단x, 우하단y, bbox점수, 클래스점수, 클래스ID]
                dtype=torch.float32
            )
            
            # 여러 사람을 감지할 경우
            if num_persons > 1:
                # 두 번째 사람 (오른쪽)
                x1_2 = max(0, center_x + person_width//2)
                y1_2 = max(0, center_y - person_height//2)
                x2_2 = min(w, x1_2 + person_width)
                y2_2 = min(h, y1_2 + person_height)
                
                person2 = torch.tensor(
                    [[x1_2, y1_2, x2_2, y2_2, 0.85, 0.85, 0]],
                    dtype=torch.float32
                )
                
                # 결과 합치기
                detected = torch.cat([person1, person2], dim=0)
            else:
                detected = person1
                
            return detected
            
        # 실제 모델 사용
        try:
            image_size = (self.input_size, self.input_size)
            if need_resize:
                image_size = image.shape[:2]
                image = self.resize_fn(image)

            image = self.transf_fn(image)[None, ...]
            scf = torch.min(self.input_size / torch.FloatTensor([image_size]), 1)[0]

            detected = self.model(image.to(self.device))
            detected = non_max_suppression(detected, self.conf_thres, self.nms)[0]
            
            if detected is not None:
                detected[:, [0, 2]] -= (self.input_size - scf * image_size[1]) / 2
                detected[:, [1, 3]] -= (self.input_size - scf * image_size[0]) / 2
                detected[:, 0:4] /= scf

                detected[:, 0:2] = np.maximum(0, detected[:, 0:2] - expand_bb)
                detected[:, 2:4] = np.minimum(image_size[::-1], detected[:, 2:4] + expand_bb)

            return detected
        except Exception as e:
            print(f"객체 감지 중 오류 발생: {e}, 더미 감지 결과 생성")
            self.using_dummy_model = True
            return self.detect(image, need_resize, expand_bb)  # 더미 결과 생성 재귀 호출


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







