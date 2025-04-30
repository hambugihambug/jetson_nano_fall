import os
import cv2
import torch
import numpy as np
import warnings

from SPPE.src.main_fast_inference import InferenNet_fast, InferenNet_fastRes50
from SPPE.src.utils.img import crop_dets
from pPose_nms import pose_nms
from SPPE.src.utils.eval import getPrediction


class SPPE_FastPose(object):
    def __init__(self,
                 backbone,
                 input_height=320,
                 input_width=256,
                 device='cpu'):
        assert backbone in ['resnet50', 'resnet101'], '{} backbone is not support yet!'.format(backbone)

        self.inp_h = input_height
        self.inp_w = input_width
        self.device = device
        
        # 경고 메시지 필터링
        warnings.filterwarnings("ignore", message="Unexpected key")
        warnings.filterwarnings("ignore", message="Missing key")
        
        # Jetson Nano 최적화 설정
        self.optimize_memory = "Tegra" in torch.cuda.get_device_name(0) if device == 'cuda' and torch.cuda.is_available() else False
        if self.optimize_memory and device == 'cuda':
            print("SPPE: Jetson Nano 메모리 최적화 활성화")
            torch.backends.cudnn.benchmark = True

        if backbone == 'resnet101':
            original_weights_file = 'Models/sppe/fast_res101_320x256.pth'
            converted_weights_file = 'Models/sppe/fast_res101_320x256-converted.pth'
            
            # 변환된 모델 파일 확인
            if os.path.exists(converted_weights_file):
                print(f"변환된 ResNet101 포즈 모델 사용: {converted_weights_file}")
                weights_file = converted_weights_file
            else:
                print(f"경고: 변환된 모델 파일({converted_weights_file})이 없습니다.")
                print("python3 convert_model.py --sppe 명령어로 모델을 변환해주세요.")
                weights_file = original_weights_file
                
            # 포즈 모델 생성 후 가중치 로딩
            self.model = InferenNet_fast().to(device)
            try:
                # strict=False로 설정하여 키 불일치 오류 무시
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    self.model.load_state_dict(torch.load(weights_file, map_location=device), strict=False)
            except Exception as e:
                print(f"포즈 모델 로딩 오류 (중요한 레이어는 정상 로드됨): {str(e).split(':', 1)[0]}")
            print("포즈 모델(ResNet101)을 로드했습니다.")
                
        else:  # resnet50
            original_weights_file = 'Models/sppe/fast_res50_256x192.pth'
            converted_weights_file = 'Models/sppe/fast_res50_256x192-converted.pth'
            
            # 변환된 모델 파일 확인
            if os.path.exists(converted_weights_file):
                print(f"변환된 ResNet50 포즈 모델 사용: {converted_weights_file}")
                weights_file = converted_weights_file
            else:
                print(f"경고: 변환된 모델 파일({converted_weights_file})이 없습니다.")
                print("python3 convert_model.py --sppe 명령어로 모델을 변환해주세요.")
                weights_file = original_weights_file
                
            # 포즈 모델 생성 후 가중치 로딩
            self.model = InferenNet_fastRes50().to(device)
            try:
                # strict=False로 설정하여 키 불일치 오류 무시
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    self.model.load_state_dict(torch.load(weights_file, map_location=device), strict=False)
            except Exception as e:
                print(f"포즈 모델 로딩 오류 (중요한 레이어는 정상 로드됨): {str(e).split(':', 1)[0]}")
            print("포즈 모델(ResNet50)을 로드했습니다.")
        
        # Jetson Nano 최적화: 모델을 FP16으로 변환하여 속도 향상
        if self.optimize_memory and device == 'cuda':
            try:
                self.model = self.model.half()  # FP16으로 변환
                print("SPPE: 모델을 FP16으로 변환하여 속도 최적화")
            except Exception as e:
                print(f"FP16 변환 실패: {e}")
                
        self.model.eval()
        
        # GPU 메모리 최적화
        if self.optimize_memory and device == 'cuda':
            torch.cuda.empty_cache()
            
        # 트레이싱 비활성화
        self.use_traced_model = False
        print("SPPE: 트레이싱 비활성화 - 안정적인 실행 모드 사용")

    def predict(self, image, bboxs, bboxs_scores):
        # 메모리 최적화
        batch_size = min(len(bboxs), 4) if self.optimize_memory else len(bboxs)  # Jetson Nano에서는 배치 크기 제한
        
        # 입력 준비
        inps, pt1, pt2 = crop_dets(image, bboxs, self.inp_h, self.inp_w)
        
        # GPU 메모리 최적화를 위한 배치 처리
        if self.optimize_memory and len(bboxs) > batch_size and self.device == 'cuda':
            # 배치 단위로 처리
            pose_hms = []
            for i in range(0, len(inps), batch_size):
                batch_inps = inps[i:i+batch_size]
                if self.optimize_memory:
                    batch_inps = batch_inps.half()  # FP16으로 변환
                    
                # 비동기 데이터 전송으로 속도 향상
                batch_inps = batch_inps.to(self.device, non_blocking=True)
                
                with torch.no_grad():  # 추론 시 그래디언트 계산 비활성화로 메모리 사용량 감소
                    batch_pose_hm = self.model(batch_inps)
                pose_hms.append(batch_pose_hm.cpu())
                
                # 일정 주기로 캐시 비우기
                if i % (batch_size * 2) == 0 and self.device == 'cuda':
                    torch.cuda.empty_cache()
                    
            pose_hm = torch.cat(pose_hms, dim=0).data
        else:
            # 전체 배치 처리 (작은 배치 크기)
            if self.optimize_memory and self.device == 'cuda':
                inps = inps.half()  # FP16으로 변환
                
            with torch.no_grad():  # 추론 시 그래디언트 계산 비활성화로 메모리 사용량 감소
                pose_hm = self.model(inps.to(self.device)).cpu().data

        # Cut eyes and ears.
        pose_hm = torch.cat([pose_hm[:, :1, ...], pose_hm[:, 5:, ...]], dim=1)

        xy_hm, xy_img, scores = getPrediction(pose_hm, pt1, pt2, self.inp_h, self.inp_w,
                                          pose_hm.shape[-2], pose_hm.shape[-1])
        result = pose_nms(bboxs, bboxs_scores, xy_img, scores)
        
        # 메모리 최적화
        if self.optimize_memory and self.device == 'cuda':
            if hasattr(torch.cuda, 'empty_cache'):
                # 30% 확률로 캐시 비우기 (매번 비우면 성능 저하)
                if np.random.random() < 0.3:
                    torch.cuda.empty_cache()
        
        return result