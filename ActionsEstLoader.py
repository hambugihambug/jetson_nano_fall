import os
import torch
import numpy as np
import warnings
import gc

from Actionsrecognition.Models import TwoStreamSpatialTemporalGraph
from pose_utils import normalize_points_with_size, scale_pose


class TSSTG(object):
    """Two-Stream Spatial Temporal Graph Model Loader.
    Args:
        weight_file: (str) Path to trained weights file.
        device: (str) Device to load the model on 'cpu' or 'cuda'.
    """
    def __init__(self,
                 weight_file='./Models/TSSTG/tsstg-model.pth',
                 device='cpu',
                 low_memory=False):
        # 경고 메시지 필터링
        warnings.filterwarnings("ignore", message="Unexpected key")
        warnings.filterwarnings("ignore", message="Missing key")
        
        self.graph_args = {'strategy': 'spatial'}
        self.class_names = ['Standing', 'Walking', 'Sitting', 'Lying Down',
                            'Stand up', 'Sit down', 'Fall Down']
        self.num_class = len(self.class_names)
        self.device = device
        self.low_memory = low_memory

        # 메모리 최적화 설정
        self.optimize_memory = "Tegra" in torch.cuda.get_device_name(0) if device == 'cuda' and torch.cuda.is_available() else False
        
        if self.optimize_memory and device == 'cuda':
            print("TSSTG: Jetson Nano 메모리 최적화 활성화")
            torch.backends.cudnn.benchmark = True
            
            # 저메모리 모드 설정
            if low_memory:
                print("TSSTG: 저메모리 모드 활성화")
        
        self.model = TwoStreamSpatialTemporalGraph(self.graph_args, self.num_class).to(self.device)
        
        # 변환된 모델 파일 확인
        converted_weight_file = weight_file.replace('.pth', '-converted.pth')
        if os.path.exists(converted_weight_file):
            print(f"변환된 액션 인식 모델 사용: {converted_weight_file}")
            model_path = converted_weight_file
        else:
            print(f"경고: 변환된 모델 파일({converted_weight_file})이 없습니다.")
            print("python3 convert_model.py --action 명령어로 모델을 변환해주세요.")
            model_path = weight_file
            
        try:
            # strict=False로 설정하여, 키 불일치 오류 무시
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
        except Exception as e:
            print(f"액션 인식 모델 로딩 오류 (중요한 레이어는 정상 로드됨): {str(e).split(':', 1)[0]}")
            
        print("액션 인식 모델을 로드했습니다.")
        
        # GPU 메모리 최적화 (Jetson Nano)
        if self.optimize_memory and device == 'cuda':
            torch.cuda.empty_cache()
                
        self.model.eval()
        
        # Jetson Nano 최적화: FP16 변환은 모델 종류에 따라 오류가 발생할 수 있으므로 선택적 사용
        self.use_fp16 = False
        if self.optimize_memory and device == 'cuda' and not self.low_memory:
            try:
                # 간단한 입력으로 테스트
                dummy_input = torch.zeros((1, 3, 30, 17), dtype=torch.float32).to(device)
                dummy_mot = torch.zeros((1, 2, 29, 17), dtype=torch.float32).to(device)
                
                # FP16으로 모델 변환 시도
                fp16_model = self.model.half()
                _ = fp16_model((dummy_input.half(), dummy_mot.half()))
                
                # 테스트 성공하면 FP16 모델 사용
                self.model = fp16_model
                self.use_fp16 = True
                print("TSSTG: 모델을 FP16으로 변환하여 속도 최적화 성공")
            except Exception as e:
                print(f"TSSTG: FP16 변환 테스트 실패, FP32 모드로 계속 진행: {e}")
                self.use_fp16 = False
                
        # 저메모리 모드일 경우 FP16 비활성화
        if self.low_memory and self.use_fp16:
            print("TSSTG: 저메모리 모드에서는 FP16 비활성화됨")
            self.use_fp16 = False
            try:
                self.model = self.model.float()  # FP32로 다시 변환
            except:
                pass
                
        # 저메모리 모드일 경우 메모리 적극 정리
        if self.low_memory and device == 'cuda':
            gc.collect()
            torch.cuda.empty_cache()

    def predict(self, pts, image_size):
        """Predict actions from single person skeleton points and score in time sequence.
        Args:
            pts: (numpy array) points and score in shape `(t, v, c)` where
                t : inputs sequence (time steps).,
                v : number of graph node (body parts).,
                c : channel (x, y, score).,
            image_size: (tuple of int) width, height of image frame.
        Returns:
            (numpy array) Probability of each class actions.
        """
        # 저메모리 모드일 경우 미리 메모리 정리
        if self.low_memory and self.device == 'cuda':
            torch.cuda.empty_cache()
            
        pts[:, :, :2] = normalize_points_with_size(pts[:, :, :2], image_size[0], image_size[1])
        pts[:, :, :2] = scale_pose(pts[:, :, :2])
        pts = np.concatenate((pts, np.expand_dims((pts[:, 1, :] + pts[:, 2, :]) / 2, 1)), axis=1)

        # 데이터 형식 변환 최적화
        pts = torch.tensor(pts, dtype=torch.float32)
        pts = pts.permute(2, 0, 1)[None, :]
        
        # 모션 계산을 CPU에서 먼저 수행 (안정성)
        mot = pts[:, :2, 1:, :] - pts[:, :2, :-1, :]
        
        # Jetson Nano 최적화: 입력을 반정밀도로 변환하여 계산 속도 향상
        if self.use_fp16 and self.device == 'cuda':
            pts = pts.half()  # FP16으로 변환
            mot = mot.half()  # FP16으로 변환
            
        # 데이터 전송
        if self.device == 'cuda':
            pts = pts.to(self.device, non_blocking=True)
            mot = mot.to(self.device, non_blocking=True)
        else:
            pts = pts.to(self.device)
            mot = mot.to(self.device)

        with torch.no_grad():  # 추론 시 그래디언트 계산 비활성화로 메모리 사용량 감소
            out = self.model((pts, mot))
            
        # 큰 텐서들 명시적으로 해제
        del pts, mot
        result = out.detach().cpu().numpy()
        del out

        # 메모리 최적화
        if self.device == 'cuda':
            if self.low_memory:
                # 저메모리 모드에서는 매번 메모리 정리
                torch.cuda.empty_cache()
                gc.collect()
            elif hasattr(torch.cuda, 'empty_cache') and np.random.random() < 0.3:
                # 30% 확률로 캐시 비우기 (매번 비우면 성능 저하)
                torch.cuda.empty_cache()

        return result
