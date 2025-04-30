import os
import torch
import numpy as np

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
                 device='cpu'):
        self.graph_args = {'strategy': 'spatial'}
        self.class_names = ['Standing', 'Walking', 'Sitting', 'Lying Down',
                            'Stand up', 'Sit down', 'Fall Down']
        self.num_class = len(self.class_names)
        self.device = device

        # 메모리 최적화 설정
        self.optimize_memory = "Tegra" in torch.cuda.get_device_name(0) if device == 'cuda' and torch.cuda.is_available() else False
        
        if self.optimize_memory and device == 'cuda':
            print("TSSTG: Jetson Nano 메모리 최적화 활성화")
            torch.backends.cudnn.benchmark = True
        
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
            
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        print("액션 인식 모델을 성공적으로 로드했습니다.")
        
        # GPU 메모리 최적화 (Jetson Nano)
        if self.optimize_memory and device == 'cuda':
            torch.cuda.empty_cache()
                
        self.model.eval()
        
        # Jetson Nano 최적화: 모델을 반정밀도(FP16)로 변환하여 속도 향상
        if self.optimize_memory and device == 'cuda':
            try:
                self.model = self.model.half()  # FP16으로 변환
                print("TSSTG: 모델을 FP16으로 변환하여 속도 최적화")
            except Exception as e:
                print(f"FP16 변환 실패: {e}")

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
        pts[:, :, :2] = normalize_points_with_size(pts[:, :, :2], image_size[0], image_size[1])
        pts[:, :, :2] = scale_pose(pts[:, :, :2])
        pts = np.concatenate((pts, np.expand_dims((pts[:, 1, :] + pts[:, 2, :]) / 2, 1)), axis=1)

        # 데이터 형식 변환 최적화
        pts = torch.tensor(pts, dtype=torch.float32)
        pts = pts.permute(2, 0, 1)[None, :]
        
        # Jetson Nano 최적화: 입력을 반정밀도로 변환하여 계산 속도 향상
        if self.optimize_memory and self.device == 'cuda':
            pts = pts.half()  # FP16으로 변환

        # 모션 계산을 CPU 또는 GPU에서 수행
        if self.device == 'cuda':
            # 데이터 전송 최적화: 모션 계산을 GPU에서 직접 수행
            mot = pts[:, :2, 1:, :] - pts[:, :2, :-1, :]
            mot = mot.to(self.device, non_blocking=True)  # 비동기 전송으로 속도 향상
            pts = pts.to(self.device, non_blocking=True)
        else:
            # CPU 계산
            mot = pts[:, :2, 1:, :] - pts[:, :2, :-1, :]
            mot = mot.to(self.device)
            pts = pts.to(self.device)

        with torch.no_grad():  # 추론 시 그래디언트 계산 비활성화로 메모리 사용량 감소
            out = self.model((pts, mot))

        # 메모리 최적화
        if self.optimize_memory and self.device == 'cuda':
            if hasattr(torch.cuda, 'empty_cache'):
                # 30% 확률로 캐시 비우기 (매번 비우면 성능 저하)
                if np.random.random() < 0.3:
                    torch.cuda.empty_cache()

        return out.detach().cpu().numpy()
