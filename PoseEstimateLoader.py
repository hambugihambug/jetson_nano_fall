import os
import cv2
import torch

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
                
            try:
                # 포즈 모델 생성 후 가중치 로딩
                self.model = InferenNet_fast().to(device)
                self.model.load_state_dict(torch.load(weights_file, map_location=device))
                print("포즈 모델(ResNet101)을 성공적으로 로드했습니다.")
            except Exception as e:
                print(f"포즈 모델(ResNet101) 로딩 실패: {e}")
                raise RuntimeError("포즈 모델 로딩 실패")
                
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
                
            try:
                # 포즈 모델 생성 후 가중치 로딩
                self.model = InferenNet_fastRes50().to(device)
                self.model.load_state_dict(torch.load(weights_file, map_location=device))
                print("포즈 모델(ResNet50)을 성공적으로 로드했습니다.")
            except Exception as e:
                print(f"포즈 모델(ResNet50) 로딩 실패: {e}")
                raise RuntimeError("포즈 모델 로딩 실패")
        
        self.model.eval()

    def predict(self, image, bboxs, bboxs_scores):
        inps, pt1, pt2 = crop_dets(image, bboxs, self.inp_h, self.inp_w)
        pose_hm = self.model(inps.to(self.device)).cpu().data

        # Cut eyes and ears.
        pose_hm = torch.cat([pose_hm[:, :1, ...], pose_hm[:, 5:, ...]], dim=1)

        xy_hm, xy_img, scores = getPrediction(pose_hm, pt1, pt2, self.inp_h, self.inp_w,
                                              pose_hm.shape[-2], pose_hm.shape[-1])
        result = pose_nms(bboxs, bboxs_scores, xy_img, scores)
        return result