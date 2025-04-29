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
            weights_file = 'Models/sppe/fast_res101_320x256.pth'
            try:
                self.model = InferenNet_fast(weights_file, device).to(device)
            except Exception as e:
                print(f"포즈 모델(ResNet101) 로딩 실패: {e}")
                try:
                    print("다른 방법으로 포즈 모델 로딩 시도...")
                    self.model = InferenNet_fast().to(device)
                    with open(weights_file, 'rb') as f:
                        state_dict = torch.load(f, map_location=device)
                        self.model.load_state_dict(state_dict)
                    print("성공적으로 모델을 로드했습니다.")
                except Exception as e2:
                    print(f"포즈 모델 로딩 재시도 실패: {e2}")
                    raise RuntimeError("포즈 모델 로딩 실패")
        else:
            weights_file = 'Models/sppe/fast_res50_256x192.pth'
            try:
                self.model = InferenNet_fastRes50(weights_file, device).to(device)
            except Exception as e:
                print(f"포즈 모델(ResNet50) 로딩 실패: {e}")
                try:
                    print("다른 방법으로 포즈 모델 로딩 시도...")
                    # 모델 구조 생성 후 가중치 로딩 시도
                    self.model = InferenNet_fastRes50().to(device)
                    with open(weights_file, 'rb') as f:
                        state_dict = torch.load(f, map_location=device)
                        self.model.load_state_dict(state_dict)
                    print("성공적으로 모델을 로드했습니다.")
                except Exception as e2:
                    print(f"포즈 모델 로딩 재시도 실패: {e2}")
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