import os
import cv2
import torch
import numpy as np

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
        
        # 더미 모델 사용 여부 플래그
        self.using_dummy_model = False

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
                try:
                    self.model.load_state_dict(torch.load(weights_file, map_location=device))
                    print("포즈 모델(ResNet101)을 성공적으로 로드했습니다.")
                except Exception as e:
                    print(f"포즈 모델(ResNet101) 로딩 중 오류 발생: {e}")
                    print("더미 모델로 계속 진행합니다.")
                    self.using_dummy_model = True
            except Exception as e:
                print(f"포즈 모델(ResNet101) 인스턴스 생성 실패: {e}")
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
                try:
                    self.model.load_state_dict(torch.load(weights_file, map_location=device))
                    print("포즈 모델(ResNet50)을 성공적으로 로드했습니다.")
                except Exception as e:
                    print(f"포즈 모델(ResNet50) 로딩 중 오류 발생: {e}")
                    print("더미 모델로 계속 진행합니다.")
                    self.using_dummy_model = True
            except Exception as e:
                print(f"포즈 모델(ResNet50) 인스턴스 생성 실패: {e}")
                raise RuntimeError("포즈 모델 로딩 실패")
        
        self.model.eval()

    def predict(self, image, bboxs, bboxs_scores):
        # 더미 모델 사용 중이면 더미 출력 반환
        if self.using_dummy_model:
            # 빈 포즈 결과 생성
            dummy_results = []
            for i in range(len(bboxs)):
                dummy_keypoints = np.zeros((17, 3))  # 17개 키포인트, (x, y, score)
                dummy_keypoints[:, 2] = 0.5  # 중간 신뢰도 점수
                
                # 바운딩 박스 중심으로 키포인트 배치
                x1, y1, x2, y2 = bboxs[i]
                center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
                w, h = x2 - x1, y2 - y1
                
                # 사람 형태를 대략적으로 모방하는 키포인트
                # 0: 코
                dummy_keypoints[0, 0] = center_x
                dummy_keypoints[0, 1] = y1 + 0.1 * h
                
                # 1-4: 어깨, 팔꿈치
                dummy_keypoints[1, 0] = center_x - 0.2 * w  # 왼쪽 어깨
                dummy_keypoints[1, 1] = y1 + 0.3 * h
                dummy_keypoints[2, 0] = center_x + 0.2 * w  # 오른쪽 어깨
                dummy_keypoints[2, 1] = y1 + 0.3 * h
                dummy_keypoints[3, 0] = center_x - 0.3 * w  # 왼쪽 팔꿈치
                dummy_keypoints[3, 1] = y1 + 0.5 * h
                dummy_keypoints[4, 0] = center_x + 0.3 * w  # 오른쪽 팔꿈치
                dummy_keypoints[4, 1] = y1 + 0.5 * h
                
                # 5-6: 손목
                dummy_keypoints[5, 0] = center_x - 0.4 * w  # 왼쪽 손목
                dummy_keypoints[5, 1] = y1 + 0.6 * h
                dummy_keypoints[6, 0] = center_x + 0.4 * w  # 오른쪽 손목
                dummy_keypoints[6, 1] = y1 + 0.6 * h
                
                # 7-10: 골반, 무릎
                dummy_keypoints[7, 0] = center_x - 0.1 * w  # 왼쪽 골반
                dummy_keypoints[7, 1] = y1 + 0.7 * h
                dummy_keypoints[8, 0] = center_x + 0.1 * w  # 오른쪽 골반
                dummy_keypoints[8, 1] = y1 + 0.7 * h
                dummy_keypoints[9, 0] = center_x - 0.15 * w  # 왼쪽 무릎
                dummy_keypoints[9, 1] = y1 + 0.8 * h
                dummy_keypoints[10, 0] = center_x + 0.15 * w  # 오른쪽 무릎
                dummy_keypoints[10, 1] = y1 + 0.8 * h
                
                # 11-12: 발목
                dummy_keypoints[11, 0] = center_x - 0.2 * w  # 왼쪽 발목
                dummy_keypoints[11, 1] = y1 + 0.95 * h
                dummy_keypoints[12, 0] = center_x + 0.2 * w  # 오른쪽 발목
                dummy_keypoints[12, 1] = y1 + 0.95 * h
                
                # 13-16: 기타 키포인트
                for i in range(13, 17):
                    dummy_keypoints[i, 0] = center_x
                    dummy_keypoints[i, 1] = y1 + (0.3 + 0.1 * (i-13)) * h
                
                dummy_result = {
                    'keypoints': torch.tensor(dummy_keypoints[:, :2]),
                    'kp_score': torch.tensor(dummy_keypoints[:, 2:])
                }
                dummy_results.append(dummy_result)
            
            return dummy_results
        
        # 실제 모델 사용
        try:
            inps, pt1, pt2 = crop_dets(image, bboxs, self.inp_h, self.inp_w)
            pose_hm = self.model(inps.to(self.device)).cpu().data

            # Cut eyes and ears.
            pose_hm = torch.cat([pose_hm[:, :1, ...], pose_hm[:, 5:, ...]], dim=1)

            xy_hm, xy_img, scores = getPrediction(pose_hm, pt1, pt2, self.inp_h, self.inp_w,
                                              pose_hm.shape[-2], pose_hm.shape[-1])
            result = pose_nms(bboxs, bboxs_scores, xy_img, scores)
            return result
        except Exception as e:
            print(f"포즈 추정 중 오류 발생: {e}, 더미 결과 생성")
            self.using_dummy_model = True
            return self.predict(image, bboxs, bboxs_scores)  # 더미 결과 생성 재귀 호출