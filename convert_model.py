#!/usr/bin/env python3
import os
import torch
import argparse
from Detection.Models import Darknet

def convert_model(input_file, output_file, config_file=None):
    """
    PyTorch 모델 파일을 로드하고 다시 저장하여 버전 호환성 문제 해결
    """
    print(f"모델 파일 변환 시작: {input_file} -> {output_file}")
    
    try:
        if config_file:
            # YOLO 모델의 경우 구조 로딩이 필요
            print(f"YOLO 모델 구조 로딩: {config_file}")
            model = Darknet(config_file)
            state_dict = torch.load(input_file, map_location='cpu')
            model.load_state_dict(state_dict)
            torch.save(model.state_dict(), output_file)
        else:
            # 그 외 다른 모델의 경우 직접 state_dict만 로드
            state_dict = torch.load(input_file, map_location='cpu')
            torch.save(state_dict, output_file)
        
        print(f"모델 변환 성공: {output_file}")
        return True
    except Exception as e:
        print(f"모델 변환 실패: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='PyTorch 모델 파일 변환 도구')
    parser.add_argument('--all', action='store_true', help='모든 모델 파일 변환')
    parser.add_argument('--yolo', action='store_true', help='YOLO 모델 파일 변환')
    parser.add_argument('--sppe', action='store_true', help='SPPE 모델 파일 변환')
    parser.add_argument('--action', action='store_true', help='TSSTG 모델 파일 변환')
    
    args = parser.parse_args()
    
    # 모든 모델 변환 모드
    if args.all:
        args.yolo = args.sppe = args.action = True
    
    # 모델 파일 경로
    yolo_model = 'Models/yolo-tiny-onecls/best-model.pth'
    yolo_config = 'Models/yolo-tiny-onecls/yolov3-tiny-onecls.cfg'
    sppe_res50 = 'Models/sppe/fast_res50_256x192.pth'
    sppe_res101 = 'Models/sppe/fast_res101_320x256.pth'
    action_model = 'Models/TSSTG/tsstg-model.pth'
    
    # 변환된 모델 저장 경로
    yolo_model_new = 'Models/yolo-tiny-onecls/best-model-converted.pth'
    sppe_res50_new = 'Models/sppe/fast_res50_256x192-converted.pth'
    sppe_res101_new = 'Models/sppe/fast_res101_320x256-converted.pth'
    action_model_new = 'Models/TSSTG/tsstg-model-converted.pth'
    
    # YOLO 모델 변환
    if args.yolo and os.path.exists(yolo_model):
        print("\n===== YOLO 모델 변환 =====")
        if convert_model(yolo_model, yolo_model_new, yolo_config):
            print(f"DetectorLoader.py 업데이트 필요: weight_file='{yolo_model_new}'")
    
    # SPPE 모델 변환
    if args.sppe:
        print("\n===== SPPE 모델 변환 =====")
        if os.path.exists(sppe_res50):
            if convert_model(sppe_res50, sppe_res50_new):
                print(f"PoseEstimateLoader.py 업데이트 필요: weights_file = '{sppe_res50_new}'")
        
        if os.path.exists(sppe_res101):
            if convert_model(sppe_res101, sppe_res101_new):
                print(f"PoseEstimateLoader.py 업데이트 필요: weights_file = '{sppe_res101_new}'")
    
    # Action 모델 변환
    if args.action and os.path.exists(action_model):
        print("\n===== TSSTG 액션 모델 변환 =====")
        if convert_model(action_model, action_model_new):
            print(f"ActionsEstLoader.py 업데이트 필요: weight_file='{action_model_new}'")
    
    print("\n모델 변환 작업 완료!")
    print("다음 명령어로 각 로더 파일을 수정하여 변환된 모델을 사용하세요.")

if __name__ == "__main__":
    main() 