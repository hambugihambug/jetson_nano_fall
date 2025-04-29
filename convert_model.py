#!/usr/bin/env python3
import os
import torch
import argparse
import pickle
import numpy as np
import warnings
from pathlib import Path

def convert_yolo_model(input_file, output_file, config_file):
    """YOLO 모델 변환 함수"""
    print(f"YOLO 모델 변환 시작: {input_file} -> {output_file}")
    
    from Detection.Models import Darknet
    
    try:
        # 모델 구조 로드
        model = Darknet(config_file)
        
        # 먼저 표준 방식으로 시도
        try:
            state_dict = torch.load(input_file, map_location='cpu')
            model.load_state_dict(state_dict)
        except Exception as e:
            print(f"표준 방식 로딩 실패: {e}")
            
            # 바이너리 모드로 시도
            with open(input_file, 'rb') as f:
                try:
                    state_dict = torch.load(f, map_location='cpu')
                    model.load_state_dict(state_dict)
                except Exception as e2:
                    print(f"바이너리 모드 로딩 실패: {e2}")
                    
                    # PyTorch 0.4 호환 방식 시도
                    try:
                        f.seek(0)
                        state_dict = pickle.load(f)
                        model.load_state_dict(state_dict)
                    except Exception as e3:
                        print(f"모든 로딩 방식 실패: {e3}")
                        return False
        
        # 모델 저장 (여러 방식으로 시도)
        try:
            # 방법 1: pickle 프로토콜 2 사용 (Python 2/3 호환)
            torch.save(model.state_dict(), output_file, _use_new_zipfile_serialization=False, pickle_protocol=2)
        except Exception as e:
            print(f"저장 방식 1 실패: {e}")
            try:
                # 방법 2: 구형 직렬화 방식 사용
                torch.save(model, output_file + '.full')
                state_dict = model.state_dict()
                with open(output_file, 'wb') as f:
                    pickle.dump(state_dict, f, protocol=2)
            except Exception as e2:
                print(f"저장 방식 2 실패: {e2}")
                return False
        
        print(f"YOLO 모델 변환 성공: {output_file}")
        return True
    except Exception as e:
        print(f"YOLO 모델 변환 중 오류 발생: {e}")
        return False

def convert_sppe_model(input_file, output_file):
    """SPPE 모델 변환 함수"""
    print(f"SPPE 모델 변환 시작: {input_file} -> {output_file}")
    
    try:
        # 여러 방식으로 로딩 시도
        try:
            state_dict = torch.load(input_file, map_location='cpu')
        except Exception as e:
            print(f"표준 방식 로딩 실패: {e}")
            
            # 바이너리 모드로 시도
            with open(input_file, 'rb') as f:
                try:
                    state_dict = torch.load(f, map_location='cpu')
                except Exception as e2:
                    print(f"바이너리 모드 로딩 실패: {e2}")
                    
                    # pickle 직접 로드 시도
                    try:
                        f.seek(0)
                        state_dict = pickle.load(f)
                    except Exception as e3:
                        print(f"모든 로딩 방식 실패: {e3}")
                        return False

        # 모델 저장 (여러 방식으로 시도)
        try:
            # 방법 1: pickle 프로토콜 2 사용 (Python 2/3 호환)
            torch.save(state_dict, output_file, _use_new_zipfile_serialization=False, pickle_protocol=2)
        except Exception as e:
            print(f"저장 방식 1 실패: {e}")
            try:
                # 방법 2: 구형 직렬화 방식 사용
                with open(output_file, 'wb') as f:
                    pickle.dump(state_dict, f, protocol=2)
            except Exception as e2:
                print(f"저장 방식 2 실패: {e2}")
                return False
                
        print(f"SPPE 모델 변환 성공: {output_file}")
        return True
    except Exception as e:
        print(f"SPPE 모델 변환 중 오류 발생: {e}")
        return False

def convert_action_model(input_file, output_file):
    """액션 인식 모델 변환 함수"""
    print(f"액션 인식 모델 변환 시작: {input_file} -> {output_file}")
    
    try:
        # 여러 방식으로 로딩 시도
        try:
            state_dict = torch.load(input_file, map_location='cpu')
        except Exception as e:
            print(f"표준 방식 로딩 실패: {e}")
            
            # 바이너리 모드로 시도
            with open(input_file, 'rb') as f:
                try:
                    state_dict = torch.load(f, map_location='cpu')
                except Exception as e2:
                    print(f"바이너리 모드 로딩 실패: {e2}")
                    
                    # pickle 직접 로드 시도
                    try:
                        f.seek(0)
                        state_dict = pickle.load(f)
                    except Exception as e3:
                        print(f"모든 로딩 방식 실패: {e3}")
                        return False

        # 모델 저장 (여러 방식으로 시도)
        try:
            # 방법 1: pickle 프로토콜 2 사용 (Python 2/3 호환)
            torch.save(state_dict, output_file, _use_new_zipfile_serialization=False, pickle_protocol=2)
        except Exception as e:
            print(f"저장 방식 1 실패: {e}")
            try:
                # 방법 2: 구형 직렬화 방식 사용
                with open(output_file, 'wb') as f:
                    pickle.dump(state_dict, f, protocol=2)
            except Exception as e2:
                print(f"저장 방식 2 실패: {e2}")
                return False
                
        print(f"액션 인식 모델 변환 성공: {output_file}")
        return True
    except Exception as e:
        print(f"액션 인식 모델 변환 중 오류 발생: {e}")
        return False

def verify_model_exists(file_path):
    """모델 파일 존재 확인 및 크기 검증"""
    if not os.path.exists(file_path):
        return False
        
    file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB 단위
    if file_size < 0.1:  # 100KB 미만은 유효한 모델이 아닐 가능성이 높음
        print(f"경고: {file_path} 파일이 너무 작습니다 ({file_size:.2f}MB). 올바른 모델 파일이 아닐 수 있습니다.")
        return False
        
    return True

def create_dummy_model(output_file, model_type):
    """테스트용 더미 모델 생성"""
    if model_type == 'yolo':
        from Detection.Models import Darknet
        model = Darknet('Models/yolo-tiny-onecls/yolov3-tiny-onecls.cfg')
        torch.save(model.state_dict(), output_file)
    elif model_type == 'sppe50':
        from SPPE.src.models.FastPose import FastPose
        model = FastPose('resnet50', 17)
        torch.save(model.state_dict(), output_file)
    elif model_type == 'sppe101':
        from SPPE.src.models.FastPose import FastPose
        model = FastPose('resnet101')
        torch.save(model.state_dict(), output_file)
    elif model_type == 'action':
        from Actionsrecognition.Models import TwoStreamSpatialTemporalGraph
        graph_args = {'strategy': 'spatial'}
        num_class = 7  # 클래스 수
        model = TwoStreamSpatialTemporalGraph(graph_args, num_class)
        torch.save(model.state_dict(), output_file)
    else:
        return False
    
    print(f"더미 모델 생성 완료: {output_file}")
    return True

def main():
    parser = argparse.ArgumentParser(description='PyTorch 모델 파일 변환 도구')
    parser.add_argument('--all', action='store_true', help='모든 모델 파일 변환')
    parser.add_argument('--yolo', action='store_true', help='YOLO 모델 파일 변환')
    parser.add_argument('--sppe', action='store_true', help='SPPE 모델 파일 변환')
    parser.add_argument('--action', action='store_true', help='TSSTG 모델 파일 변환')
    parser.add_argument('--dummy', action='store_true', help='모델 파일이 없는 경우 더미 모델 생성')
    parser.add_argument('--force', action='store_true', help='기존 변환 파일이 있어도 강제로 다시 변환')
    
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
    
    print("\n===== PyTorch 모델 변환 도구 =====")
    print(f"PyTorch 버전: {torch.__version__}")
    print(f"모델 변환 시작...\n")
    
    # 디렉토리 확인 및 생성
    os.makedirs(os.path.dirname(yolo_model_new), exist_ok=True)
    os.makedirs(os.path.dirname(sppe_res50_new), exist_ok=True)
    os.makedirs(os.path.dirname(action_model_new), exist_ok=True)
    
    success_count = 0
    fail_count = 0
    
    # YOLO 모델 변환
    if args.yolo:
        print("\n===== YOLO 모델 변환 =====")
        if os.path.exists(yolo_model_new) and not args.force:
            print(f"이미 변환된 파일이 존재합니다: {yolo_model_new}")
            print("기존 파일을 덮어쓰려면 --force 옵션을 사용하세요.")
            success_count += 1
        else:
            if verify_model_exists(yolo_model):
                if convert_yolo_model(yolo_model, yolo_model_new, yolo_config):
                    success_count += 1
                else:
                    fail_count += 1
                    if args.dummy:
                        print("더미 YOLO 모델 생성 시도...")
                        if create_dummy_model(yolo_model_new, 'yolo'):
                            print("더미 모델을 성공적으로 생성했습니다.")
                            success_count += 1
                            fail_count -= 1
            else:
                print(f"YOLO 모델 파일을 찾을 수 없습니다: {yolo_model}")
                fail_count += 1
                if args.dummy:
                    print("더미 YOLO 모델 생성 시도...")
                    if create_dummy_model(yolo_model_new, 'yolo'):
                        print("더미 모델을 성공적으로 생성했습니다.")
                        success_count += 1
                        fail_count -= 1
    
    # SPPE 모델 변환
    if args.sppe:
        print("\n===== SPPE ResNet50 모델 변환 =====")
        if os.path.exists(sppe_res50_new) and not args.force:
            print(f"이미 변환된 파일이 존재합니다: {sppe_res50_new}")
            print("기존 파일을 덮어쓰려면 --force 옵션을 사용하세요.")
            success_count += 1
        else:
            if verify_model_exists(sppe_res50):
                if convert_sppe_model(sppe_res50, sppe_res50_new):
                    success_count += 1
                else:
                    fail_count += 1
                    if args.dummy:
                        print("더미 SPPE ResNet50 모델 생성 시도...")
                        if create_dummy_model(sppe_res50_new, 'sppe50'):
                            print("더미 모델을 성공적으로 생성했습니다.")
                            success_count += 1
                            fail_count -= 1
            else:
                print(f"SPPE ResNet50 모델 파일을 찾을 수 없습니다: {sppe_res50}")
                fail_count += 1
                if args.dummy:
                    print("더미 SPPE ResNet50 모델 생성 시도...")
                    if create_dummy_model(sppe_res50_new, 'sppe50'):
                        print("더미 모델을 성공적으로 생성했습니다.")
                        success_count += 1
                        fail_count -= 1
        
        print("\n===== SPPE ResNet101 모델 변환 =====")
        if os.path.exists(sppe_res101_new) and not args.force:
            print(f"이미 변환된 파일이 존재합니다: {sppe_res101_new}")
            print("기존 파일을 덮어쓰려면 --force 옵션을 사용하세요.")
            success_count += 1
        else:
            if verify_model_exists(sppe_res101):
                if convert_sppe_model(sppe_res101, sppe_res101_new):
                    success_count += 1
                else:
                    fail_count += 1
                    if args.dummy:
                        print("더미 SPPE ResNet101 모델 생성 시도...")
                        if create_dummy_model(sppe_res101_new, 'sppe101'):
                            print("더미 모델을 성공적으로 생성했습니다.")
                            success_count += 1
                            fail_count -= 1
            else:
                print(f"SPPE ResNet101 모델 파일을 찾을 수 없습니다: {sppe_res101}")
                fail_count += 1
                if args.dummy:
                    print("더미 SPPE ResNet101 모델 생성 시도...")
                    if create_dummy_model(sppe_res101_new, 'sppe101'):
                        print("더미 모델을 성공적으로 생성했습니다.")
                        success_count += 1
                        fail_count -= 1
    
    # Action 모델 변환
    if args.action:
        print("\n===== TSSTG 액션 모델 변환 =====")
        if os.path.exists(action_model_new) and not args.force:
            print(f"이미 변환된 파일이 존재합니다: {action_model_new}")
            print("기존 파일을 덮어쓰려면 --force 옵션을 사용하세요.")
            success_count += 1
        else:
            if verify_model_exists(action_model):
                if convert_action_model(action_model, action_model_new):
                    success_count += 1
                else:
                    fail_count += 1
                    if args.dummy:
                        print("더미 액션 모델 생성 시도...")
                        if create_dummy_model(action_model_new, 'action'):
                            print("더미 모델을 성공적으로 생성했습니다.")
                            success_count += 1
                            fail_count -= 1
            else:
                print(f"TSSTG 액션 모델 파일을 찾을 수 없습니다: {action_model}")
                fail_count += 1
                if args.dummy:
                    print("더미 액션 모델 생성 시도...")
                    if create_dummy_model(action_model_new, 'action'):
                        print("더미 모델을 성공적으로 생성했습니다.")
                        success_count += 1
                        fail_count -= 1
    
    print("\n===== 모델 변환 결과 =====")
    print(f"성공: {success_count}, 실패: {fail_count}")
    
    if success_count > 0:
        print("\n모델 변환이 완료되었습니다!")
        print("이제 다음 명령어로 프로그램을 실행할 수 있습니다:")
        print("  python3 main.py -C 0")
    else:
        print("\n모든 모델 변환에 실패했습니다.")
        print("모델 파일이 있는지 확인하고 다시 시도하세요.")
        print("기존 파일이 있는 경우 --force 옵션을 사용하여 강제로 다시 변환할 수 있습니다.")
        print("모델 파일이 없는 경우 --dummy 옵션을 사용하여 임시 더미 모델을 생성할 수 있습니다.")
        print("예: python3 convert_model.py --all --dummy")

if __name__ == "__main__":
    main() 