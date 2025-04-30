import os
import cv2
import time
import torch
import argparse
import numpy as np
import gc

from Detection.Utils import ResizePadding
from CameraLoader import CamLoader, CamLoader_Q
from DetectorLoader import TinyYOLOv3_onecls

from PoseEstimateLoader import SPPE_FastPose
from fn import draw_single

from Track.Tracker import Detection, Tracker
from ActionsEstLoader import TSSTG

# source = '../Data/test_video/test7.mp4'
# source = '../Data/falldata/Home/Videos/video (2).avi'  # hard detect
source = '../Data/falldata/Home/Videos/video (1).avi'


# source = 2

def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized


def preproc(image):
    """preprocess function for CameraLoader.
    """
    image = resize_fn(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def kpt2bbox(kpt, ex=20):
    """Get bbox that hold on all of the keypoints (x,y)
    kpt: array of shape `(N, 2)`,
    ex: (int) expand bounding box,
    """
    return np.array((kpt[:, 0].min() - ex, kpt[:, 1].min() - ex,
                     kpt[:, 0].max() + ex, kpt[:, 1].max() + ex))


# Jetson Nano GPU 최적화 설정
def optimize_cuda_for_jetson():
    """Jetson Nano GPU 최적화 함수"""
    # CUDA 설정 초기화
    torch.backends.cudnn.enabled = True
    
    # 메모리 사용 최적화
    torch.backends.cudnn.benchmark = True  # 반복적인 크기의 입력에 대해 성능 향상
    
    # 결정적 알고리즘 비활성화 (속도 향상)
    torch.backends.cudnn.deterministic = False
    
    # 메모리 할당 전략 설정
    if hasattr(torch.cuda, 'empty_cache'):
        torch.cuda.empty_cache()
    
    # TensorRT 최적화 활성화 (가능한 경우)
    try:
        import torch2trt
        print("TensorRT 최적화 사용 가능")
    except ImportError:
        print("TensorRT 최적화 사용 불가 (torch2trt 설치되지 않음)")
    
    # Jetson-specific 최적화
    os.environ['CUDA_MODULE_LOADING'] = 'LAZY'  # 필요한 시점에 모듈 로딩
    
    # 메모리 사용 제한 설정 (MB 단위, Jetson Nano는 대략 4GB)
    # total_memory = torch.cuda.get_device_properties(0).total_memory
    # Jetson Nano를 위한 메모리 제한 (전체 메모리의 75%)
    # max_memory = int(total_memory * 0.75)
    # torch.cuda.set_per_process_memory_fraction(0.75)
    
    return True


if __name__ == '__main__':
    par = argparse.ArgumentParser(description='Human Fall Detection Demo.')
    par.add_argument('-C', '--camera', default=source,  # required=True,  # default=2,
                     help='Source of camera or video file path.')
    par.add_argument('--detection_input_size', type=int, default=320,
                     help='Size of input in detection model in square must be divisible by 32 (int).')
    par.add_argument('--pose_input_size', type=str, default='192x128',
                     help='Size of input in pose model must be divisible by 32 (h, w)')
    par.add_argument('--pose_backbone', type=str, default='resnet50',
                     help='Backbone model for SPPE FastPose model.')
    par.add_argument('--show_detected', default=False, action='store_true',
                     help='Show all bounding box from detection.')
    par.add_argument('--show_skeleton', default=True, action='store_true',
                     help='Show skeleton pose.')
    par.add_argument('--save_out', type=str, default='',
                     help='Save display to video file.')
    par.add_argument('--device', type=str, default='cuda',
                     help='Device to run model on cpu or cuda.')
    par.add_argument('--debug', default=False, action='store_true',
                     help='Display debug information.')
    par.add_argument('--optimize', default=True, action='store_true',
                     help='Jetson Nano 최적화 활성화')
    par.add_argument('--skip_frames', type=int, default=0,
                     help='처리 속도 향상을 위해 건너뛸 프레임 수 (0: 건너뛰기 없음)')
    args = par.parse_args()

    device = args.device
    
    # CUDA 사용 가능한지 확인
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA를 사용할 수 없습니다. CPU로 전환합니다.")
        device = 'cpu'
    else:
        if device == 'cuda':
            print(f"CUDA 사용 가능: {torch.cuda.get_device_name(0)}")
            total_memory = torch.cuda.get_device_properties(0).total_memory / 1024 / 1024
            print(f"CUDA 메모리: {total_memory:.2f} MB")
            
            # Jetson Nano GPU 최적화
            if args.optimize and "Tegra" in torch.cuda.get_device_name(0):
                print("Jetson Nano GPU 최적화 적용 중...")
                optimize_cuda_for_jetson()

    # DETECTION MODEL.
    inp_dets = args.detection_input_size
    detect_model = TinyYOLOv3_onecls(inp_dets, device=device)

    # POSE MODEL.
    inp_pose = args.pose_input_size.split('x')
    inp_pose = (int(inp_pose[0]), int(inp_pose[1]))
    pose_model = SPPE_FastPose(args.pose_backbone, inp_pose[0], inp_pose[1], device=device)

    # Tracker.
    max_age = 30
    tracker = Tracker(max_age=max_age, n_init=3)

    # Actions Estimate.
    action_model = TSSTG(device=device)  # device 파라미터 전달

    resize_fn = ResizePadding(inp_dets, inp_dets)

    cam_source = args.camera
    if type(cam_source) is str and os.path.isfile(cam_source):
        # Use loader thread with Q for video file.
        cam = CamLoader_Q(cam_source, queue_size=1000, preprocess=preproc).start()
    else:
        # Use normal thread loader for webcam.
        cam = CamLoader(int(cam_source) if cam_source.isdigit() else cam_source,
                        preprocess=preproc).start()

    # frame_size = cam.frame_size
    # scf = torch.min(inp_size / torch.FloatTensor([frame_size]), 1)[0]

    outvid = False
    if args.save_out != '':
        outvid = True
        codec = cv2.VideoWriter_fourcc(*'MJPG')
        writer = cv2.VideoWriter(args.save_out, codec, 30, (inp_dets * 2, inp_dets * 2))

    fps_time = 0
    f = 0
    fps_counter = 0
    fps_avg = 0
    processing_times = []
    
    # 메모리 사용량 모니터링 초기화
    if device == 'cuda':
        initial_memory = torch.cuda.memory_allocated() / 1024 / 1024
    
    print("\n============= 낙상 감지 시스템 실행 =============")
    print("종료하려면 'q' 키를 누르세요.\n")
    
    while cam.grabbed():
        f += 1
        
        # 프레임 건너뛰기 (성능 향상)
        if args.skip_frames > 0 and f % (args.skip_frames + 1) != 0:
            continue
            
        start_time = time.time()
        
        frame = cam.getitem()
        image = frame.copy()

        # 주기적으로 CUDA 캐시 비우기 (메모리 누수 방지)
        if device == 'cuda' and f % 30 == 0:
            torch.cuda.empty_cache()
            if args.debug:
                current_memory = torch.cuda.memory_allocated() / 1024 / 1024
                print(f"CUDA 메모리 사용량: {current_memory:.2f} MB")

        # Detect humans bbox in the frame with detector model.
        detected = detect_model.detect(frame, need_resize=False, expand_bb=10)

        # Predict each tracks bbox of current frame from previous frames information with Kalman filter.
        tracker.predict()
        # Merge two source of predicted bbox together.
        for track in tracker.tracks:
            det = torch.tensor([track.to_tlbr().tolist() + [0.5, 1.0, 0.0]], dtype=torch.float32)
            detected = torch.cat([detected, det], dim=0) if detected is not None else det

        detections = []  # List of Detections object for tracking.
        if detected is not None:
            # detected = non_max_suppression(detected[None, :], 0.45, 0.2)[0]
            # Predict skeleton pose of each bboxs.
            try:
                poses = pose_model.predict(frame, detected[:, 0:4], detected[:, 4])

                # Create Detections object.
                detections = [Detection(kpt2bbox(ps['keypoints'].numpy()),
                                        np.concatenate((ps['keypoints'].numpy(),
                                                      ps['kp_score'].numpy()), axis=1),
                                        ps['kp_score'].mean().numpy()) for ps in poses]
            except Exception as e:
                if args.debug:
                    print(f"포즈 추정 실패: {e}")
                # 예외 발생 시 빈 detections 리스트 유지
                detections = []

            # VISUALIZE.
            if args.show_detected:
                for bb in detected[:, 0:5]:
                    frame = cv2.rectangle(frame, (bb[0], bb[1]), (bb[2], bb[3]), (0, 0, 255), 1)

        # Update tracks by matching each track information of current and previous frame or
        # create a new track if no matched.
        tracker.update(detections)

        event = ""

        # Predict Actions of each track.
        for i, track in enumerate(tracker.tracks):
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            bbox = track.to_tlbr().astype(int)
            center = track.get_center().astype(int)

            action = 'pending..'
            clr = (0, 255, 0)
            # Use 30 frames time-steps to prediction.
            if len(track.keypoints_list) == 30:
                pts = np.array(track.keypoints_list, dtype=np.float32)
                out = action_model.predict(pts, frame.shape[:2])
                action_name = action_model.class_names[out[0].argmax()]
                action = '{}: {:.2f}%'.format(action_name, out[0].max() * 100)
                if action_name == 'Fall Down':
                    clr = (255, 0, 0)
                    event = "Fall Down"
                elif action_name == 'Lying Down':
                    clr = (255, 200, 0)
                    event = "Lying Down"

            # VISUALIZE.
            if track.time_since_update == 0:
                if args.show_skeleton:
                    frame = draw_single(frame, track.keypoints_list[-1])
                frame = cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 1)
                frame = cv2.putText(frame, str(track_id), (center[0], center[1]), cv2.FONT_HERSHEY_COMPLEX,
                                    0.4, (255, 0, 0), 2)
                frame = cv2.putText(frame, action, (bbox[0] + 5, bbox[1] + 15), cv2.FONT_HERSHEY_COMPLEX,
                                    0.4, clr, 1)

        # 이벤트 발생 시 알림 표시
        if event:
            # 화면 상단에 알림 메시지 표시
            cv2.putText(frame, f"경보: {event} 감지됨!", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                        1.0, (0, 0, 255), 2)
            # 화면 테두리를 빨간색으로 표시
            h, w = frame.shape[:2]
            thickness = 10
            border = np.zeros_like(frame)
            border = cv2.rectangle(border, (0, 0), (w, h), (0, 0, 255), thickness)
            frame = cv2.addWeighted(frame, 1, border, 0.7, 0)

        # 프레임 처리 시간 및 FPS 계산
        process_time = time.time() - start_time
        processing_times.append(process_time)
        
        # 최근 30프레임의 평균 처리 시간으로 FPS 계산
        if len(processing_times) > 30:
            processing_times.pop(0)
        avg_process_time = sum(processing_times) / len(processing_times)
        current_fps = 1.0 / avg_process_time if avg_process_time > 0 else 0
        
        # FPS 이동 평균 계산
        fps_counter += 1
        fps_avg = fps_avg * 0.9 + current_fps * 0.1 if fps_counter > 1 else current_fps

        # Show Frame.
        frame = cv2.resize(frame, (0, 0), fx=2., fy=2.)
        frame = cv2.putText(frame, 'FPS: %.2f (Avg: %.2f)' % (current_fps, fps_avg),
                            (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        frame = frame[:, :, ::-1]
        fps_time = time.time()

        if outvid:
            writer.write(frame)
        
        # 디스플레이 크기 조정
        display_height = min(1080, 800)  # 최대 세로 크기 제한
        imS = image_resize(frame, height=display_height)
        cv2.imshow('Fall Detection System', imS)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Clear resource.
    cam.stop()
    if outvid:
        writer.release()
        
    # 최종 메모리 사용량 표시
    if device == 'cuda':
        final_memory = torch.cuda.memory_allocated() / 1024 / 1024
        print(f"최종 CUDA 메모리 사용량: {final_memory:.2f} MB")
        
    # 리소스 해제    
    if device == 'cuda':
        torch.cuda.empty_cache()
        gc.collect()
    
    cv2.destroyAllWindows()
    print("\n============= 낙상 감지 시스템 종료 =============")
