import numpy as np
import collections
import time
from pathlib import Path

import cv2
import numpy as np
from IPython import display
from numpy.lib.stride_tricks import as_strided
import openvino as ov

import openvino.properties.hint as hints
from .openpose_decoder import OpenPoseDecoder


decoder = OpenPoseDecoder()


# --- 자세 보정 및 민감도 설정 ---
CALIBRATION_TIME = 4  # 5초 후에 자세를 측정합니다.
SENSITIVITY = 45      # 기준 자세보다 이 값(픽셀)만큼 더 숙이면 경고합니다.

# --- 내부 상태 변수 (수정하지 마세요) ---
g_calibration_start_time = time.time()
g_baseline_y_diff = None

colors = (
    (255, 0, 0),
    (255, 0, 255),
    (170, 0, 255),
    (255, 0, 85),
    (255, 0, 170),
    (85, 255, 0),
    (255, 170, 0),
    (0, 255, 0),
    (255, 255, 0),
    (0, 255, 85),
    (170, 255, 0),
    (0, 85, 255),
    (0, 255, 170),
    (0, 0, 255),
    (0, 255, 255),
    (85, 0, 255),
    (0, 170, 255),
)

default_skeleton = (
    (15, 13),
    (13, 11),
    (16, 14),
    (14, 12),
    (11, 12),
    (5, 11),
    (6, 12),
    (5, 6),
    (5, 7),
    (6, 8),
    (7, 9),
    (8, 10),
    (1, 2),
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (3, 5),
    (4, 6),
)

    
class Process:
   
    def __init__(self, compiled_model):
       
        self.compiled_model = compiled_model
     
   
    def pool2d(self,A, kernel_size, stride, padding, pool_mode="max"):

        # Padding
            A = np.pad(A, padding, mode="constant")

            # Window view of A
            output_shape = (
                (A.shape[0] - kernel_size) // stride + 1,
                (A.shape[1] - kernel_size) // stride + 1,
            )
            kernel_size = (kernel_size, kernel_size)
            A_w = as_strided(
                A,
                shape=output_shape + kernel_size,
                strides=(stride * A.strides[0], stride * A.strides[1]) + A.strides,
            )
            A_w = A_w.reshape(-1, *kernel_size)

            # Return the result of pooling.
            if pool_mode == "max":
                return A_w.max(axis=(1, 2)).reshape(output_shape)
            elif pool_mode == "avg":
                return A_w.mean(axis=(1, 2)).reshape(output_shape)
        
    def heatmap_nms(self,heatmaps, pooled_heatmaps):
            return heatmaps * (heatmaps == pooled_heatmaps)

   
    # Get poses from results.
    def process_results(self,img, pafs, heatmaps):
        
        # This processing comes from
        # https://github.com/openvinotoolkit/open_model_zoo/blob/master/demos/common/python/models/open_pose.py
        pooled_heatmaps = np.array([[self.pool2d(h, kernel_size=3, stride=1, padding=1, pool_mode="max") for h in heatmaps[0]]])
        nms_heatmaps = self.heatmap_nms(heatmaps, pooled_heatmaps)

        # Decode poses.
        poses, scores = decoder(heatmaps, nms_heatmaps, pafs)
        output_shape = list(self.compiled_model.output(index=0).partial_shape)
        output_scale = (
            img.shape[1] / output_shape[3].get_length(),
            img.shape[0] / output_shape[2].get_length(),
        )
        # Multiply coordinates by a scaling factor.
        poses[:, :, :2] *= output_scale
        return poses, scores


    def draw_poses(self,img, poses, point_score_threshold, skeleton=default_skeleton):
        # 전역 변수 사용 선언
        global g_calibration_start_time, g_baseline_y_diff

        if poses.size == 0 and g_baseline_y_diff is None:
            # 화면에 사람이 없으면 보정 진행이 안됨을 알림
            cv2.putText(img, "Please be in front of the camera.", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            return img

        # 1. 기준점이 아직 설정되지 않았을 경우 (보정 및 캡처 단계)
        if g_baseline_y_diff is None:
            elapsed_time = time.time() - g_calibration_start_time

            # 1-1. 5초 카운트다운
            if elapsed_time <= CALIBRATION_TIME:
                status_text = f"Get Ready... {CALIBRATION_TIME - elapsed_time:.1f}s"
                cv2.putText(img, status_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
                cv2.putText(img, "At 0 sec, hold your BEST posture!", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            # 1-2. 5초 후, 기준점 캡처
            else:
                capture_success = False
                for pose in poses:
                    nose_score = pose[0, 2]
                    l_shoulder_score = pose[5, 2]
                    r_shoulder_score = pose[6, 2]

                    if nose_score > point_score_threshold and l_shoulder_score > point_score_threshold and r_shoulder_score > point_score_threshold:
                        # 유효한 자세가 감지되면, 현재 y좌표 차이를 기준점으로 설정
                        nose_y = pose[0, 1]
                        l_shoulder_y = pose[5, 1]
                        r_shoulder_y = pose[6, 1]
                        shoulder_y_avg = (l_shoulder_y + r_shoulder_y) / 2
                        
                        g_baseline_y_diff = nose_y - shoulder_y_avg
                        
                        print(f"Baseline Captured! Value: {g_baseline_y_diff:.2f}")
                        cv2.putText(img, "Baseline CAPTURED!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
                        capture_success = True
                        break  # 첫 번째 감지된 자세로 기준을 잡고 반복 중단
                
                if not capture_success:
                    cv2.putText(img, "Looking for you... Hold posture!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        # 2. 기준점이 설정된 후 (모니터링 단계)
        else:
            for pose in poses:
                nose_score = pose[0, 2]
                l_shoulder_score = pose[5, 2]
                r_shoulder_score = pose[6, 2]

                if nose_score > point_score_threshold and l_shoulder_score > point_score_threshold and r_shoulder_score > point_score_threshold:
                    nose_y = pose[0, 1]
                    l_shoulder_y = pose[5, 1]
                    r_shoulder_y = pose[6, 1]
                    shoulder_y_avg = (l_shoulder_y + r_shoulder_y) / 2
                    current_y_diff = nose_y - shoulder_y_avg
                    
                    cv2.putText(img, f"Baseline: {g_baseline_y_diff:.1f}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(img, f"Current: {current_y_diff:.1f}", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                    # 거북목 판단 로직
                    if current_y_diff > g_baseline_y_diff + SENSITIVITY:
                        cv2.putText(img, "WARNING: Turtle Neck!", (100, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

        # 자세 시각화 (기존과 동일)
        img_limbs = np.copy(img)
        for pose in poses:
            points = pose[:, :2].astype(np.int32)
            points_scores = pose[:, 2]
            for i, (p, v) in enumerate(zip(points, points_scores)):
                if v > point_score_threshold:
                    cv2.circle(img, tuple(p), 1, colors[i], 2)
            for i, j in skeleton:
                if points_scores[i] > point_score_threshold and points_scores[j] > point_score_threshold:
                    cv2.line(img_limbs, tuple(points[i]), tuple(points[j]), color=colors[j], thickness=4)
        cv2.addWeighted(img, 0.4, img_limbs, 0.6, 0, dst=img)
        return img
