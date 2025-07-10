import collections
import time
from pathlib import Path

import cv2
import numpy as np
from IPython import display
from numpy.lib.stride_tricks import as_strided
import openvino as ov
import openvino.properties.hint as hints

import requests
from turtle_neck import OpenPoseDecoder , Process , prepare_notebook_utils , download_model , run_pose_estimation



prepare_notebook_utils()

# 모델 다운로드 (이미 있으면 스킵됨)
model_path = download_model()




run_pose_estimation(use_popup=True)




