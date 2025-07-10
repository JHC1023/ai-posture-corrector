import collections
import time
from pathlib import Path
import cv2
import numpy as np
from IPython import display
import openvino as ov
import openvino.properties.hint as hints

from turtle_neck import OpenPoseDecoder, Process ,prepare_notebook_utils, download_model


def run_pose_estimation(source=0, flip=False, use_popup=False, skip_first_frames=0):
   
    utils = prepare_notebook_utils()
    # 모델 다운로드 (이미 있으면 스킵됨)
   
    model_path = download_model()
    core = ov.Core()
    model = core.read_model(model_path)
    compiled_model = core.compile_model(
        model=model,
        device_name='CPU',
        config={hints.performance_mode(): hints.PerformanceMode.LATENCY}
    )
    decoder = OpenPoseDecoder()
    processor = Process(compiled_model)
    input_layer = compiled_model.input(0)
    output_layers = compiled_model.outputs

    # Get the input size.
    height, width = list(input_layer.shape)[2:]

    input_layer.any_name, [o.any_name for o in output_layers]
    pafs_output_key = compiled_model.output("Mconv7_stage2_L1")
    heatmaps_output_key = compiled_model.output("Mconv7_stage2_L2")
    player = None
    try:
        # Create a video player to play with target fps.
        player = utils.VideoPlayer(source, flip=flip, fps=30, skip_first_frames=skip_first_frames)
        # Start capturing.
        player.start()
        if use_popup:
            title = "Press ESC to Exit"
            cv2.namedWindow(title, cv2.WINDOW_GUI_NORMAL | cv2.WINDOW_AUTOSIZE)

        processing_times = collections.deque()

        while True:
            # Grab the frame.
            frame = player.next()
            if frame is None:
                print("Source ended")
                break
            
            # If the frame is larger than full HD, reduce size to improve the performance.
            scale = 1280 / max(frame.shape)
            if scale < 1:
                frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

            # Resize the image and change dims to fit neural network input.
            # (see https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/intel/human-pose-estimation-0001)
            input_img = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
            # Create a batch of images (size = 1).
            input_img = input_img.transpose((2, 0, 1))[np.newaxis, ...]

            # Measure processing time.
            start_time = time.time()
            # Get results.
            results = compiled_model([input_img])
            stop_time = time.time()

            pafs = results[pafs_output_key]
            heatmaps = results[heatmaps_output_key]
            # Get poses from network results.
            process_results = processor.process_results(frame, pafs, heatmaps)
            poses, scores = processor.process_results(frame, pafs, heatmaps)

            # Draw poses on a frame.
            draw_poses = processor.draw_poses(frame, poses, 0.1)
            frame = processor.draw_poses(frame, poses, 0.1)

            processing_times.append(stop_time - start_time)
            # Use processing times from last 200 frames.
            if len(processing_times) > 200:
                processing_times.popleft()

            _, f_width = frame.shape[:2]
            # mean processing time [ms]
            processing_time = np.mean(processing_times) * 1000
            fps = 1000 / processing_time
            cv2.putText(
                frame,
                f"Inference time: {processing_time:.1f}ms ({fps:.1f} FPS)",
                (20, 40),
                cv2.FONT_HERSHEY_COMPLEX,
                f_width / 1000,
                (0, 0, 255),
                1,
                cv2.LINE_AA,
            )

            # Use this workaround if there is flickering.
            if use_popup:
                cv2.imshow(title, frame)
                key = cv2.waitKey(1)
                # escape = 27
                if key == 27:
                    break
            else:
                # Encode numpy array to jpg.
                _, encoded_img = cv2.imencode(".jpg", frame, params=[cv2.IMWRITE_JPEG_QUALITY, 90])
                # Create an IPython image.
                i = display.Image(data=encoded_img)
                # Display the image in this notebook.
                display.clear_output(wait=True)
                display.display(i)
    # ctrl-c
    except KeyboardInterrupt:
        print("Interrupted")
    # any different error
    except RuntimeError as e:
        print(e)
    finally:
        if player is not None:
            # Stop capturing.
            player.stop()
        if use_popup:
            cv2.destroyAllWindows()

USE_WEBCAM = True
cam_id = 0


additional_options = {"skip_first_frames": 500} if not USE_WEBCAM else {}

