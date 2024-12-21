import csv
import os
import numpy as np
import openvino as ov
import cv2
import datetime as dt
import streamlit as st

a = True
total_data = []

# OpenVINO 모델 로딩
try:
    core = ov.Core()
    model_face = core.read_model(model='./model/face-detection-adas-0001.xml')
    compiled_model_face = core.compile_model(model=model_face, device_name="CPU")
    input_layer_face = compiled_model_face.input(0)
    output_layer_face = compiled_model_face.output(0)

    model_emo = core.read_model(model='./model/emotions-recognition-retail-0003.xml')
    compiled_model_emo = core.compile_model(model=model_emo, device_name="CPU")
    input_layer_emo = compiled_model_emo.input(0)
    output_layer_emo = compiled_model_emo.output(0)
except Exception as e:
    st.error(f"Error loading models: {e}")

# Preprocess 함수 정의
def preprocess(frame, input_layer_face):
    try:
        N, input_channels, input_height, input_width = input_layer_face.shape
        resized_frame = cv2.resize(frame, (input_width, input_height))
        transposed_frame = resized_frame.transpose(2, 0, 1)
        input_frame = np.expand_dims(transposed_frame, 0)
        return input_frame
    except Exception as e:
        st.error(f"Error in preprocess: {e}")
        return None

# 얼굴 박스 찾는 함수
def find_faceboxes(frame, results, confidence_threshold):
    try:
        results = results.squeeze()
        scores = results[:, 2]
        boxes = results[:, -4:]
        face_boxes = boxes[scores >= confidence_threshold]
        scores = scores[scores >= confidence_threshold]
        frame_h, frame_w, frame_channels = frame.shape
        face_boxes = face_boxes * np.array([frame_w, frame_h, frame_w, frame_h])
        face_boxes = face_boxes.astype(np.int64)
        return face_boxes, scores
    except Exception as e:
        st.error(f"Error in find_faceboxes: {e}")
        return [], []

# 감정 그리기 함수
def draw_emotion(face_boxes, frame, scores):
    try:
        show_frame = frame.copy()
        EMOTION_NAMES = ['neutral', 'happy', 'sad', 'surprise', 'anger']
        for i in range(len(face_boxes)):
            xmin, ymin, xmax, ymax = face_boxes[i]
            face = frame[ymin:ymax, xmin:xmax]
            input_frame = preprocess(face, input_layer_emo)
            if input_frame is None:
                continue
            results_emo = compiled_model_emo([input_frame])[output_layer_emo]
            results_emo = results_emo.squeeze()
            index = np.argmax(results_emo)
            total_data.append(results_emo)
            text = EMOTION_NAMES[index] + ' ' + str(f"{results_emo}")
            box_color = (255, 255, 255)
            fontScale = frame.shape[1] / 1000
            cv2.putText(show_frame, text, (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, fontScale, (0, 200, 0), 2)
            cv2.rectangle(img=show_frame, pt1=(xmin, ymin), pt2=(xmax, ymax), color=box_color, thickness=2)
        return show_frame
    except Exception as e:
        st.error(f"Error in draw_emotion: {e}")
        return frame

# Main 함수 정의
def Main():
    try:
        confidence_threshold = 0.2  # 신뢰도 기준 설정
        video_file = st.camera_input("Capture Image")  # Streamlit 카메라 입력 위젯

        if video_file:
            # 카메라에서 이미지를 읽어옴
            frame = np.array(cv2.imdecode(np.frombuffer(video_file.read(), np.uint8), 1))

            if frame is None:
                st.warning("No image captured. Please try again.")
                return

            # 얼굴 탐지를 위해 프레임을 전처리하고 모델에 전달
            input_frame = preprocess(frame, input_layer_face)
            if input_frame is None:
                return

            results = compiled_model_face([input_frame])[output_layer_face]

            # 얼굴 상자와 신뢰도를 추출
            face_boxes, scores = find_faceboxes(frame, results, confidence_threshold)

            # 감정을 그림에 그려 넣기
            show_frame = draw_emotion(face_boxes, frame, scores)

            # Streamlit으로 이미지를 출력
            st.image(show_frame, channels="BGR", use_column_width=True)

    except Exception as e:
        st.error(f"Error in Main: {e}")
