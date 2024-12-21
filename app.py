import streamlit as st
import numpy as np
import openvino as ov
import cv2
import io
import PIL
from PIL import Image
import os
import time
import datetime as dt

csv_directory = './model/model/2024_12_21_12_38_19.csv'

directory = os.listdir()

for filename in directory:
    if filename.endswith(".csv"):
        csv_directory = True
        csv_name = filename

csv_name = "2024_12_21_12_38_19.csv"

now = dt.datetime.now()

if (csv_name == None or csv_name == ""):
    csv_name = now.strftime('20%y_%m_%d_%H_%M_%S')

st.set_page_config(
    page_title="I want go to home",
    page_icon="✋",
    layout="centered",
    initial_sidebar_state="expanded")

st.title("Welcome to '살자 예방 게이트 키퍼' ✋")
st.sidebar.header("Setting")

#for i in range(len(os.listdir())):
# 기존 코드의 로직 유지
if (csv_directory == False):
    source_model = st.sidebar.radio("The data is not available. But You can use basic csv file",["Data Collection","Utilize collected data"])
    if source_model == "Data Collection":
        #time.sleep(5)
        try:
            import Collection
            Main()
            Main()  # Main 함수 호출
        except:
            pass
else:
    source_model = st.sidebar.radio("Select",["Data Collection","Utilize collected data"])

    if (source_model == "Utilize collected data"):
        try:
            import utils
            Main()
            Main()  # Main 함수 호출
        except:
            pass

    elif (source_model == "Data Collection"):
        try:
            import Collection
            Main()
            Main()  # Main 함수 호출
        except:
            pass
# 여기에 Main() 함수 수정 부분 추가
def Main():
    try:
        confidence_threshold = 0.2  # 신뢰도 기준 설정
        video_file = st.camera_input("Capture Image")  # Streamlit 카메라 입력 위젯
        
        if video_file:
            # 카메라에서 이미지를 읽어옴
            frame = np.array(cv2.imdecode(np.frombuffer(video_file.read(), np.uint8), 1))
            
            # 얼굴 탐지를 위해 프레임을 전처리하고 모델에 전달
            input_frame = preprocess(frame, input_layer_face)
            results = compiled_model_face([input_frame])[output_layer_face]
            
            # 얼굴 상자와 신뢰도를 추출
            face_boxes, scores = find_faceboxes(frame, results, confidence_threshold)
            
            # 감정을 그림에 그려 넣기
            show_frame = draw_emotion(face_boxes, frame, scores)
            
            # Streamlit으로 이미지를 출력
            st.image(show_frame, channels="BGR", use_column_width=True)
    
    except Exception as e:
        st.error(f"Error: {e}")
