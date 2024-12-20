import numpy as np
import openvino as ov
import cv2
import os
import shutil

def play_video(video_source):
    camera = cv2.VideoCapture(video_source)

    st_frame = st.empty()
    while(camera.isOpened()):
        ret, frame = camera.read()

        if ret:
            visualized_image = utils.predict_image(frame, conf_threshold)
            st_frame.image(visualized_image, channels = "BGR")
        else:
            camera.release()
            break

st.set_page_config(
    page_title="Fire/smoke-detection",
    page_icon="🔥",
    layout="centered",
    initial_sidebar_state="expanded")

st.title("Fire/smoke-detection Project :fire:")
source_radio = st.sidebar.radio("Select Source",["IMAGE","VIDEO","WEBCAM"])

st.sidebar.header("Confidence")
conf_threshold = float(st.sidebar.slider("Select the Confidence Threshold", 10, 100, 20))/100
