import streamlit as st
import numpy as np
import openvino as ov
import cv2
import io
import PIL
from PIL import Image
import os
#import utils
import time
import datetime as dt
#import Collection

csv_directory = './model/model/2024_12_21_12_38_19.csv'

directory = os.listdir()

for filename in directory:
    if filename.endswith(".csv"):
        csv_directory = True
        csv_name = filename

default_csv_name = "2024_12_21_12_38_19.csv"
csv_name = "2024_12_21_12_38_19.csv"
now = dt.datetime.now()

#def Image():

#def Video():

def WebCam():
    import WebCam
    def Main():
        try:
            camera = cv2.VideoCapture(source)
            while(True):
                ret, frame = camera.read()
                if not ret:
                    a = True
                    break
                    input_frame = preprocess(frame, input_layer_face)
                    results = compiled_model_face([input_frame])[output_layer_face]
                    face_boxes, scores = find_faceboxes(frame, results, confidence_threshold)    
                    show_frame = draw_emotion(face_boxes, frame, scores)
                    cv2.imshow("Webcam", show_frame)
                    if cv2.waitKey(1) & 0xff == ord('q'):
                        a = False
                        raise Breaking
                camera.release()
                cv2.destroyAllWindows()
        except NameError or error:
            pass
        try:
            confidence_threshold = .2
            source = 0
            if __name__ == '__main__':
                Main()
          except NameError or error:
              pass

if (csv_name == None or csv_name == ""):
    csv_name = now.strftime('20%y_%m_%d_%H_%M_%S')

st.set_page_config(
    page_title="I want go to home",
    page_icon="✋",
    layout="centered",
    initial_sidebar_state="expanded")

st.title("Welcome to 'Helpe'! ✋")
st.sidebar.header("Setting")

if (csv_directory == False):
    source_model = st.sidebar.radio("The data is not available. But You can use basic csv file",["Data Collection","Utilize collected data"])
    if source_model == "Data Collection":
        st.sidebar.header("Output")
        sou_Fal_Col_radio = st.sidebar.radio("Choose",["WebCam","Video"])
        try:
            if (sou_Fal_Col_radio == "WebCam"):
                WebCam()
            else:
                Video()
        except:
            pass
    else:
        st.sidebar("Output")
        sou_Fal_Uti_radio = st.sidebar.radio("Choose",["WebCam","Video","Image"])
        try:
            if (source_radio == "WebCam"):
                WebCam()
            elif (source_radio == "Video"):
                Video()
            else:
                Image()
        except:
            pass
else:
    source_model = st.sidebar.radio("Select",["Data Collection","Utilize collected data"])

    if (source_model == "Utilize collected data"):
        st.sidebar.header("Output")
        sou_True_Uti_radio = st.sidebar.radio("Choose",["WebCam","Video","Image"])
        try:
            if (sou_True_Uti_radio):
                WebCam()
            elif (sou_True_Uti_radio):
                Video()
            else:
                Image()
        except:
            pass
            
    elif (source_model == "Data Collection"):
        st.sidebar.header("Output")
        sou_True_Col_radio = st.sidebar.radio("Choose",["WebCam","Video"])
        try:
            if (sou_True_Col_radio == "WebCam"):
                WebCam()
            else:
                Video()
        except:
            pass
