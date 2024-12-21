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
