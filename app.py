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

if (csv_directory == False):
    source_model = st.sidebar.radio("The data is not available. But You can use basic csv file",["Data Collection","Utilize collected data"])
    if source_model == "Data Collection":
        #time.sleep(5)
        try:
            import Collection
            Main()
        except:
            pass
else:
    source_model = st.sidebar.radio("Select",["Data Collection","Utilize collected data"])

    if (source_model == "Utilize collected data"):
        st.sidebar.header("Output")
        st.sidebar.radio("Choose",["WebCam","Video","Image"])
        try:
            import utils
            Main()
        except:
            pass

    elif (source_model == "Data Collection"):
        try:
            import Collection
            Main()
        except:
            pass
