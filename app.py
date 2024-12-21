import streamlit as st
import numpy as np
import openvino as ov
import cv2
import io
import PIL
from PIL import Image
import os
import utils
import time
import datetime as dt
import Collection

csv_directory = False

directory = os.listdir()

for filename in directory:
    if filename.endswith(".csv"):
        csv_directory = True
        csv_name = filename

csv_name = ""

now = dt.datetime.now()

if (csv_name == None or csv_name == ""):
    csv_name = now.strftime('20%y_%m_%d_%H_%M_%S')

st.set_page_config(
    page_title="I want go to home",
    page_icon="✋",
    layout="centered",
    initial_sidebar_state="expanded")

st.title("Welcome to '자살 예방 게이트 키퍼' ✋")
st.sidebar.header("Setting")

#for i in range(len(os.listdir())):

if (csv_directory == False):
    source_model = st.sidebar.radio("The data is not available. But You can use basic csv file",["Data Collection","Utilize collected data"])
    if source_model == "Data Collection":
        #time.sleep(5)
        try:
            Col_start()
        except:
            pass
else:
    source_model = st.sidebar.radio("Select",["Data Collection","Utilize collected data"])

    if (source_model == "Utilize collected data"):
        try:
            normal_start()
        except:
            pass

    elif (source_model == "Data Collection"):
        try:
            Col_start()
        except:
            pass
