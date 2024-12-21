import streamlit as st
import numpy as np
import openvino as ov
import cv2
import io
import PIL
from PIL import Image
import os
import utils
#import shutil
#import utils

st.set_page_config(
    page_title="I want go to home",
    page_icon="✋",
    layout="centered",
    initial_sidebar_state="expanded")

st.title("Welcome to 'I want go home' ✋")
st.sidebar.header("Setting")

if csv_name == None or csv_name == "":
    source_model = st.sidebar.radio("The data is not available.","Data Collection")
else:
    source_model = st.sidebar.radio("Select","Data Collection","Utilize collected data"])

if source_model == ""
