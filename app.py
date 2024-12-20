import streamlit as st
import numpy as np
import openvino as ov
import cv2
import io
import PIL
from PIL import Image
#import os
#import shutil
#import utils

st.set_page_config(
    page_title="Ai name",
    page_icon="✋",
    layout="centered",
    initial_sidebar_state="expanded")

st.title("Ai name project ✋")
