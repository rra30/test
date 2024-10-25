import streamlit as st
import utils
import cv2
import numpy as np
import io
import PIL
from PIL import Image

st.set_page_config(
    page_title="test",
    page_icon="🌚",
    layout="centered",
    initial_sidebar_state="expanded")

st.title("Fire/smoke-detection Project :fire:")
source_model = st.sidebar.radio("Select Source",["Model_Create","Model_test"])

if source_model == "Model_test":
    st.sidebar.radio("Choose Model type",["Test_Image","Test_Video"])

if source_model == "Model_Create":
    st.sidebar.radio("Choose Model type",["Load_Model","study_Model"])
