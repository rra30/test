import csv
import os
import numpy as np
import openvino as ov
import cv2
import datetime as dt
import streamlit as st

csv_directory = './model/model/2024_12_21_12_38_19.csv'

directory = os.listdir()

for filename in directory:
    if filename.endswith(".csv"):
        csv_directory = True
        csv_name = filename

csv_name = "2024_12_21_12_38_19.csv"

now = dt.datetime.now()

if csv_name == None or csv_name == "":
    csv_name = now.strftime('20%y_%m_%d_%H_%M_%S')

st.set_page_config(
    page_title="I want go to home",
    page_icon="✋",
    layout="centered",
    initial_sidebar_state="expanded")

st.title("Welcome to '살자 예방 게이트 키퍼' ✋")
st.sidebar.header("Setting")

# 기존 코드의 로직 유지
if not csv_directory:
    source_model = st.sidebar.radio("The data is not available. But You can use basic csv file", ["Data Collection", "Utilize collected data"])
    if source_model == "Data Collection":
        try:
            import Collection
            Main()  # Main 함수 호출
        except:
            pass
else:
    source_model = st.sidebar.radio("Select", ["Data Collection", "Utilize collected data"])

    if source_model == "Utilize collected data":
        try:
            import utils
            Main()  # Main 함수 호출
        except:
            pass

    elif source_model == "Data Collection":
        try:
            import Collection
            Main()  # Main 함수 호출
        except:
            pass
