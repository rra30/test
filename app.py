import streamlit as st
import numpy as np
import openvino as ov
import cv2
import io
import PIL
from PIL import Image
import os
import utils
import datetime as dt

csv_directory = False

directory = os.listdir()

for i in range(len(directory)):
    if directory.endswitch('.csv'):
        csv_directory = True
    
csv_name = ""

now = dt.datetime.now()

if (csv_name == None or csv_name == ""):
    csv_name = now.strftime('20%y_%m_%d_%H_%M_%S')
    
st.set_page_config(
    page_title="I want go to home",
    page_icon="✋",
    layout="centered",
    initial_sidebar_state="expanded")

st.title("Welcome to 'I want go home' ✋")
st.sidebar.header("Setting")

for i in range(len(os.listdir())):
        

if (csv_directory == False):
    source_model = st.sidebar.radio("The data is not available.",["Data Collection"])
    if source_model == "Data Collection"
else:
    source_model = st.sidebar.radio("Select",["Data Collection","Utilize collected data"])
    
    if (source_model == "Utilize collected data"):
        try:
            if (csv_directory != True): #  학습된 데이터로 이동
                single_data = []
                Multiple_data = [['neutral', 'happy', 'sad', 'surprise', 'anger']]
    
    
                columns = ['neutral', 'happy', 'sad', 'surprise', 'anger']
            
                for i in range(len(total_data)):
                    for j in range(5):
                        single_data.append(total_data[i][j])
                    for a in range(5):
                        Multiple_data[i].append(single_data[i])
                    Multiple_data.append([])
    
                f = open(f"{csv_name}.csv", "w")
                writer = csv.writer(f)

                writer.writerows(Multiple_data)
                f.close()
        except:
            pass
    
    elif (source_model == "Data Collection"):
        a
            

