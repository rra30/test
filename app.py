import streamlit as st
import utils
import learn
import cv2
import numpy as np
import io
import PIL
import os
import glob
from PIL import Image

st.set_page_config(
    page_title="test",
    page_icon="🌚",
    layout="centered",
    initial_sidebar_state="expanded")

st.title("Test-Project-🌚")
label_map = ['', '']
source_model = st.sidebar.radio("Select Model",["Model_Create","Model_test"])

try:
    os.mkdir("model")
except:
    pass

if source_model == "Model_test":
    with st.sidebar:
        try:
            uploaded_file = st.sidebar.file_uploader("Upload Model", type=("xml","bin"))
            model_dir_set = "./model"
            model_path_set = os.path.join(model_dir_set, uploaded_file.name)
            with open(model_path_set, "wb") as f:
                f.write(uploaded_file.getbuffer())
        except:
            pass
        model_path = st.text_input("model path : ","./")
        st.write("model_path : ",model_path)
        label_map = st.text_input("model label : ","")
        st.write("label_map : ",label_map)
    #model_path = st.siderbar.header(st.text_input("model path : ","."))
    source_model = st.sidebar.radio("Choose Model type",["Test_Image","Test_Video"])

#if source_model == "Model_Create":
    # 추가

def list_files(startpath):
    tree_structure = ""
    for root, dirs, files in os.walk(startpath):
        level = root.replace(startpath, '').count(os.sep)
        indent = ' ' * 4 * level
        tree_structure += f"{indent}{os.path.basename(root)}/\n"
        subindent = ' ' * 4 * (level + 1)
        for f in files:
            tree_structure += f"{subindent}{f}\n"
    return tree_structure

with st.sidebar:
    selected = st.radio( 'Check_Path' , ['normal','tree'] )
    if selected == "normal":
        st.title(st.write("PATH : ",os.getcwd())
                 ,st.write("subdirectory : ",os.listdir(path="."))
                )
    if selected == "tree":
            user_input = st.text_input("directory list",".")
            tree = list_files(user_input)
            st.text(tree)

input = None
if source_model == "Test_Image":
    st.sidebar.header("Imgae_Upload")
    input = st.sidebar.file_uploader("Choose an image.", type=("jpg","png"))
    
    if input is not None:
        uploaded_image = PIL.Image.open(input)
        uploaded_image_cv =cv2.cvtColor(np.array(uploaded_image), cv2.COLOR_RGB2BGR)
        visualized_image = utils.predict_image(uploaded_image_cv, conf_threshold = conf_threshold)
        st.image(visualized_image, channels = "BGR")

if source_model == "Test_Video":
    try:
        st.sidebar.header("Upload")
        input = st.sidebar.file_uploader("Choose an video.", type=("mp4"))

        if input is not None:
            g = io.BytesIO(input.read())
            temporary_location = "upload.mp4"

            with open(temporary_location, "wb") as out:
                out.write(g.read())

            out.close()
        if temporary_location is not None:
            play_video(temporary_location)
            if st.button("Replay", type="primary"):
                pass
    except:
        pass
