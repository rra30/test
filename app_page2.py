import streamlit as st
#import test
#import learn
import cv2
import numpy as np
import io
import PIL
import os
from PIL import Image

st.set_page_config(page_title="app_page1", page_icon="📈")
st.sidebar.header("Plotting Demo")

progress_bar = st.sidebar.progress(0)
status_text = st.sidebar.empty()
last_rows = np.random.randn(1, 1)
chart = st.line_chart(last_rows)

for i in range(1, 101):
    new_rows = last_rows[-1, :] + np.random.randn(5, 1).cumsum(axis=0)
    status_text.text("%i%% Complete" % i)
    chart.add_rows(new_rows)
    progress_bar.progress(i)
    last_rows = new_rows
    time.sleep(0.05)

progress_bar.empty()

st.button("Re-run")

st.title("Test-Project-🌚")

source_model = st.sidebar.radio("Select Model",["Model_Create","Model_test"])
model_path = st.sidebar.header(st.text_input("model path : ","."))

if source_model == "Model_test":
    source_model = st.sidebar.radio("Choose Model type",["Test_Image","Test_Video"])
    
#if source_model == "Model_Create":
    # 추가
    
if source_model == "Test_Image":
    st.sidebar.header("Imgae_Upload")
    input = st.sidebar.file_uploader("Choose an image.", type=("jpg","png"))

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


