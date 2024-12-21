import streamlit as st
import numpy as np
import openvino as ov
import cv2
import os
import time
import datetime as dt

# Initialize the CSV directory variable
csv_directory = None
csv_name = None

# Look for a CSV file in the directory
directory = os.listdir()

# Find the first CSV file in the directory and store its name
for filename in directory:
    if filename.endswith(".csv"):
        csv_directory = filename
        break  # Stop once we find the first CSV file

# If no CSV file is found, generate a default filename based on the current time
if csv_directory is None:
    now = dt.datetime.now()
    csv_name = now.strftime('20%y_%m_%d_%H_%M_%S.csv')  # Use current timestamp for filename
else:
    csv_name = csv_directory  # Use the found CSV file name

st.set_page_config(
    page_title="I want go to home",
    page_icon="✋",
    layout="centered",
    initial_sidebar_state="expanded")

st.title("Welcome to '살자 예방 게이트 키퍼' ✋")
st.sidebar.header("Setting")

# Sidebar content and user options
if csv_directory is None:  # If no CSV file exists, show Data Collection option
    source_model = st.sidebar.radio("No data available. You can collect data or utilize existing data.", ["Data Collection"])
    if source_model == "Data Collection":
        try:
            import Collection  # Ensure the Collection module exists
            Collection.Main()  # Assuming the Collection module has a Main function
        except ImportError as e:
            st.error(f"Error importing Collection module: {e}")
        except Exception as e:
            st.error(f"Unexpected error: {e}")

else:  # If a CSV file exists, offer both options
    source_model = st.sidebar.radio("Select an option:", ["Data Collection", "Utilize collected data"])

    if source_model == "Utilize collected data":
        try:
            import utils  # Ensure the utils module exists
            utils.Main()  # Assuming utils has a Main function to utilize the data
        except ImportError as e:
            st.error(f"Error importing utils module: {e}")
        except Exception as e:
            st.error(f"Unexpected error: {e}")

    elif source_model == "Data Collection":
        try:
            import Collection  # Assuming Collection module is used for collecting data
            Collection.Main()
        except ImportError as e:
            st.error(f"Error importing Collection module: {e}")
        except Exception as e:
            st.error(f"Unexpected error: {e}")
