import csv
import os
import numpy as np
import openvino as ov
import cv2
import os
import shutil
import csv
import datetime as dt
a = True
total_data = []
csv_name = None

if (csv_name == None or csv_name == ""):
    csv_name = now.strftime('20%y_%m_%d_%H_%M_%S')
