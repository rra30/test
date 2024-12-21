import csv
import os
import numpy as np
import openvino as ov
import matplotlib.pyplot as plt
import cv2
import os
import shutil
import csv
import datetime as dt
a = True
total_data = []
try:
  core = ov.Core()
  model_face = core.read_model(model='./model/face-detection-adas-0001.xml')
  compiled_model_face = core.compile_model(model = model_face, device_name="CPU")
  input_layer_face = compiled_model_face.input(0)
  output_layer_face = compiled_model_face.output(0)
  model_emo = core.read_model(model='./model/emotions-recognition-retail-0003.xml')
  compiled_model_emo = core.compile_model(model = model_emo, device_name="CPU")
  input_layer_emo = compiled_model_emo.input(0)
  output_layer_emo = compiled_model_emo.output(0)
except:
  pass
def preprocess(frame, input_layer_face):
  try:
    N, input_channels, input_height, input_width = input_layer_face.shape
    resized_frame = cv2.resize(frame, (input_width, input_height))
    transposed_frame = resized_frame.transpose(2, 0, 1)
    input_frame = np. expand_dims(transposed_frame, 0)
    return input_frame
  except NameError or error:
     pass
def find_faceboxes(frame, results, confidence_threshold):
  try:
    results = results.squeeze()
    scores = results[:,2]
    boxes = results[:, -4:]
    face_boxes = boxes[scores >= confidence_threshold]
    scores = scores[scores >= confidence_threshold]
    frame_h, frame_w, frame_channels = frame.shape
    face_boxes = face_boxes*np.array([frame_w, frame_h, frame_w, frame_h])
    face_boxes = face_boxes.astype(np.int64)
    return face_boxes, scores
  except NameError or error:
    pass
def draw_emotion(face_boxes, frame, scores):
  try:
    show_frame = frame.copy()
    EMOTION_NAMES = ['neutral', 'happy', 'sad', 'surprise', 'anger']
    for i in range(len(face_boxes)):
      xmin, ymin, xmax, ymax = face_boxes[i]
      face = frame[ymin:ymax, xmin:xmax]
      input_frame = preprocess(face, input_layer_emo)
      results_emo = compiled_model_emo([input_frame])[output_layer_emo]
      results_emo = results_emo.squeeze()
      index = np.argmax(results_emo)
      total_data.append(results_emo)
      text = EMOTION_NAMES[index] + ' ' + str(f"{results_emo}") #str(scores * 100)
      box_color = (255,255,255)
      fontScale = frame.shape[1]/1000
      cv2.putText(show_frame, text, (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, fontScale, (0, 200, 0), 2)
      cv2.rectangle(img=show_frame, pt1=(xmin,ymin), pt2=(xmax,ymax), color=box_color, thickness=2)
    return show_frame
  except NameError or error:
    pass
def Main():
    try:
        confidence_threshold = 0.2
        video_file = st.camera_input("Take a picture")
        if video_file:
            frame = np.array(cv2.imdecode(np.frombuffer(video_file.read(), np.uint8), 1))
            input_frame = preprocess(frame, input_layer_face)
            results = None 
            face_boxes, scores = find_faceboxes(frame, results, confidence_threshold)
            show_frame = draw_emotion(face_boxes, frame, scores)
            st.image(show_frame) 
    except Exception as e:
        st.error(f"Error: {e}")

if __name__ == '__main__':
    Main()
