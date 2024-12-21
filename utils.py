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
def Data_Collection():
    try:
        while a == True:
            try:
                try:
                    cv2.destroyAllWindows()
                except:
                    pass
                try:
                    core = ov.Core()
                    model_face = core.read_model(model='model/face-detection-adas-0001.xml')
                    compiled_model_face = core.compile_model(model = model_face, device_name="CPU")
                    input_layer_face = compiled_model_face.input(0)
                    output_layer_face = compiled_model_face.output(0)
                    model_emo = core.read_model(model='model/emotions-recognition-retail-0003.xml')
                    compiled_model_emo = core.compile_model(model = model_emo, device_name="CPU")
                    input_layer_emo = compiled_model_emo.input(0)
                    output_layer_emo = compiled_model_emo.output(0)
                except:
                    pass
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
                        if (len(total_data) >= 1000):
                            a = False
                            raise Breaking
                            break
                    return show_frame
                except NameError or error:
                    pass
            def Main():
                try:
                    camera = cv2.VideoCapture(source)
                    while(True):
                        ret, frame = camera.read()
                        if not ret:
                            a = True
                            break
                        input_frame = preprocess(frame, input_layer_face)
                        results = compiled_model_face([input_frame])[output_layer_face]
                        face_boxes, scores = find_faceboxes(frame, results, confidence_threshold)
                        show_frame = draw_emotion(face_boxes, frame, scores)
                        cv2.imshow("Webcam", show_frame)
                        if cv2.waitKey(1) & 0xff == ord('q'):
                            a = False
                            raise Breaking
                    camera.release()
                    cv2.destroyAllWindows()
                except NameError or error:
                    pass
            try:
                confidence_threshold = .2
                source = 0
                if __name__ == '__main__':
                    Main()
            except NameError or error:
                pass
        if (a != True):
            raise IndexErr
    except:
        pass
core = ov.Core()
# 얼굴 인식 모델
model_face = core.read_model(model='C:/BrainAI/model/face-detection-adas-0001.xml')
compiled_model_face = core.compile_model(model = model_face, device_name="CPU")
input_layer_face = compiled_model_face.input(0)
output_layer_face = compiled_model_face.output(0)
#감정 인식 모델
model_emo = core.read_model(model='C:/BrainAI/model/emotions-recognition-retail-0003.xml')
compiled_model_emo = core.compile_model(model = model_emo, device_name="CPU")
input_layer_emo = compiled_model_emo.input(0)
output_layer_emo = compiled_model_emo.output(0)
# 나이, 성별 인식 모델
model_ag = core.read_model(model='C:/BrainAI/model/age-gender-recognition-retail-0013.xml')
compiled_model_ag = core.compile_model(model = model_ag, device_name="CPU")
input_layer_ag = compiled_model_ag.input(0)
output_layer_ag = compiled_model_ag.output
def preprocess(image, input_layer_face):
	N, input_channels, input_height, input_width = input_layer_face.shape
	resized_image = cv2.resize(image, (input_width, input_height))
	transposed_image = resized_image.transpose(2, 0, 1)
	input_image = np. expand_dims(transposed_image, 0)
	return input_image
def find_faceboxes(image, results, confidence_threshold):
	results = results.squeeze()
	
	scores = results[:,2]
	boxes = results[:, -4:]
	face_boxes = boxes[scores >= confidence_threshold]
	scores = scores[scores >= confidence_threshold]
	image_h, image_w, image_channels = image.shape
	face_boxes = face_boxes*np.array([image_w, image_h, image_w, image_h])
	face_boxes = face_boxes.astype(np.int64)
	return face_boxes, scores
def draw_faceboxes(image, face_boxes, scores):
	show_image = image.copy()
	for i in range(len(face_boxes)):
		
		xmin, ymin, xmax, ymax = face_boxes[i]
		cv2.rectangle(img=show_image, pt1=(xmin,ymin), pt2=(xmax,ymax), color=(0,200,0), thickness=2)
	return show_image
def draw_emotions(face_boxes, image, show_image):
    EMOTION_NAMES = ['neutral', 'happy', 'sad', 'surprise', 'anger']
    for i in range(len(face_boxes)):
        xmin, ymin, xmax, ymax = face_boxes[i]
        face = image[ymin:ymax, xmin:xmax]
        input_image = preprocess(face, input_layer_emo)
        results_emo = compiled_model_emo([input_image])[output_layer_emo]
        results_emo = results_emo.squeeze()
        index = np.argmax(results_emo)
        text = EMOTION_NAMES[index]
        cv2.putText(show_image, text, (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 200, 0), 2)
def draw_age_gender(face_boxes, image):
    EMOTION_NAMES = ['neutral', 'happy', 'sad', 'surprise', 'anger']
    show_image = image.copy()
    for i in range(len(face_boxes)):
        xmin, ymin, xmax, ymax = face_boxes[i]
        xmin = max(0, xmin)
        ymin = max(0, ymin)
        xmax = min(image.shape[1], xmax)
        ymax = min(image.shape[0], ymax)
        face = image[ymin:ymax, xmin:xmax]
        #--- emotion ---
        input_image = preprocess(face, input_layer_emo)
        results_emo = compiled_model_emo([input_image])[output_layer_emo]
        results_emo = results_emo.squeeze()
        index = np.argmax(results_emo)
        if index >= len(EMOTION_NAMES):
            index = 0
        #--- age and gender ---
        input_image_ag = preprocess(face, input_layer_ag)
        results_ag = compiled_model_ag([input_image_ag])
        age, gender = results_ag[1], results_ag[0]
        age = np.squeeze(age)
        age = int(age * 100)
        gender = np.squeeze(gender)
        if gender[0] >= 0.65:
            gender_label = "female"
            box_color = (200, 200, 0)
        elif gender[1] >= 0.65:
            gender_label = "male"
            box_color = (0, 200, 200)
        else:
            gender_label = "unknown"
            box_color = (200, 200, 200)
        fontScale = max(0.5, image.shape[1] / 750)
        text = f"{gender_label} {age} {EMOTION_NAMES[index]}"
        cv2.putText(show_image, text, (xmin, ymin - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, fontScale, (0, 200, 0), 2)
        cv2.rectangle(show_image, (xmin, ymin), (xmax, ymax), box_color, 2)
    return show_image
def predict_image(image, conf_threshold):
    input_image = preprocess(image, input_layer_face)
    results = compiled_model_face([input_image])[output_layer_face]
    face_boxes, scores = find_faceboxes(image, results, conf_threshold)
    visualize_image = draw_age_gender(face_boxes, image)
    return visualize_image
