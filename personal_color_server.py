import flask

from flask import Flask, render_template, Response
import cv2
import numpy as np
import time
import datetime
import sys

from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
import matplotlib.pyplot as plt
import numpy as np
import os

from keras.models import load_model
import dlib

img_height = 256
img_width = 256
                    
predict_dictionary = {0:'autumn_warm',1:'spring_warm',2:'summer_cool', 3:"winter_cool"}

model_path = "./resnet152v2_best_2.h5"
model = load_model(model_path)

faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
predictor = dlib.shape_predictor("./shape_predictor_68_face_landmarks.dat")

JAWLINE_POINTS = list(range(0, 17))
RIGHT_EYEBROW_POINTS = list(range(17, 22))
LEFT_EYEBROW_POINTS = list(range(22, 27))
NOSE_POINTS = list(range(27, 36))
RIGHT_EYE_POINTS = list(range(36, 42))
LEFT_EYE_POINTS = list(range(42, 48))
MOUTH_OUTLINE_POINTS = list(range(48, 61))
MOUTH_INNER_POINTS = list(range(61, 68))

num = 3

app = Flask(__name__)

@app.route('/')
def index():
    """Video streaming home page."""
    now = datetime.datetime.now()
    timeString = now.strftime("%Y-%m-%d %H:%M")
    templateData = {
            'title':'Image Streaming',
            'time': timeString
            }
    return render_template('index.html', **templateData)

def gen_frames():

    camera = cv2.VideoCapture(0)
    time.sleep(0.2)
    lastTime = time.time()*1000.0

    while True:
        ret, image = camera.read()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
       
        faces = faceCascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=6)

        for (x, y, w, h) in faces:
            # 오픈 CV 이미지를 dlib용 사각형으로 변환하고
            dlib_rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
            # 랜드마크 포인트들 지정
            landmarks = np.matrix([[p.x, p.y] for p in predictor(image, dlib_rect).parts()])
            # 원하는 포인트들을 넣는다, 지금은 전부
            # landmarks_display = landmarks[0:68]       

            aa = ((landmarks[RIGHT_EYEBROW_POINTS][0,0], landmarks[LEFT_EYEBROW_POINTS][0,1]), (landmarks[LEFT_EYEBROW_POINTS][4,0], landmarks[JAWLINE_POINTS][8,1]))
            img = image[aa[0][1]:aa[1][1], aa[0][0]:aa[1][0]]

            # original = load_img(img, target_size = (img_height,img_width))

            original = cv2.resize(img, (img_height,img_width))
            numpy_image = img_to_array(original)
            image_batch = np.expand_dims(numpy_image , axis = 0)

            predict = np.argmax(model.predict(image_batch/255.))
            ss = predict_dictionary[predict]

            print(ss)
            cv2.putText(img, ss, (10, 25),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("Frame", img)

            if ss != None:
                cv2.destroyAllWindows()
                break
    
        key = cv2.waitKey(1) & 0xFF
            
        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            
            cv2.destroyAllWindows()
            break

        ret, buffer = cv2.imencode('.jpg', img)
        frame = buffer.tobytes()
        
        yield (b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    
@app.route('/video_feed')

def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0')       