from start_webcam import *
from haarcascade_detection import *
from fer_mediapipe_detection import *
from human_falling_recognition import *
from face_identification import *

from flask import Flask, render_template, Response, request
from flask_socketio import SocketIO
import mediapipe as mp

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret_key'
socketio = SocketIO(app)
model = load_model('FER_64.5acc_0.99loss.h5')
labels_dict = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Neutral', 5: 'Sad', 6: 'Surprise'}
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
camera = cv2.VideoCapture(0)
action_status = None
name = None
process_current_frame = False

@socketio.on("status")
def detect_falling():
    return falling_detection(camera,socketio)

@socketio.on("verification")
def start_recognision():
    known_face_encodings, known_face_names = FaceRecognition.encode_faces()
    fr = FaceRecognition()
    return fr.run_recognition(camera,socketio,known_face_encodings, known_face_names)
      
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(camera), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/button1')
def button1():
    return render_template('fer-haarcascade.html')

@app.route('/page1')   
def page1():
    return Response(fer_haarcascade(camera,model,labels_dict), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/button2')
def button2():
    return render_template('fer-mediapipe.html')

@app.route('/page2')   
def page2():
    return Response(fer_mediapipe(camera,model,labels_dict), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/button3')
def button3():
    return render_template('falling_detection.html')

@app.route('/page3')
def page3():
    return Response(detect_falling(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/button4')
def button4():
    return render_template('facial_recognition.html')
 
@app.route('/page4')
def page4():
    #global process_current_frame
    #process_current_frame = True
    # known_face_encodings, known_face_names = FaceRecognition.encode_faces()
    # fr = FaceRecognition()
    return Response(start_recognision(), mimetype='multipart/x-mixed-replace; boundary=frame')
    # known_face_encodings, known_face_names = FaceRecognition.encode_faces()
    # return Response(FaceRecognition.run_recognition(known_face_encodings, known_face_names), mimetype='multipart/x-mixed-replace; boundary=frame')
   
if __name__ == '__main__':
    #app.run(debug=True)
    process_current_frame = False
    socketio.start_background_task(falling_detection)  # Run falling_detection in the background
    socketio.run(app, debug=True)