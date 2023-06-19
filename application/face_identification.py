import face_recognition
import os
from flask_socketio import SocketIO, emit
import numpy as np
import time
import cv2
import math

def face_confidence(face_distance, face_match_threshold=0.6):
    range = (1.0 - face_match_threshold)
    linear_val = (1.0 - face_distance) / (range * 2.0)

    if face_distance > face_match_threshold:
        return str(round(linear_val * 100, 2)) + '%'
    else:
        value = (linear_val + ((1.0 - linear_val) * math.pow((linear_val - 0.5) * 2, 0.2))) * 100
        return str(round(value, 2)) + '%'

class FaceRecognition:
    @staticmethod
    def encode_faces():
        known_face_encodings = []
        known_face_names = []

        for image in os.listdir('faces'):
            face_image = face_recognition.load_image_file(f"faces/{image}")
            face_encoding = face_recognition.face_encodings(face_image)[0]

            known_face_encodings.append(face_encoding)
            known_face_names.append(image)
        print(known_face_names)
        
        return known_face_encodings, known_face_names

    @staticmethod
    def run_recognition(camera,socketio,known_face_encodings, known_face_names):
        global name
        face_locations = []
        face_encodings = []
        face_names = []
        #global process_current_frame
        #process_current_frame = False  # Set initial value to False

        while True:
            ret, frame = camera.read()

            #if process_current_frame:
            # Resize frame of video to 1/4 size for faster face recognition processing
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

            # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
            rgb_small_frame = small_frame[:, :, ::-1]

            # Find all the faces and face encodings in the current frame of video
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            face_names = []
            for face_encoding in face_encodings:
                # See if the face is a match for the known face(s)
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                name = "Unknown"
                confidence = '???'
                # Calculate the shortest distance to face
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)

                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]
                    confidence = face_confidence(face_distances[best_match_index])

                face_names.append(f'{name} ({confidence})')
                print("im here")
            
            #process_current_frame = False

            # Display the results
            for (top, right, bottom, left), name in zip(face_locations, face_names):
                # Scale back up face locations since the frame we detected in was scaled to 1/4 size
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4

                # Create the frame with the name
                cv2.putText(frame, name, (10, 30), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)
                socketio.emit("verification_update", {"verification": name}, namespace="/")
            return
            # Display the resulting image
            #cv2.imshow('Face Recognition', frame)
            # Hit 'q' on the keyboard to quit!
            if cv2.waitKey(1) == ord('q'):
                break
            ret, jpeg = cv2.imencode('.jpg', frame)
            frame = jpeg.tobytes()
            yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')