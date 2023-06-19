import cv2
import numpy as np
from keras.models import load_model

frontalfaceDetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
profilefaceDetect = cv2.CascadeClassifier('haarcascade_profileface.xml')

def fer_haarcascade(camera,model,labels_dict):
    while True:
        ret, frame = camera.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        frontal_faces = frontalfaceDetect.detectMultiScale(gray, 1.3, 5)
        if len(frontal_faces) > 0:
            for x, y, w, h in frontal_faces:
                sub_face_img = gray[y:y+h, x:x+w]
                resized = cv2.resize(sub_face_img, (48, 48))
                normalize = resized / 255.0
                reshaped = np.reshape(normalize, (1, 48, 48, 1))
                result = model.predict(reshaped)
                label = np.argmax(result, axis=1)[0]
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                cv2.putText(frame, labels_dict[label], (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                #cv2.putText(frame, "frontal face", (7, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            profile_faces = profilefaceDetect.detectMultiScale(gray, 1.3, 5)
            if len(profile_faces) > 0:
                for x, y, w, h in profile_faces:
                    sub_face_img = gray[y:y+h, x:x+w]
                    resized = cv2.resize(sub_face_img, (48, 48))
                    normalize = resized / 255.0
                    reshaped = np.reshape(normalize, (1, 48, 48, 1))
                    result = model.predict(reshaped)
                    label = np.argmax(result, axis=1)[0]
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                    cv2.putText(frame, labels_dict[label], (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    #cv2.putText(frame, "profile face", (7, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')