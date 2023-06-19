import cv2
import mediapipe as mp
from keras.models import load_model
import tensorflow as tf

def fer_mediapipe(camera,model,labels_dict):
    mpFaceDetection = mp.solutions.face_detection
    mpDraw = mp.solutions.drawing_utils
    faceDetection = mpFaceDetection.FaceDetection(0.75)

    while True:
        success, img = camera.read()

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = faceDetection.process(imgRGB)

        if results.detections:
            detection = results.detections[0]
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, ic = img.shape
            bbox = (int(bboxC.xmin * iw), int(bboxC.ymin * ih),
                    int(bboxC.width * iw), int(bboxC.height * ih))

            face_img = img[bbox[1]:bbox[1] + bbox[3], bbox[0]:bbox[0] + bbox[2]]
            if face_img.shape[0] > 0 and face_img.shape[1] > 0:
                face_img = cv2.resize(face_img, (48, 48))
                face_gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
                face_gray = face_gray / 255.0
                face_gray = face_gray.reshape(1, 48, 48, 1)
            else:
                print("Error: Failed to extract face region")
                face_gray = None  # Assign a default value

            if face_gray is not None:
                emotion_probs = model.predict(face_gray)[0]
                emotion_id = tf.argmax(emotion_probs)
                emotion_label = labels_dict[emotion_id.numpy()]

                cv2.putText(img, emotion_label, (bbox[0], bbox[1] - 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)
                cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), (255, 0, 255), 2)
                cv2.putText(img, f'{int(detection.score[0] * 100)}%', (bbox[0], bbox[1] - 20), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)
            
        ret, jpeg = cv2.imencode('.jpg', img)
        frame = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')