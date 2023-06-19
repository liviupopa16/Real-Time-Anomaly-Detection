import math
from flask_socketio import SocketIO, emit
import mediapipe as mp
import numpy as np
import time
import cv2


mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
def calculate_angle(a,b,c):
    
    a = np.array(a) # First
    b = np.array(b) # Mid
    c = np.array(c) # End
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle >180.0:
        angle = 360-angle
        
    return angle 

def falling_detection(camera,socketio):
    #cap = cv2.VideoCapture("video_1.mp4")
    #cap = cv2.VideoCapture("queda.mp4")
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    ## Setup mediapipe instance
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        # Declarare de vectori pentru history buffer
        cog_history = []
        feet_cog_history = []
        nose_history_y = []
        nose_history_x = []
        left_shoulder_history_x = []
        detection_duration = 4 # durata pentru adaugare de frame-uri in history buffer
        sample_frames = 35 # nr de frame-uri pentru care se calculeaza avg
        
        global action_status
        while True:
            success, frame = camera.read()
            
            # Recolor image to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
        
            # Make detection
            results = pose.process(image)
            
            try:
                  
                landmarks = results.pose_landmarks.landmark
                nose_landmark = landmarks[0]
                h, w, _ = frame.shape
                #print(h,w)
                # variabile unde stocam coordonatele x si y pentru anumite landmark-uri
                nose = results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE]
                
                left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
                left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                
                right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
                right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
                right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                
                # Calculate angle
                left_knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
                right_knee_angle = calculate_angle(right_hip, right_knee, right_ankle)
                left_hip_angle = calculate_angle(left_shoulder,left_hip,left_knee)
                right_hip_angle = calculate_angle(right_shoulder,right_hip,right_knee)

                # Vizualizare on screen unghiuri
                cv2.putText(image, str((int)(left_knee_angle)),
                            tuple(np.multiply(left_knee, [640, 480]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(image, str((int)(right_knee_angle)),
                            tuple(np.multiply(right_knee, [640, 480]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(image, str((int)(left_hip_angle)),
                            tuple(np.multiply(left_hip, [640, 480]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(image, str((int)(right_hip_angle)),
                            tuple(np.multiply(right_hip, [640, 480]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

                left_shoulder_x = int(left_shoulder[0]*w)
                left_shoulder_y = int(left_shoulder[1]*h)
                # Extract x and y coordinates of the nose landmark
                nose_x = int(nose_landmark.x * w)
                nose_y = int(nose_landmark.y * h)
                #print("NOSE: ",nose_x,nose_y)
                # Draw a circle at the nose landmark position
                cv2.circle(image, (nose_x, nose_y), 5, (0, 255, 0), -1)
                # Calculate center of gravity for feet
                foot_landmarks = [landmarks[27], landmarks[28], landmarks[29], landmarks[30], landmarks[31], landmarks[32]]
                feet_x = sum(landmark.x for landmark in foot_landmarks) / len(foot_landmarks)
                feet_y = sum(landmark.y for landmark in foot_landmarks) / len(foot_landmarks)
                
                # Step 6: Draw center of gravity for feet
                image_height, image_width, _ = frame.shape
                center_x = int(feet_x * image_width)
                center_y = int(feet_y * image_height)
                
                cog_x = sum([lm.x for lm in landmarks]) / len(landmarks)
                cog_y = sum([lm.y for lm in landmarks]) / len(landmarks)
                # print(cog_x)  
                # print(cog_y)                                               
                
                cog_px_x = int(cog_x * w)
                cog_px_y = int(cog_y * h)
                # print(cog_px_x)
                # print(cog_px_y)
                
                cv2.circle(image, (cog_px_x, cog_px_y), 20, (0, 255, 0), -1)
                cv2.circle(image, (center_x, center_y), 15, (0, 255, 0), -1)
                
                # Step 7: Update COG history
                #salvam in vectorii de history valoarea curenta si momentul la care a fost adaugat
                current_time = time.time()
                cog_history.append((current_time, cog_px_y))
                feet_cog_history.append((current_time, center_y))
                nose_history_y.append((current_time,nose_y))
                nose_history_x.append((current_time,nose_x))
                left_shoulder_history_x.append((current_time,left_shoulder_x))
                # print(len(cog_history))
                # print(len(feet_cog_history))
                
                
                # print("BODY: ",cog_px_x,cog_px_y)
                # print("FEET: ",center_x,center_y)
                
                # verific daca buffer-ul e gol, daca nu verific daca distanta de timp intre primul element adaugat
                # in buffer si ultimul e mai mare decat durata de timp cu care trebuie sa compar (detection_duration),
                # caz in care depaseste, eliminam elementul de pe pozitia 0 din buffer
                while len(cog_history) > 0 and current_time - cog_history[0][0] > detection_duration:
                    cog_history.pop(0)
                    feet_cog_history.pop(0)
                    nose_history_y.pop(0)
                    nose_history_x.pop(0)
                    left_shoulder_history_x.pop(0)

                # if left_hip_angle > 75 and left_hip_angle < 105 and right_hip_angle > 75 and right_hip_angle < 105:
                #     if left_knee_angle > 75 and left_knee_angle < 105 and right_knee_angle > 75 and right_knee_angle < 105:
                #         action_status = "Sitting"
                # else: 
                #     if left_hip_angle > 160 and right_hip_angle > 160:
                #         if left_knee_angle > 160 and right_knee_angle > 160:
                #             action_status = "Standing"
                #     else:
                #         if left_hip_angle < 50 and right_hip_angle < 50:
                #            if left_knee_angle < 40 and right_knee_angle < 40:
                #                 action_status="Crouching" 
                #if (left_knee_angle < 160 or right_knee_angle < 160) and action_status == "Standing":
                if len(nose_history_y)>0:
                    # verific daca numarul de elemente din history e mai mare sau egal decat numarul de frame-uri necesar pentru calcularea avg-ului
                    if len(nose_history_y) >= sample_frames:
                        #calculez avg frames pentru primele 35 si ultimele 35 din buffer
                        first_15_nose = nose_history_y[:sample_frames]
                        averagefirst_nose_y = (int)(sum(nose[1] for nose in first_15_nose) / len(first_15_nose))
                        
                        last_15_nose = nose_history_y[-sample_frames:]
                        averagelast_nose_y = (int)(sum(nose[1] for nose in last_15_nose) / len(last_15_nose))
                        #print("avg nose - Y",averagelast_nose_y - averagefirst_nose_y)
                        
                    if len(nose_history_x) >= sample_frames:
                        first_15_nose_x = nose_history_x[:sample_frames]
                        averagefirst_nose_x = (int)(sum(nose[1] for nose in first_15_nose_x) / len(first_15_nose_x))
                        
                        last_15_nose_x = nose_history_x[-sample_frames:]
                        averagelast_nose_x = (int)(sum(nose[1] for nose in last_15_nose_x) / len(last_15_nose_x))
                        #print("avg nose - X",abs(averagelast_nose_x - averagefirst_nose_x))
                        
                    # daca diferenta dintre media pentru ultimele frame uri si primele e mai mare de 150, trec la urmatoarea conditie   
                    if ((averagelast_nose_y - averagefirst_nose_y) > 150):
                        # if (abs(averagelast_nose_x - averagefirst_nose_x)) < 25:
                        #     if (int)(left_knee_angle) < 50 and (int)(right_knee_angle) < 50:
                        #         first_15_left_shoulder_x = left_shoulder_history_x[:sample_frames]
                        #         averagefirst_left_shoulder_x = (int)(sum(left_shoulder[1] for left_shoulder in first_15_left_shoulder_x) / len(first_15_left_shoulder_x))
                                
                        #         last_15_left_shoulder_x = left_shoulder_history_x[-sample_frames:]
                        #         averagelast_left_shoulder_x = (int)(sum(left_shoulder[1] for left_shoulder in last_15_left_shoulder_x) / len(last_15_left_shoulder_x))
                                
                        #         if (abs(averagelast_left_shoulder_x - averagefirst_left_shoulder_x)) < 25:                           
                        #             action_status= "Crouching"
                        # else: 
                            # Step 8: Fall detection
                        
                            if len(cog_history) > 0 and len(feet_cog_history)>0:
                                if len(cog_history) >= sample_frames:
                                    first_15_cog = cog_history[:sample_frames]
                                    averagefirst_cog_y = (int)(sum(cog[1] for cog in first_15_cog) / len(first_15_cog))
                                    
                                    last_15_cog = cog_history[-sample_frames:]
                                    averagelast_cog_y = (int)(sum(cog[1] for cog in last_15_cog) / len(last_15_cog))
                                    
                                if len(feet_cog_history) >= sample_frames:
                                    first_15_feet_cog = feet_cog_history[:sample_frames]
                                    averagefirst_feetcog_y = (int)(sum(cog[1] for cog in first_15_feet_cog) / len(first_15_feet_cog))
                                    
                                    last_15_feet_cog = feet_cog_history[-sample_frames:]
                                    averagelast_feetcog_y = (int)(sum(cog[1] for cog in last_15_feet_cog) / len(last_15_feet_cog))
                                    
                                # print("BODY ", averagefirst_cog_y, averagelast_cog_y)
                                # print("FEET: ", averagefirst_feetcog_y, averagelast_feetcog_y)
                                # calculez diferenta intre avg-uri dintre distantele dintre cog si cog_feet
                                avg_first = averagefirst_feetcog_y - averagefirst_cog_y
                                avg_last = averagelast_feetcog_y - averagelast_cog_y
                                # print(avg_first,avg_last)
                                # daca diferenta dintre avg-uri e mai mare de 100, detectam cadere
                                if avg_first - avg_last >100:
                                        action_status = "Falling detected"
                                        socketio.emit("status_update", {"status": action_status}, namespace="/")
                    else:
                        #daca unghiurile de la solduri si genunchi sunt mai mari de 150 atunci detectam normal activity
                        if (left_hip_angle > 150 or right_hip_angle > 150) and (left_knee_angle > 150 or right_knee_angle > 150):
                            action_status = "Normal activity"
                            socketio.emit("status_update", {"status": action_status}, namespace="/")
        
            except:
                pass
        
            # Recolor back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            # Render detections
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                    mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                    mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                    )               
            # cv2.rectangle(image, (0,0), (260,70), (245,117,16), -1)

            # Rep data
            # cv2.putText(image, 'status', (8,18),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 1, cv2.LINE_AA)
            # cv2.putText(image, action_status,
            #             (8,53),
            #             cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
            #cv2.imshow('REALTIME FALLING DETECTION', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
            ret, jpeg = cv2.imencode('.jpg', image)
            frame = jpeg.tobytes()
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')   