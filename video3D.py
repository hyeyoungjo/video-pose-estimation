# Video Pose Estimation
# This script takes a fitness video (with single person), runs 2D/3D pose estimation using BlazePose and then outputs json files of 2D/3D coordinates of key joints and facial features.
# For more information on BlazePose, see https://google.github.io/mediapipe/solutions/pose.html

import cv2
import mediapipe as mp
import numpy as np
import sys
import codecs
import json

# source
# sample video is from the YouTube yoga video(https://youtu.be/Wkmarh2Ps_o)
# video quality is degraded for smaller file size
clipName = 'YogaBoy'

# output
video_path = './output/pose_3D2D_'+clipName+'.mp4'
npz_path = './output/pose_3D_'+clipName+'.npz'
json_path = './output/pose_3D_'+clipName+'.json'  # your path variable
npz_path_screen = './output/pose_2D_'+clipName+'.npz'
json_path_screen = './output/pose_2D_'+clipName+'.json' # your path variable

# load video path
cap = cv2.VideoCapture(
    './src/'+clipName+'.mp4')  

# set resolution (HD)
video_width = 1920
video_height = 1080
screen_width = 1920
screen_height = 1080

# save video path
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # for video codec (if it does not work, try 'XVID')
out = cv2.VideoWriter(video_path, fourcc, 30, (video_width, video_height))

# get total frame
totalframe = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# get pose estimation model and drawing utilities
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# check if file exist
if (cap.isOpened() == False):
    print("Error opening video stream or file")

# define array for 3D pose estimation (x, y, z)
frame_index = np.array([])
joint_a = np.empty((0, 3))  
joint_b = np.empty((0, 3))  

J_nose = np.empty((0, 3))  

J_eyeInner_L = np.empty((0, 3))  
J_eye_L = np.empty((0, 3))  
J_eyeOuter_L = np.empty((0, 3))  

J_eyeInner_R = np.empty((0, 3))  
J_eye_R = np.empty((0, 3))  
J_eyeOuter_R = np.empty((0, 3))  

J_ear_L = np.empty((0, 3))  
J_ear_R = np.empty((0, 3))  

J_mouth_L = np.empty((0, 3))  
J_mouth_R = np.empty((0, 3))  

J_shoulder_L = np.empty((0, 3))  
J_elbow_L = np.empty((0, 3))  
J_wrist_L = np.empty((0, 3))  
J_pinky_L = np.empty((0, 3))  
J_index_L = np.empty((0, 3))  
J_thumb_L = np.empty((0, 3))  

J_shoulder_R = np.empty((0, 3))  
J_elbow_R = np.empty((0, 3))  
J_wrist_R = np.empty((0, 3))  
J_pinky_R = np.empty((0, 3))  
J_index_R = np.empty((0, 3))  
J_thumb_R = np.empty((0, 3))  

J_hip_L = np.empty((0, 3))  
J_knee_L = np.empty((0, 3))  
J_ankle_L = np.empty((0, 3))  
J_heel_L = np.empty((0, 3))  
J_footIndex_L = np.empty((0, 3))  

J_hip_R = np.empty((0, 3))  
J_knee_R = np.empty((0, 3))  
J_ankle_R = np.empty((0, 3))  
J_heel_R = np.empty((0, 3))  
J_footIndex_R = np.empty((0, 3))  

# define array for 2D pose estimation (x, y, 0)
frame_index_screen = np.array([])
joint_a_screen = np.empty((0, 3))  
joint_b_screen = np.empty((0, 3))  

J_nose_screen = np.empty((0, 3))  

J_eyeInner_L_screen = np.empty((0, 3))  
J_eye_L_screen = np.empty((0, 3))  
J_eyeOuter_L_screen = np.empty((0, 3))  

J_eyeInner_R_screen = np.empty((0, 3))  
J_eye_R_screen = np.empty((0, 3))  
J_eyeOuter_R_screen = np.empty((0, 3))  

J_ear_L_screen = np.empty((0, 3))  
J_ear_R_screen = np.empty((0, 3))  

J_mouth_L_screen = np.empty((0, 3))  
J_mouth_R_screen = np.empty((0, 3))  

J_shoulder_L_screen = np.empty((0, 3))  
J_elbow_L_screen = np.empty((0, 3))  
J_wrist_L_screen = np.empty((0, 3))  
J_pinky_L_screen = np.empty((0, 3))  
J_index_L_screen = np.empty((0, 3))  
J_thumb_L_screen = np.empty((0, 3))  

J_shoulder_R_screen = np.empty((0, 3))  
J_elbow_R_screen = np.empty((0, 3))  
J_wrist_R_screen = np.empty((0, 3))  
J_pinky_R_screen = np.empty((0, 3))  
J_index_R_screen = np.empty((0, 3))  
J_thumb_R_screen = np.empty((0, 3))  

J_hip_L_screen = np.empty((0, 3))  
J_knee_L_screen = np.empty((0, 3))  
J_ankle_L_screen = np.empty((0, 3))  
J_heel_L_screen = np.empty((0, 3))  
J_footIndex_L_screen = np.empty((0, 3))  

J_hip_R_screen = np.empty((0, 3))  
J_knee_R_screen = np.empty((0, 3))  
J_ankle_R_screen = np.empty((0, 3))  
J_heel_R_screen = np.empty((0, 3))  
J_footIndex_R_screen = np.empty((0, 3))  

# run pose estimation (check below for quality setting)
# https://google.github.io/mediapipe/solutions/pose.html#pose-estimation-quality
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5, model_complexity=2) as pose:
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            # get frame
            fcnt = cap.get(cv2.CAP_PROP_POS_FRAMES)

            # color BGR to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            # Make detection
            results = pose.process(image)

            # Recolor image ((mediapipe)RGB -> (cv2)BGR)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Extract landmarks_world
            #landmarks_world = results.pose_world_landmarks_world.landmark

            try:
                # 2D pose pose estimation result
                # landmarks_screen = results.pose_landmarks_world.landmark
                landmarks_basic = results.pose_landmarks.landmark
                # 3D pose pose estimation result
                # landmarks_world = results.pose_world_landmarks_world.landmark
                landmarks = results.pose_world_landmarks.landmark

                # 3D result per landmark
                nose = landmarks[mp_pose.PoseLandmark.NOSE]
                left_eye_inner = landmarks[mp_pose.PoseLandmark.LEFT_EYE_INNER]
                left_eye = landmarks[mp_pose.PoseLandmark.LEFT_EYE]
                left_eye_outer = landmarks[mp_pose.PoseLandmark.LEFT_EYE_OUTER]
                right_eye_inner = landmarks[mp_pose.PoseLandmark.RIGHT_EYE_INNER]
                right_eye = landmarks[mp_pose.PoseLandmark.RIGHT_EYE]
                right_eye_outer = landmarks[mp_pose.PoseLandmark.RIGHT_EYE_OUTER]
                left_ear = landmarks[mp_pose.PoseLandmark.LEFT_EAR]
                right_ear = landmarks[mp_pose.PoseLandmark.RIGHT_EAR]
                left_mouth = landmarks[mp_pose.PoseLandmark.MOUTH_LEFT]
                right_mouth = landmarks[mp_pose.PoseLandmark.MOUTH_RIGHT]
                left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
                left_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW]
                left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
                left_pinky = landmarks[mp_pose.PoseLandmark.LEFT_PINKY]
                left_index = landmarks[mp_pose.PoseLandmark.LEFT_INDEX]
                left_thumb = landmarks[mp_pose.PoseLandmark.LEFT_THUMB]
                right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
                right_elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW]
                right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]
                right_pinky = landmarks[mp_pose.PoseLandmark.RIGHT_PINKY]
                right_index = landmarks[mp_pose.PoseLandmark.RIGHT_INDEX]
                right_thumb = landmarks[mp_pose.PoseLandmark.RIGHT_THUMB]
                left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
                left_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE]
                left_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE]
                left_heel = landmarks[mp_pose.PoseLandmark.LEFT_HEEL]
                left_footIndex = landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX]
                right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
                right_knee = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE]
                right_ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE]
                right_heel = landmarks[mp_pose.PoseLandmark.RIGHT_HEEL]
                right_footIndex = landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX]
                # save result to 3D array
                J_nose = np.append(J_nose, np.array(
                    [[nose.x, nose.y, nose.z]]), axis=0)
                J_eyeInner_L = np.append(J_eyeInner_L, np.array(
                    [[left_eye_inner.x, left_eye_inner.y, left_eye_inner.z]]), axis=0)
                J_eye_L = np.append(J_eye_L, np.array(
                    [[left_eye.x, left_eye.y, left_eye.z]]), axis=0)
                J_eyeOuter_L = np.append(J_eyeOuter_L, np.array(
                    [[left_eye_outer.x, left_eye_outer.y, left_eye_outer.z]]), axis=0)
                J_eyeInner_R = np.append(J_eyeInner_R, np.array(
                    [[right_eye_inner.x, right_eye_inner.y, right_eye_inner.z]]), axis=0)
                J_eye_R = np.append(J_eye_R, np.array(
                    [[right_eye.x, right_eye.y, right_eye.z]]), axis=0)
                J_eyeOuter_R = np.append(J_eyeOuter_R, np.array(
                    [[right_eye_outer.x, right_eye_outer.y, right_eye_outer.z]]), axis=0)
                J_ear_L = np.append(J_ear_L, np.array(
                    [[left_ear.x, left_ear.y, left_ear.z]]), axis=0)
                J_ear_R = np.append(J_ear_R, np.array(
                    [[right_ear.x, right_ear.y, right_ear.z]]), axis=0)
                J_mouth_L = np.append(J_mouth_L, np.array(
                    [[left_mouth.x, left_mouth.y, left_mouth.z]]), axis=0)
                J_mouth_R = np.append(J_mouth_R, np.array(
                    [[right_mouth.x, right_mouth.y, right_mouth.z]]), axis=0)
                J_shoulder_L = np.append(J_shoulder_L, np.array(
                    [[left_shoulder.x, left_shoulder.y, left_shoulder.z]]), axis=0)
                J_elbow_L = np.append(J_elbow_L, np.array(
                    [[left_elbow.x, left_elbow.y, left_elbow.z]]), axis=0)
                J_wrist_L = np.append(J_wrist_L, np.array(
                    [[left_wrist.x, left_wrist.y, left_wrist.z]]), axis=0)
                J_pinky_L = np.append(J_pinky_L, np.array(
                    [[left_pinky.x, left_pinky.y, left_pinky.z]]), axis=0)
                J_index_L = np.append(J_index_L, np.array(
                    [[left_index.x, left_index.y, left_index.z]]), axis=0)
                J_thumb_L = np.append(J_thumb_L, np.array(
                    [[left_thumb.x, left_thumb.y, left_thumb.z]]), axis=0)
                J_shoulder_R = np.append(J_shoulder_R, np.array(
                    [[right_shoulder.x, right_shoulder.y, right_shoulder.z]]), axis=0)
                J_elbow_R = np.append(J_elbow_R, np.array(
                    [[right_elbow.x, right_elbow.y, right_elbow.z]]), axis=0)
                J_wrist_R = np.append(J_wrist_R, np.array(
                    [[right_wrist.x, right_wrist.y, right_wrist.z]]), axis=0)
                J_pinky_R = np.append(J_pinky_R, np.array(
                    [[right_pinky.x, right_pinky.y, right_pinky.z]]), axis=0)
                J_index_R = np.append(J_index_R, np.array(
                    [[right_index.x, right_index.y, right_index.z]]), axis=0)
                J_thumb_R = np.append(J_thumb_R, np.array(
                    [[right_thumb.x, right_thumb.y, right_thumb.z]]), axis=0)
                J_hip_L = np.append(J_hip_L, np.array(
                    [[left_hip.x, left_hip.y, left_hip.z]]), axis=0)
                J_knee_L = np.append(J_knee_L, np.array(
                    [[left_knee.x, left_knee.y, left_knee.z]]), axis=0)
                J_ankle_L = np.append(J_ankle_L, np.array(
                    [[left_ankle.x, left_ankle.y, left_ankle.z]]), axis=0)
                J_heel_L = np.append(J_heel_L, np.array(
                    [[left_heel.x, left_heel.y, left_heel.z]]), axis=0)
                J_footIndex_L = np.append(J_footIndex_L, np.array(
                    [[left_footIndex.x, left_footIndex.y, left_footIndex.z]]), axis=0)
                J_hip_R = np.append(J_hip_R, np.array(
                    [[right_hip.x, right_hip.y, right_hip.z]]), axis=0)
                J_knee_R = np.append(J_knee_R, np.array(
                    [[right_knee.x, right_knee.y, right_knee.z]]), axis=0)          
                J_ankle_R = np.append(J_ankle_R, np.array(
                    [[right_ankle.x, right_ankle.y, right_ankle.z]]), axis=0)
                J_heel_R = np.append(J_heel_R, np.array(
                    [[right_heel.x, right_heel.y, right_heel.z]]), axis=0)
                J_footIndex_R = np.append(J_footIndex_R, np.array(
                    [[right_footIndex.x, right_footIndex.y, right_footIndex.z]]), axis=0)

                # 2D result per landmark
                # nose
                nose_XY = [landmarks_basic[mp_pose.PoseLandmark.NOSE.value].x, landmarks_basic[mp_pose.PoseLandmark.NOSE.value].y]
                nose_XY_toScreen = np.multiply(nose_XY, [screen_width, screen_height])
                # left shoulder
                left_shoulder_XY = [landmarks_basic[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks_basic[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                left_shoulder_XY_toScreen = np.multiply(left_shoulder_XY, [screen_width, screen_height])
                # left elbow
                left_elbow_XY = [landmarks_basic[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks_basic[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                left_elbow_XY_toScreen = np.multiply(left_elbow_XY, [screen_width, screen_height])
                # left wrist
                left_wrist_XY = [landmarks_basic[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks_basic[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                left_wrist_XY_toScreen = np.multiply(left_wrist_XY, [screen_width, screen_height])
                # right shoulder
                right_shoulder_XY = [landmarks_basic[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks_basic[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                right_shoulder_XY_toScreen = np.multiply(right_shoulder_XY, [screen_width, screen_height])
                # right elbow
                right_elbow_XY = [landmarks_basic[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, landmarks_basic[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                right_elbow_XY_toScreen = np.multiply(right_elbow_XY, [screen_width, screen_height])
                # right wrist
                right_wrist_XY = [landmarks_basic[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, landmarks_basic[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
                right_wrist_XY_toScreen = np.multiply(right_wrist_XY, [screen_width, screen_height])
                # left hip
                left_hip_XY = [landmarks_basic[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks_basic[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                left_hip_XY_toScreen = np.multiply(left_hip_XY, [screen_width, screen_height])
                # left knee
                left_knee_XY = [landmarks_basic[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks_basic[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                left_knee_XY_toScreen = np.multiply(left_knee_XY, [screen_width, screen_height])
                # left ankle
                left_ankle_XY = [landmarks_basic[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks_basic[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
                left_ankle_XY_toScreen = np.multiply(left_ankle_XY, [screen_width, screen_height])
                # right hip
                right_hip_XY = [landmarks_basic[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks_basic[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                right_hip_XY_toScreen = np.multiply(right_hip_XY, [screen_width, screen_height])
                # right knee
                right_knee_XY = [landmarks_basic[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, landmarks_basic[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
                right_knee_XY_toScreen = np.multiply(right_knee_XY, [screen_width, screen_height])
                # right ankle
                right_ankle_XY = [landmarks_basic[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, landmarks_basic[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
                right_ankle_XY_toScreen = np.multiply(right_ankle_XY, [screen_width, screen_height])
                # 2D pose pose estimation result
                J_nose_screen = np.append(J_nose_screen, np.array(
                    [[nose_XY_toScreen[0], nose_XY_toScreen[1], 0]]), axis=0)
                J_shoulder_L_screen = np.append(J_shoulder_L_screen, np.array(
                    [[left_shoulder_XY_toScreen[0], left_shoulder_XY_toScreen[1], 0]]), axis=0)
                J_elbow_L_screen = np.append(J_elbow_L_screen, np.array(
                    [[left_elbow_XY_toScreen[0], left_elbow_XY_toScreen[1], 0]]), axis=0)
                J_wrist_L_screen = np.append(J_wrist_L_screen, np.array(
                    [[left_wrist_XY_toScreen[0], left_wrist_XY_toScreen[1], 0]]), axis=0)
                J_shoulder_R_screen = np.append(J_shoulder_R_screen, np.array(
                    [[right_shoulder_XY_toScreen[0], right_shoulder_XY_toScreen[1], 0]]), axis=0)
                J_elbow_R_screen = np.append(J_elbow_R_screen, np.array(
                    [[right_elbow_XY_toScreen[0], right_elbow_XY_toScreen[1], 0]]), axis=0)
                J_wrist_R_screen = np.append(J_wrist_R_screen, np.array(
                    [[right_wrist_XY_toScreen[0], right_wrist_XY_toScreen[1], 0]]), axis=0)
                J_hip_L_screen = np.append(J_hip_L_screen, np.array(
                    [[left_hip_XY_toScreen[0], left_hip_XY_toScreen[1], 0]]), axis=0)
                J_knee_L_screen = np.append(J_knee_L_screen, np.array(
                    [[left_knee_XY_toScreen[0], left_knee_XY_toScreen[1], 0]]), axis=0)
                J_ankle_L_screen = np.append(J_ankle_L_screen, np.array(
                    [[left_ankle_XY_toScreen[0], left_ankle_XY_toScreen[1], 0]]), axis=0)
                J_hip_R_screen = np.append(J_hip_R_screen, np.array(
                    [[right_hip_XY_toScreen[0], right_hip_XY_toScreen[1], 0]]), axis=0)
                J_knee_R_screen = np.append(J_knee_R_screen, np.array(
                    [[right_knee_XY_toScreen[0], right_knee_XY_toScreen[1], 0]]), axis=0)
                J_ankle_R_screen = np.append(J_ankle_R_screen, np.array(
                    [[right_ankle_XY_toScreen[0], right_ankle_XY_toScreen[1], 0]]), axis=0)
                
                # frame
                frame_index = np.append(frame_index, fcnt)
                frame_index_screen = np.append(frame_index_screen, fcnt)

                # mark on video for checking
                cv2.putText(image, str(int(fcnt)), 
                        (1800, 1000), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,0,0), 4, cv2.LINE_AA)
                cv2.putText(image, "nose", 
                    tuple(nose_XY_toScreen.astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2, cv2.LINE_AA)
                cv2.putText(image, "shoulder_L", 
                    tuple(left_shoulder_XY_toScreen.astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2, cv2.LINE_AA)
                cv2.putText(image, "elbow_L", 
                    tuple(left_elbow_XY_toScreen.astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2, cv2.LINE_AA)
                cv2.putText(image, "wrist_L", 
                    tuple(left_wrist_XY_toScreen.astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2, cv2.LINE_AA)
                cv2.putText(image, "shoulder_R", 
                    tuple(right_shoulder_XY_toScreen.astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2, cv2.LINE_AA)
                cv2.putText(image, "elbow_R", 
                    tuple(right_elbow_XY_toScreen.astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2, cv2.LINE_AA)
                cv2.putText(image, "wrist_R", 
                    tuple(right_wrist_XY_toScreen.astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2, cv2.LINE_AA)
                cv2.putText(image, "hip_L", 
                    tuple(left_hip_XY_toScreen.astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2, cv2.LINE_AA)
                cv2.putText(image, "knee_L", 
                    tuple(left_knee_XY_toScreen.astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2, cv2.LINE_AA)
                cv2.putText(image, "ankle_L", 
                    tuple(left_ankle_XY_toScreen.astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2, cv2.LINE_AA)
                cv2.putText(image, "hip_R", 
                    tuple(right_hip_XY_toScreen.astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2, cv2.LINE_AA)
                cv2.putText(image, "knee_R", 
                    tuple(right_knee_XY_toScreen.astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2, cv2.LINE_AA)
                cv2.putText(image, "ankle_R", 
                    tuple(right_ankle_XY_toScreen.astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2, cv2.LINE_AA)
            except:
                print("No detection")
                pass

            # Render detections
            mp_drawing.draw_landmarks(
                image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 255),
                                       thickness=2, circle_radius=2),  # joints (BGR)
                mp_drawing.DrawingSpec(color=(245, 66, 280),
                                       thickness=2, circle_radius=2)  # bones (BGR)
            )
            cv2.imshow('Mediapipe Processed', image)
            out.write(image)
            # press Q to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            print("No video frame")
            break

# close source and detection videos
cap.release()
out.release()
cv2.destroyAllWindows()

jointDict = {}
jointDict_screen = {}

def convertArrToDict(array, key):
    joint_coord_all = []
    for i in range(totalframe):
        try:
            joint_coord = dict({'x':array[i][0], 'y':array[i][1], 'z': array[i][2]})
            joint_coord_all.append(joint_coord)
        except IndexError as error:
            print(error)
        except Exception as exception:
            print(exception)
    jointDict[key] = joint_coord_all

def convertFrame(array, key):
    frame_lst = []
    for i in range(totalframe):
        try:
            frame_curr = array[i]
            frame_lst.append(frame_curr)
        except IndexError as error:
            print(error)
        except Exception as exception:
            print(exception)
    jointDict[key] = frame_lst

def convertArrToDict_screen(array, key):
    joint_coord_all = []
    for i in range(totalframe):
        try:
            joint_coord = dict({'x':array[i][0], 'y':array[i][1], 'z': array[i][2]})
            joint_coord_all.append(joint_coord)
        except IndexError as error:
            print(error)
        except Exception as exception:
            print(exception)
    jointDict_screen[key] = joint_coord_all

def convertFrame_screen(array, key):
    frame_lst = []
    for i in range(totalframe):
        try:
            frame_curr = array[i]
            frame_lst.append(frame_curr)
        except IndexError as error:
            print(error)
        except Exception as exception:
            print(exception)
    jointDict_screen[key] = frame_lst

# make 3D dictionary
convertFrame(frame_index, "frame_index")
convertArrToDict(J_nose, "J_nose")
convertArrToDict(J_eyeInner_L, "J_eyeInner_L")
convertArrToDict(J_eye_L, "J_eye_L")
convertArrToDict(J_eyeOuter_L, "J_eyeOuter_L")
convertArrToDict(J_eyeInner_R, "J_eyeInner_R")
convertArrToDict(J_eye_R, "J_eye_R")
convertArrToDict(J_eyeOuter_R, "J_eyeOuter_R")
convertArrToDict(J_ear_L, "J_ear_L")
convertArrToDict(J_ear_R, "J_ear_R")
convertArrToDict(J_mouth_L, "J_mouth_L")
convertArrToDict(J_mouth_R, "J_mouth_R")
convertArrToDict(J_shoulder_L, "J_shoulder_L")
convertArrToDict(J_elbow_L, "J_elbow_L")
convertArrToDict(J_wrist_L, "J_wrist_L")
convertArrToDict(J_pinky_L, "J_pinky_L")
convertArrToDict(J_index_L, "J_index_L")
convertArrToDict(J_thumb_L, "J_thumb_L")
convertArrToDict(J_shoulder_R, "J_shoulder_R")
convertArrToDict(J_elbow_R, "J_elbow_R")
convertArrToDict(J_wrist_R, "J_wrist_R")
convertArrToDict(J_pinky_R, "J_pinky_R")
convertArrToDict(J_index_R, "J_index_R")
convertArrToDict(J_thumb_R, "J_thumb_R")
convertArrToDict(J_hip_L, "J_hip_L")
convertArrToDict(J_knee_L, "J_knee_L")
convertArrToDict(J_ankle_L, "J_ankle_L")
convertArrToDict(J_heel_L, "J_heel_L")
convertArrToDict(J_footIndex_L, "J_footIndex_L")
convertArrToDict(J_hip_R, "J_hip_R")
convertArrToDict(J_knee_R, "J_knee_R")
convertArrToDict(J_ankle_R, "J_ankle_R")
convertArrToDict(J_heel_R, "J_heel_R")
convertArrToDict(J_footIndex_R, "J_footIndex_R")

# make 2D dictionary
convertFrame_screen(frame_index_screen, "frame_index")
convertArrToDict_screen(J_nose_screen, "J_nose")
convertArrToDict_screen(J_shoulder_L_screen, "J_shoulder_L")
convertArrToDict_screen(J_elbow_L_screen, "J_elbow_L")
convertArrToDict_screen(J_wrist_L_screen, "J_wrist_L")
convertArrToDict_screen(J_shoulder_R_screen, "J_shoulder_R")
convertArrToDict_screen(J_elbow_R_screen, "J_elbow_R")
convertArrToDict_screen(J_wrist_R_screen, "J_wrist_R")
convertArrToDict_screen(J_hip_L_screen, "J_hip_L")
convertArrToDict_screen(J_knee_L_screen, "J_knee_L")
convertArrToDict_screen(J_ankle_L_screen, "J_ankle_L")
convertArrToDict_screen(J_hip_R_screen, "J_hip_R")
convertArrToDict_screen(J_knee_R_screen, "J_knee_R")
convertArrToDict_screen(J_ankle_R_screen, "J_ankle_R")

# make 3D json
with open(json_path, 'w') as fp:
    json.dump(jointDict, fp, sort_keys=True, indent=4)

# make 2D json
with open(json_path_screen, 'w') as fp_screen:
    json.dump(jointDict_screen, fp_screen, sort_keys=True, indent=4)