import streamlit as st
import cv2
import numpy as np
import boto3

# Function to detect faces in an image
def detect_faces(frame):
    # Convert to gray scale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Load the Haar Cascade Classifier for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    return faces

# WebSocket connection to trigger Lambda (pseudo code, adapt as needed)
def trigger_lambda():
    # Set up your API Gateway WebSocket connection here
    # Use boto3 to invoke your Lambda function
    pass

# Main Streamlit application
st.title("Real-Time Face Detection")

# Start the webcam
camera = cv2.VideoCapture(0)

# if st.checkbox('Show Raw Video'):
#     FRAME_WINDOW = st.image([])

while True:
    ret, frame = camera.read()
    
    if not ret:
        st.error("Could not read from webcam.")
        break

    # faces = detect_faces(frame)

    # Draw rectangles around detected faces
    # for (x, y, w, h) in faces:
    #     cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
    
    # Display the video stream with detected faces
    # FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels='RGB')

    # Trigger the Lambda function if a face is detected
    # if len(faces) > 0:
    #     trigger_lambda()
