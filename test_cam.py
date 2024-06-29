import streamlit as st
import cv2
import numpy as np
from queue import Queue
from threading import Thread, Lock
import time
import face_tool
from scrfd import SCRFD

def main():

    st.title("Real-Time Camera Feed")

    # Initialize the camera
    camera1, camera2 = cv2.VideoCapture(0), cv2.VideoCapture(0)

    # Create a placeholder to display the frames
    frame_placeholder = st.empty()
    # Queue and threading
   
    frame_index, is_face_recognized, frame_count, start_time = 0, False, 0, time.time()


    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_thickness = 2
    text_color = (0, 255, 0)

    while camera1.isOpened() and camera2.isOpened():
        # camera1ture frame-by-frame
        ret, frame = camera1.read()
        ret2, frame2 = camera2.read()
        if not ret or not ret2:
            st.error("Failed to capture image.")
            break

        # Convert the frame from BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        frame_count += 1
        fps_value = frame_count / (time.time() - start_time)

        cv2.putText(frame, f'FPS: {fps_value:.2f}', (10, 30), font, font_scale, text_color, font_thickness)
        # Display the frame
        frame_placeholder.image(frame, channels="RGB")


        # Control the frame rate
        cv2.waitKey(1)
        cv2.destroyAllWindows()
    # Release the camera
    camera1.release()

if __name__ == "__main__":
    main()
