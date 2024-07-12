import streamlit as st
import cv2
import numpy as np
from queue import Queue
from threading import Thread, Lock
import time
import face_tool
from scrfd import SCRFD
from fps_metric import FPS

def main():

    # Initialize face detector
    face_detector = SCRFD(model_file='/Users/dominhtuan/Downloads/AnNinhSoiChieu/scrfd_10g_bnkps.onnx')

    def recognize_faces(image, landmarks, known_face_encodings):
        if not known_face_encodings:
            return []
        face_encodings = face_tool.face_encoding(image=image, kpss=landmarks)
        face_matches = []
        for face_encoding in face_encodings:
            matches = face_tool.compare_faces(known_face_encodings, face_encoding, tolerance=0.8)
            face_distances = face_tool.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                face_matches.append(True)
            else:
                face_matches.append(False)
        return face_matches

    def update_face_list(face_list, image, landmarks):
        face_encodings = face_tool.face_encoding(image=image, kpss=landmarks)
        if not face_list:
            face_list.append((image, face_encodings))
        else:
            is_known_face = any(recognize_faces(image, landmarks, known_face_encodings)[0] for _, known_face_encodings in face_list)
            if not is_known_face:
                face_list.append((image, face_encodings))
        return face_list

    def face_detection_worker(input_queue, output_queue, lock):
        while True:
            face_list, frame = input_queue.get()
            face_boxes, landmarks = face_detector.detect(frame, threshold_w=0.1, threshold_h=0.1, thresh=0.2)
            if face_boxes is not None and landmarks is not None:
                with lock:
                    face_list = update_face_list(face_list, frame, landmarks)
                output_queue.put({"face_list": face_list, "landmarks": landmarks, "face_boxes": face_boxes})
            else:
                output_queue.put({"face_list": face_list, "landmarks": None, "face_boxes": None})

    def face_recognition_worker(input_queue, output_queue, lock, face_list):
        while True:
            frame = input_queue.get()
            face_boxes, landmarks_from_id_card = face_detector.detect(frame, threshold_w=0.05, threshold_h=0.05, thresh=0.2)
            if len(face_list) > 0 and face_boxes is not None:
                for known_image, known_face_encodings in face_list:
                    if recognize_faces(frame, landmarks_from_id_card, known_face_encodings)[0]:
                        with lock:
                            output_queue.put([known_image,frame])
            if face_boxes is not None:
                with lock:
                    output_queue.put([0,frame])

            with lock:
                output_queue.put(None)

    # Setup Streamlit
    st.title("Real-time Face Recognition")
    notification_container = st.empty()
    progress_bar = st.progress(0)
    camera1, camera2 = cv2.VideoCapture(1), cv2.VideoCapture(1)
    col1, col2 = st.columns([4, 2])
    frame_display1, frame_display2 = col1.empty(), col2.empty()
    next_button = st.button("Next")

    # Queue and threading
    lock = Lock()
    face_list = []
    input_queue1, output_queue1 = Queue(maxsize=3), Queue()
    input_queue2, output_queue2 = Queue(maxsize=3), Queue()
    Thread(target=face_detection_worker, args=(input_queue1, output_queue1, lock), daemon=True).start()
    Thread(target=face_recognition_worker, args=(input_queue2, output_queue2, lock, face_list), daemon=True).start()

    fps = FPS().start()
    frame_index, is_face_recognized, frame_count, start_time = 0, False, 0, time.time()

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_thickness = 2
    text_color = (0, 255, 0)
    duration = 1
    steps = 100
    sleep_duration = duration / steps

    # while camera1.isOpened() and camera2.isOpened():
    while True:
        frame_index += 1
        if next_button:
            is_face_recognized = False
        if not is_face_recognized:
            ret1, frame1 = camera1.read()
            ret2, frame2 = camera2.read()

            if not ret1 or not ret2:
                break

            frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
            frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
            if frame_index % 10 == 0:
                if not input_queue1.full():
                    input_queue1.put((face_list, frame1))
                if not output_queue1.empty():
                    result = output_queue1.get()
                    with lock:
                        face_list = result["face_list"]
                        landmarks = result["landmarks"]
                        face_boxes = result["face_boxes"]
            # if frame_index % 5 == 0 and len((face_list)) > 0:
                if not input_queue2.full():
                    input_queue2.put(frame2)
                if not output_queue2.empty():
                    out_put_2 = output_queue2.get()
                    if out_put_2 is not None:
                        notification_container.warning("Đang xác thực hành khách ")

                        output_queue2.queue.clear()
                        known_image, frame2 = out_put_2[0],out_put_2[1]
                        if not isinstance(known_image, int):
                            is_face_recognized = True
                            next_button = False
                            frame1 = known_image
                            notification_container.success("Xác thực thành công ")
                            # for i in range(3,0,-1):
                            #     time.sleep(0.7)
                            #     notification_container.success("Xác thực thành công "  + str(i))
                            #     next_button = True
                        else:
                            notification_container.error("Xác thực khong thành công!")
                    else:
                        notification_container.empty()
        fps_value = frame_count / (time.time() - start_time)
        cv2.putText(frame1, f'FPS: {fps_value:.2f}', (10, 30), font, font_scale, text_color, font_thickness)
        frame_count += 1
        fps.update()
        frame_display1.image(frame1, channels="RGB", use_column_width=True)
        frame_display2.image(frame2, channels="RGB", use_column_width=True)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    camera1.release()
    camera2.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
