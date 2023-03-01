import cv2
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_objectron = mp.solutions.objectron
cap = cv2.VideoCapture("images/silla2.mp4")
with mp_objectron.Objectron(
        static_image_mode=False,
        max_num_objects=5,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.99,
        # bikes, books, bottles, cameras, cereal boxes, chairs, cups, laptops
        model_name="Chair") as objectron:
    while True:
        ret, frame = cap.read()
        if ret == False:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = objectron.process(frame_rgb)

        if results.detected_objects is not None:
            for detected_object in results.detected_objects:
                mp_drawing.draw_landmarks(
                    frame, detected_object.landmarks_2d, mp_objectron.BOX_CONNECTIONS)
                mp_drawing.draw_axis(
                    frame, detected_object.rotation, detected_object.translation)
        cv2.imshow("IA-ReciclApp", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break
cap.release()
cv2.destroyAllWindows()
