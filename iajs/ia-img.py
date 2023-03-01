
import cv2
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_objectron = mp.solutions.objectron
with mp_objectron.Objectron(
        static_image_mode=True,
        max_num_objects=5,
        min_detection_confidence=0.5,
        model_name="bottle,Chair") as objectron:
    image = cv2.imread("images/bottle.jpg")
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = objectron.process(image_rgb)
    print("results.detected_objects: ", results.detected_objects)
    if results.detected_objects is not None:
        for detected_object in results.detected_objects:
            mp_drawing.draw_landmarks(image, detected_object.landmarks_2d, mp_objectron.BOX_CONNECTIONS,
                                      mp_drawing.DrawingSpec(
                                          color=(255, 255, 255), thickness=2, circle_radius=6),
                                      mp_drawing.DrawingSpec(color=(128, 0, 255), thickness=2))
            mp_drawing.draw_axis(
                image, detected_object.rotation, detected_object.translation)
    cv2.imshow("IA-ReciclApp", image)
    cv2.waitKey(0)
