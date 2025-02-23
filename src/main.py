import mediapipe as mp
import cv2
from detection import DetectionHandler
import time


def main() -> None:
    detection_handler = DetectionHandler('./face_landmarker.task')

    cap = cv2.VideoCapture(1)
    start_time = time.time()

    ret = True
    while ret:
        ret, frame = cap.read()

        if not ret:
            print("Failed to grab frame")
            break

        rgb_frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        timestamp = int((time.time() - start_time) * 1000)
        detection_handler.update_handler_image(rgb_frame, timestamp)

        annotated_image = detection_handler.draw_facial_landmarks(rgb_frame.numpy_view())
        annotated_image = cv2.flip(annotated_image, 1)

        cv2.imshow('FaceNav', annotated_image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            ret = False

        detection_handler.perform_computer_action()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()