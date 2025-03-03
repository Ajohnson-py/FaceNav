import mediapipe as mp
import numpy as np
import cv2
import time
from numpy import ndarray
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mouse import MouseHandler


class DetectionHandler:
    def __init__(self, model_asset_path, not_paused):
        def result_callback(result, output_timestamp_ms, unused_arg=None):
            """callback method for detector"""
            self.detection_result = result

        self.not_paused = not_paused.value

        self.brow_status = None

        self.mouse = MouseHandler(0.1)
        self.last_click_time = 0
        self.eye_blink_start_time = None
        self.eyebrow_raise_count = 0

        self.base_options = python.BaseOptions(model_asset_path=model_asset_path)
        self.detector = vision.FaceLandmarker.create_from_options(
            vision.FaceLandmarkerOptions(
                base_options=self.base_options,
                output_face_blendshapes=True,
                output_facial_transformation_matrixes=True,
                num_faces=1,
                running_mode=vision.RunningMode.LIVE_STREAM,
                result_callback=result_callback
            )
        )
        self.detection_result = None

    def update_handler_image(self, rgb_image, timestamp) -> None:
        """Updates the image the handler has access to"""
        self.detector.detect_async(rgb_image, timestamp)

    def update_handler_pause_status(self, new_status) -> None:
        self.not_paused = new_status

    def get_running_status(self) -> bool:
        return self.not_paused

    def perform_computer_action(self) -> None:
        """Finds blendshapes and performs appropriate OS actions to emulate mouse inputs"""
        try:
            blendshapes = self.detection_result.face_blendshapes[0]

            for category in blendshapes:
                # Right and left mouse movement
                if category.category_name == "mouthLeft" and category.score > 0.25 and self.not_paused:
                    self.mouse.expression_action = (-10, 0)
                elif category.category_name == "mouthRight" and category.score > 0.25 and self.not_paused:
                    self.mouse.expression_action = (10, 0)

                # Up and down mouse movement
                if category.category_name == "mouthShrugUpper" and category.score > 0.57 and self.not_paused:
                    self.mouse.expression_action = (0, -10)
                elif category.category_name == "mouthRollLower" and category.score > 0.2 and self.not_paused:
                    self.mouse.expression_action = (0, 10)

                # Left click mouse input
                if category.category_name == "browInnerUp":
                    # Check if eyebrow is raised
                    if category.score > 0.2 and self.brow_status is None:
                        self.brow_status = "up"
                        if self.not_paused:
                            self.mouse.expression_action = ("clickLeft", False)

                    # Check if eyebrow is lowered
                    if self.brow_status == "up" and category.score < 0.02:
                        if self.not_paused:
                            self.mouse.expression_action = self.mouse.expression_action = ("clickLeft", True)

                        self.last_click_time = time.time()
                        self.brow_status = None

                        # Increment eyebrow raise count if program is paused
                        if not self.not_paused:
                            self.eyebrow_raise_count += 1
                            self.last_click_time = time.time()

                    # Reset eyebrow raise count
                    if time.time() - self.last_click_time > 1.5:
                        self.eyebrow_raise_count = 0
                    # Unpause program
                    elif self.eyebrow_raise_count >= 2:
                        self.not_paused = True
                        self.eyebrow_raise_count = 0

                # Right click mouse input
                if category.category_name == "eyeBlinkLeft" and self.not_paused:
                    current_time = time.time()

                    if category.score > 0.6:
                        if self.eye_blink_start_time is None:
                            self.eye_blink_start_time = current_time

                        # If enough time has passed, trigger the right-click
                        if current_time - self.eye_blink_start_time >= 0.8:
                            self.mouse.expression_action = "clickRight"
                            self.last_click_time = current_time
                            self.eye_blink_start_time = None

                    # Reset only when eye fully opens again
                    elif category.score < 0.5:
                        self.eye_blink_start_time = None
        except (IndexError, AttributeError):
            pass

    def draw_facial_landmarks(self, rgb_image) -> ndarray:
        """Draws facial mesh on OpenCV image"""
        # Ensure face landmarks were detected
        if not self.detection_result or not self.detection_result.face_landmarks:
            return rgb_image

        face_landmarks = self.detection_result.face_landmarks[0]
        annotated_image = np.copy(rgb_image)

        # Draw the face landmark mesh, including face border, face tesselation, and eyes
        face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        face_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in face_landmarks
        ])

        solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles
            .get_default_face_mesh_tesselation_style())
        solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles
            .get_default_face_mesh_contours_style())
        solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_IRISES,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles
            .get_default_face_mesh_iris_connections_style())

        return annotated_image


def facial_detection_loop(not_paused) -> None:
    # Variables to control frame update time
    frame_target_time = 1 / 60
    previous_frame_time = time.time()

    detection_handler = DetectionHandler('./face_landmarker.task', not_paused)

    cap = cv2.VideoCapture(1)
    start_time = time.time()

    ret = True
    while ret:
        current_time = time.time()
        elapsed_time = current_time - previous_frame_time
        time_to_wait = frame_target_time - elapsed_time

        if time_to_wait > 0:
            time.sleep(time_to_wait)

        ret, frame = cap.read()

        if not ret:
            print("Failed to grab frame")
            break

        rgb_frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        timestamp = int((time.time() - start_time) * 1000)
        detection_handler.update_handler_image(rgb_frame, timestamp)

        detection_handler.update_handler_pause_status(not_paused.value)
        detection_handler.perform_computer_action()

        not_paused.value = detection_handler.get_running_status()

        annotated_image = detection_handler.draw_facial_landmarks(rgb_frame.numpy_view())
        annotated_image = cv2.flip(annotated_image, 1)

        cv2.imshow('FaceNav', annotated_image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            ret = False

    cap.release()
    cv2.destroyAllWindows()
