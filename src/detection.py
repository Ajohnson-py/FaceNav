import time

import mediapipe as mp
import numpy as np
from numpy import ndarray
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mouse import MouseHandler


class DetectionHandler:
    def __init__(self, model_asset_path, running):

        def result_callback(result, output_timestamp_ms, unused_arg=None):
            """callback method for detector"""
            self.detection_result = result

        self.running = running.value

        self.mouse = MouseHandler(0.2)
        self.last_click_time = 0
        self.eye_blink_start_time = None
        self.eyebrow_raise_count = 0
        self.last_eyebrow_raise_time = 0

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

    def update_handler_running_status(self, new_status) -> None:
        self.running = new_status

    def get_running_status(self) -> bool:
        return self.running

    def perform_computer_action(self) -> None:
        """Finds blendshapes and performs appropriate OS actions to emulate mouse inputs"""
        try:
            blendshapes = self.detection_result.face_blendshapes[0]

            for category in blendshapes:
                # Face can only do one action at a time to prevent accidental input
                # Right and left mouse movement
                if category.category_name == "mouthLeft" and category.score > 0.25 and self.running:
                    self.mouse.expression_action = (-10, 0)
                elif category.category_name == "mouthRight" and category.score > 0.25 and self.running:
                    self.mouse.expression_action = (10, 0)

                # Up and down mouse movement
                if category.category_name == "mouthShrugUpper" and category.score > 0.57 and self.running:
                    self.mouse.expression_action = (0, -10)
                elif category.category_name == "mouthRollLower" and category.score > 0.2 and self.running:
                    self.mouse.expression_action = (0, 10)

                # Left and right click mouse input
                if category.category_name == "browInnerUp" and category.score > 0.18:
                    current_time = time.time()

                    # Resume program if paused
                    if not self.running:
                        print(current_time - self.last_eyebrow_raise_time)
                        if self.eyebrow_raise_count >= 2 and (current_time - self.last_eyebrow_raise_time) <= 2.5:
                            print(current_time - self.last_eyebrow_raise_time)
                            self.running = True
                            self.last_eyebrow_raise_time = 0
                            self.eyebrow_raise_count = 0
                        if current_time - self.last_eyebrow_raise_time > 0.75:
                            print("Brow count +1")
                            self.eyebrow_raise_count += 1
                            if self.eyebrow_raise_count == 1:
                                self.last_eyebrow_raise_time = current_time
                        if current_time - self.last_eyebrow_raise_time > 2.5:
                            print("Reset")
                            self.eyebrow_raise_count = 0
                            self.last_eyebrow_raise_time = 0

                    # Only click if enough time has passed since the last click
                    elif current_time - self.last_click_time > 0.5:
                        self.mouse.expression_action = "clickLeft"
                        self.last_click_time = current_time

                # TODO: Make right click more polished to use
                if category.category_name == "eyeBlinkLeft" and self.running:
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
