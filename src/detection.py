import mediapipe as mp
import numpy as np
from numpy import ndarray
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mouse import MouseHandler


class DetectionHandler:
    def __init__(self, model_asset_path):
        def result_callback(result, output_timestamp_ms, unused_arg=None):
            """callback method for detector"""
            self.detection_result = result  # Store the latest detection result

        self.mouse = MouseHandler(3)
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

        self.detector.detect_async(rgb_image, timestamp)

    def perform_computer_action(self) -> None:
        """
        Finds blendshapes and performs appropriate OS actions to emulate mouse inputs
        """
        try:
            blendshapes = self.detection_result.face_blendshapes[0]

            for category in blendshapes:
                # Face can only do one action at a time to prevent accidental input
                # Right and left mouse movement
                if category.category_name == "mouthLeft" and category.score > 0.25:
                    self.mouse.move_cursor(-1, 0)
                elif category.category_name == "mouthRight" and category.score > 0.25:
                    self.mouse.move_cursor(1, 0)

                # Up and down mouse movement
                elif category.category_name == "mouthShrugUpper" and category.score > 0.5:
                    self.mouse.move_cursor(0, -1)
                elif category.category_name == "mouthRollLower" and category.score > 0.045:
                    self.mouse.move_cursor(0, 1)

                # Left click mouse input
                # TODO: Make clicking happen only once and add right click
                elif category.category_name == "browInnerUp" and category.score > 0.35:
                    self.mouse.click(Button.left)

        except IndexError:
            pass

    def draw_facial_landmarks(self, rgb_image) -> ndarray:
        # Ensure face landmarks were detected
        if not self.detection_result or not self.detection_result.face_landmarks:
            return rgb_image

        face_landmarks= self.detection_result.face_landmarks[0]
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
