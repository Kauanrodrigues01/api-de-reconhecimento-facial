import base64
from io import BytesIO
from typing import Optional

import numpy as np
from django.contrib.auth import get_user_model
from django.contrib.auth.backends import BaseBackend
from PIL import Image

from services.face_recognition_service import FaceRecognitionService

User = get_user_model()


class FaceRecognitionBackend(BaseBackend):
    """
    Custom authentication backend for face recognition
    """

    def __init__(self):
        self.face_service = FaceRecognitionService(
            model_name="facenet512", detector_backend="mtcnn"
        )

    def authenticate(self, request, face_image=None, **kwargs):
        """
        Authenticate user using face recognition

        Args:
            request: Django request object
            face_image: Base64 encoded image or numpy array

        Returns:
            User object if authentication successful, None otherwise
        """
        if not face_image:
            return None

        try:
            # Convert base64 to numpy array if needed
            if isinstance(face_image, str):
                face_image = self._base64_to_numpy(face_image)

            # Check for fraud first
            fraud_result = self.face_service.detect_fraud_in_image(face_image)
            if not fraud_result["success"]:
                return None

            if fraud_result["is_fraudulent"]:
                return None

            # Generate embedding for the input image
            embedding_result = self.face_service.generate_embedding(face_image)
            if not embedding_result["success"]:
                return None

            input_embedding = embedding_result["embedding"]

            # Compare with all users who have face embeddings
            users_with_faces = User.objects.filter(
                is_face_verified=True, face_embeddings__isnull=False
            ).exclude(face_embeddings=[])

            best_match = None
            best_confidence = 0
            min_distance = float("inf")

            for user in users_with_faces:
                for stored_embedding in user.face_embeddings:
                    comparison = self.face_service.compare_embeddings(
                        input_embedding, stored_embedding
                    )

                    if comparison["success"] and comparison["is_same_person"]:
                        if comparison["distance"] < min_distance:
                            min_distance = comparison["distance"]
                            best_match = user
                            best_confidence = comparison["confidence"]

            # Return the best match if confidence is high enough
            if best_match and best_confidence >= 80:  # 80% confidence threshold
                return best_match

            return None

        except Exception as e:
            return None

    def get_user(self, user_id):
        """
        Get user by ID
        """
        try:
            return User.objects.get(pk=user_id)
        except User.DoesNotExist:
            return None

    def _base64_to_numpy(self, base64_string):
        """
        Convert base64 string to numpy array
        """
        try:
            # Remove data URL prefix if present
            if "," in base64_string:
                base64_string = base64_string.split(",")[1]

            # Decode base64
            image_data = base64.b64decode(base64_string)

            # Convert to PIL Image
            image = Image.open(BytesIO(image_data))

            # Convert to RGB if necessary
            if image.mode != "RGB":
                image = image.convert("RGB")

            # Convert to numpy array
            return np.array(image)

        except Exception as e:
            raise ValueError(f"Failed to decode base64 image: {str(e)}")
