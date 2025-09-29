import base64
from io import BytesIO

import numpy as np
from django.contrib.auth import get_user_model
from django.contrib.auth.password_validation import validate_password
from PIL import Image
from rest_framework import serializers

from services.face_recognition_service import FaceRecognitionService

from .models import FaceLoginAttempt, FaceRegistrationAttempt

User = get_user_model()


class UserRegistrationSerializer(serializers.ModelSerializer):
    """
    Serializer for user registration with face recognition
    """

    password = serializers.CharField(write_only=True, validators=[validate_password])
    password_confirm = serializers.CharField(write_only=True)
    face_image = serializers.ImageField(
        write_only=True, help_text="Face image file upload"
    )

    class Meta:
        model = User
        fields = [
            "email",
            "username",
            "first_name",
            "last_name",
            "password",
            "password_confirm",
            "face_image",
        ]

    def validate(self, attrs):
        """
        Validate password confirmation and face image
        """
        if attrs["password"] != attrs["password_confirm"]:
            raise serializers.ValidationError("Passwords don't match")

        # Validate face image
        face_image = attrs["face_image"]
        try:
            # Convert uploaded image to numpy array
            face_array = self._image_to_numpy(face_image)

            # Initialize face recognition service
            face_service = FaceRecognitionService()

            # Check for fraud
            fraud_result = face_service.detect_fraud_in_image(face_array)
            if not fraud_result["success"]:
                raise serializers.ValidationError(
                    f"Face image analysis failed: {fraud_result.get('error', 'Unknown error')}"
                )

            if fraud_result["is_fraudulent"]:
                raise serializers.ValidationError(
                    f"Fraudulent image detected. Risk level: {fraud_result['risk_level']}"
                )

            # Generate face embedding
            embedding_result = face_service.generate_embedding(face_array)
            if not embedding_result["success"]:
                raise serializers.ValidationError(
                    f"Failed to generate face embedding: {embedding_result.get('error', 'Unknown error')}"
                )

            # Store embedding and model info for later use
            attrs["_face_embedding"] = embedding_result["embedding"]
            attrs["_face_model"] = embedding_result["model_used"]

        except Exception as e:
            raise serializers.ValidationError(f"Invalid face image: {str(e)}")

        return attrs

    def create(self, validated_data):
        """
        Create user with face embedding
        """
        # Remove password confirmation and face data from validated_data
        validated_data.pop("password_confirm")
        face_embedding = validated_data.pop("_face_embedding")
        face_model = validated_data.pop("_face_model")
        validated_data.pop("face_image")

        # Create user
        user = User.objects.create_user(**validated_data)

        # Add face embedding
        user.face_embeddings = [face_embedding]
        user.face_model_used = face_model
        user.is_face_verified = True
        user.save()

        return user

    def _image_to_numpy(self, image_file):
        """
        Convert uploaded image file to numpy array
        """
        try:
            # Open image using PIL
            image = Image.open(image_file)

            # Convert to RGB if necessary
            if image.mode != "RGB":
                image = image.convert("RGB")

            # Convert to numpy array
            return np.array(image)

        except Exception as e:
            raise ValueError(f"Failed to process image file: {str(e)}")

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


class FaceLoginSerializer(serializers.Serializer):
    """
    Serializer for face recognition login
    """

    email = serializers.EmailField(help_text="User email address")
    face_image = serializers.ImageField(help_text="Face image file upload")

    def validate_face_image(self, value):
        """
        Validate face image format
        """
        try:
            # Convert uploaded image to numpy array to validate format
            self._image_to_numpy(value)
            return value
        except Exception as e:
            raise serializers.ValidationError(f"Invalid face image format: {str(e)}")

    def validate_email(self, value):
        """
        Validate that user exists with this email
        """
        try:
            user = User.objects.get(email=value)
            if not user.is_face_verified:
                raise serializers.ValidationError("User has no face verification setup")
            return value
        except User.DoesNotExist:
            raise serializers.ValidationError("User with this email does not exist")

    def authenticate_user(self, request):
        """
        Authenticate user using email + face recognition
        """
        email = self.validated_data["email"]
        face_image = self.validated_data["face_image"]

        try:
            # Get user by email
            user = User.objects.get(email=email)

            # Convert face image to numpy array
            face_array = self._image_to_numpy(face_image)

            # Initialize face recognition service
            face_service = FaceRecognitionService()

            # Generate embedding for provided face
            embedding_result = face_service.generate_embedding(face_array)
            if not embedding_result["success"]:
                return None

            new_embedding = embedding_result["embedding"]

            # Compare with stored embeddings
            if not user.face_embeddings:
                return None

            # Check if any stored embedding matches
            for stored_embedding in user.face_embeddings:
                comparison = face_service.compare_embeddings(
                    new_embedding,
                    stored_embedding,
                    threshold=0.4  # You can adjust this threshold
                )

                if comparison["success"] and comparison["is_same_person"]:
                    # Store comparison confidence for logging
                    request._face_confidence = comparison["confidence"]
                    request._face_distance = comparison["distance"]
                    return user

            return None

        except User.DoesNotExist:
            return None
        except Exception as e:
            return None

    def _image_to_numpy(self, image_file):
        """
        Convert uploaded image file to numpy array
        """
        try:
            # Open image using PIL
            image = Image.open(image_file)

            # Convert to RGB if necessary
            if image.mode != "RGB":
                image = image.convert("RGB")

            # Convert to numpy array
            return np.array(image)

        except Exception as e:
            raise ValueError(f"Failed to process image file: {str(e)}")

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


class UserSerializer(serializers.ModelSerializer):
    """
    Serializer for user information
    """

    class Meta:
        model = User
        fields = [
            "id",
            "email",
            "username",
            "first_name",
            "last_name",
            "is_face_verified",
            "created_at",
        ]
        read_only_fields = ["id", "is_face_verified", "created_at"]


class FaceLoginAttemptSerializer(serializers.ModelSerializer):
    """
    Serializer for face login attempts
    """

    class Meta:
        model = FaceLoginAttempt
        fields = "__all__"
        read_only_fields = ["id", "timestamp"]


class FaceRegistrationAttemptSerializer(serializers.ModelSerializer):
    """
    Serializer for face registration attempts
    """

    class Meta:
        model = FaceRegistrationAttempt
        fields = "__all__"
        read_only_fields = ["id", "timestamp"]
