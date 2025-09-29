import base64
from io import BytesIO
from typing import Dict, List, Optional, Union

import cv2
import numpy as np
from deepface import DeepFace
from PIL import Image


class FaceRecognitionService:
    """
    Pure face recognition service - no user management
    Only performs: detection, comparison, embedding generation, and fraud detection
    """

    # Best models for accuracy
    MODELS = {
        "facenet512": "Facenet512",  # Best overall - RECOMMENDED
        "arcface": "ArcFace",  # Best for large databases
        "vggface": "VGG-Face",  # Good balance
        "facenet": "Facenet",  # Lighter version
    }

    # Detection backends (retinaface excluded as requested)
    DETECTORS = {
        "mtcnn": "mtcnn",  # Good accuracy - RECOMMENDED
        "opencv": "opencv",  # Fast but less accurate
        "ssd": "ssd",  # Good balance
        "dlib": "dlib",  # Good quality
    }

    # Distance thresholds for each model
    DEFAULT_THRESHOLDS = {
        "Facenet512": 0.30,
        "ArcFace": 0.68,
        "VGG-Face": 0.40,
        "Facenet": 0.40,
    }

    def __init__(self, model_name: str = "facenet512", detector_backend: str = "mtcnn"):
        """
        Initialize the face recognition service

        Args:
            model_name: Model to use for face recognition
            detector_backend: Backend for face detection
        """
        self.model_name = self.MODELS.get(model_name, "Facenet512")
        self.detector_backend = self.DETECTORS.get(detector_backend, "mtcnn")
        self.threshold = self.DEFAULT_THRESHOLDS.get(self.model_name, 0.40)

    def detect_faces(
        self, image: Union[str, np.ndarray], min_confidence: float = 0.9
    ) -> Dict:
        """
        Detect all faces in an image

        Args:
            image: Path to image or numpy array
            min_confidence: Minimum confidence threshold (0-1)

        Returns:
            {
                'success': bool,
                'faces_detected': int,
                'faces': [
                    {
                        'facial_area': {'x', 'y', 'w', 'h'},
                        'confidence': float,
                        'face_image': numpy array (cropped face)
                    }
                ]
            }
        """
        try:
            face_objs = DeepFace.extract_faces(
                img_path=image,
                detector_backend=self.detector_backend,
                enforce_detection=False,
                align=True,
            )

            # Filter by confidence
            valid_faces = []
            for face in face_objs:
                if face["confidence"] >= min_confidence:
                    valid_faces.append(
                        {
                            "facial_area": face["facial_area"],
                            "confidence": round(face["confidence"], 4),
                            "face_image": face["face"],
                        }
                    )

            return {
                "success": True,
                "faces_detected": len(valid_faces),
                "faces": valid_faces,
            }

        except Exception as e:
            return {"success": False, "faces_detected": 0, "error": str(e)}

    def generate_embedding(self, image: Union[str, np.ndarray]) -> Dict:
        """
        Generate face embedding (vector representation)

        Args:
            image: Path to image or numpy array

        Returns:
            {
                'success': bool,
                'embedding': list of floats (face vector),
                'embedding_size': int,
                'model_used': str
            }
        """
        try:
            embeddings = DeepFace.represent(
                img_path=image,
                model_name=self.model_name,
                detector_backend=self.detector_backend,
                enforce_detection=True,
                align=True,
            )

            if not embeddings or len(embeddings) == 0:
                return {"success": False, "error": "No face detected in image"}

            if len(embeddings) > 1:
                return {
                    "success": False,
                    "error": f"Multiple faces detected ({len(embeddings)}). Use single face image.",
                }

            embedding_vector = embeddings[0]["embedding"]

            return {
                "success": True,
                "embedding": embedding_vector,
                "embedding_size": len(embedding_vector),
                "model_used": self.model_name,
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    def compare_faces(
        self,
        image1: Union[str, np.ndarray],
        image2: Union[str, np.ndarray],
        threshold: Optional[float] = None,
    ) -> Dict:
        """
        Compare two faces and return if they match

        Args:
            image1: First face image (path or numpy array)
            image2: Second face image (path or numpy array)
            threshold: Custom similarity threshold (lower = stricter)

        Returns:
            {
                'success': bool,
                'is_same_person': bool,
                'confidence': float (0-100),
                'confidence_level': str,
                'distance': float,
                'threshold_used': float,
                'model_used': str
            }
        """
        try:
            if threshold is None:
                threshold = self.threshold

            result = DeepFace.verify(
                img1_path=image1,
                img2_path=image2,
                model_name=self.model_name,
                detector_backend=self.detector_backend,
                distance_metric="cosine",
                enforce_detection=True,
                align=True,
                threshold=threshold,
            )

            # Calculate confidence percentage
            distance = result["distance"]
            confidence = max(0, min(100, (1 - distance) * 100))

            # Determine confidence level
            if distance < threshold * 0.5:
                level = "VERY_HIGH"
            elif distance < threshold * 0.75:
                level = "HIGH"
            elif distance < threshold:
                level = "MEDIUM"
            else:
                level = "LOW"

            return {
                "success": True,
                "is_same_person": result["verified"],
                "confidence": round(confidence, 2),
                "confidence_level": level,
                "distance": round(distance, 4),
                "threshold_used": threshold,
                "model_used": self.model_name,
            }

        except Exception as e:
            return {"success": False, "is_same_person": False, "error": str(e)}

    def compare_embeddings(
        self,
        embedding1: List[float],
        embedding2: List[float],
        threshold: Optional[float] = None,
    ) -> Dict:
        """
        Compare two face embeddings directly

        Args:
            embedding1: First face embedding vector
            embedding2: Second face embedding vector
            threshold: Custom similarity threshold

        Returns:
            {
                'success': bool,
                'is_same_person': bool,
                'confidence': float,
                'distance': float
            }
        """
        try:
            if threshold is None:
                threshold = self.threshold

            # Calculate cosine distance
            embedding1 = np.array(embedding1)
            embedding2 = np.array(embedding2)

            # Normalize vectors
            embedding1_norm = embedding1 / np.linalg.norm(embedding1)
            embedding2_norm = embedding2 / np.linalg.norm(embedding2)

            # Cosine similarity
            cosine_similarity = np.dot(embedding1_norm, embedding2_norm)
            distance = 1 - cosine_similarity

            is_same = distance < threshold
            confidence = max(0, min(100, (1 - distance) * 100))

            return {
                "success": True,
                "is_same_person": is_same,
                "confidence": round(confidence, 2),
                "distance": round(float(distance), 4),
                "threshold_used": threshold,
            }

        except Exception as e:
            return {"success": False, "is_same_person": False, "error": str(e)}

    def detect_fraud_in_image(self, image: Union[str, np.ndarray]) -> Dict:
        """
        Detect fraud attempts in static image
        Analyzes: face count, image quality, blur, brightness, face size, etc.

        Args:
            image: Path to image or numpy array

        Returns:
            {
                'success': bool,
                'is_fraudulent': bool,
                'fraud_score': int (0-100),
                'risk_level': str,
                'fraud_indicators': list of strings,
                'details': dict with metrics
            }
        """
        try:
            fraud_indicators = []
            fraud_score = 0

            # Load image
            if isinstance(image, str):
                img = cv2.imread(image)
            else:
                img = image.copy()

            if img is None:
                return {"success": False, "error": "Failed to load image"}

            # Check 1: Face detection
            detection = self.detect_faces(image, min_confidence=0.5)

            if not detection["success"]:
                fraud_indicators.append("face_detection_failed")
                fraud_score += 30
            elif detection["faces_detected"] == 0:
                fraud_indicators.append("no_face_detected")
                fraud_score += 40
            elif detection["faces_detected"] > 1:
                fraud_indicators.append(
                    f"multiple_faces_detected_{detection['faces_detected']}"
                )
                fraud_score += 20

            # Check 2: Image sharpness (blur detection)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

            if laplacian_var < 50:
                fraud_indicators.append(f"blurry_image_sharpness_{laplacian_var:.1f}")
                fraud_score += 15

            # Check 3: Brightness analysis
            brightness = np.mean(gray)

            if brightness < 30:
                fraud_indicators.append(f"too_dark_brightness_{brightness:.1f}")
                fraud_score += 10
            elif brightness > 220:
                fraud_indicators.append(f"too_bright_brightness_{brightness:.1f}")
                fraud_score += 10

            # Check 4: Contrast
            contrast = gray.std()
            if contrast < 20:
                fraud_indicators.append(f"low_contrast_{contrast:.1f}")
                fraud_score += 5

            # Check 5: Face size ratio
            if detection["faces_detected"] == 1:
                face = detection["faces"][0]
                face_area = face["facial_area"]
                face_width = face_area["w"]
                face_height = face_area["h"]
                img_height, img_width = img.shape[:2]

                face_ratio = (face_width * face_height) / (img_width * img_height)

                if face_ratio < 0.02:
                    fraud_indicators.append(f"face_too_small_{face_ratio * 100:.1f}%")
                    fraud_score += 15
                elif face_ratio > 0.9:
                    fraud_indicators.append(f"face_too_close_{face_ratio * 100:.1f}%")
                    fraud_score += 5

                # Check detection confidence
                if face["confidence"] < 0.8:
                    fraud_indicators.append(
                        f"low_detection_confidence_{face['confidence']:.2f}"
                    )
                    fraud_score += 10

            # Check 6: Color saturation (printed photos have low saturation)
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            saturation = hsv[:, :, 1].mean()

            if saturation < 15:
                fraud_indicators.append(f"low_saturation_{saturation:.1f}")
                fraud_score += 10

            # Check 7: Edge density (printed photos have uniform edges)
            edges = cv2.Canny(gray, 100, 200)
            edge_density = np.sum(edges > 0) / edges.size

            if edge_density < 0.02:
                fraud_indicators.append(f"low_edge_density_{edge_density:.3f}")
                fraud_score += 10

            # Check 8: Color histogram analysis
            hist_b = cv2.calcHist([img], [0], None, [256], [0, 256])
            hist_g = cv2.calcHist([img], [1], None, [256], [0, 256])
            hist_r = cv2.calcHist([img], [2], None, [256], [0, 256])

            # Check for unnatural color distribution
            hist_variance = np.var([hist_b.std(), hist_g.std(), hist_r.std()])
            if hist_variance > 50000:
                fraud_indicators.append(f"abnormal_color_distribution")
                fraud_score += 5

            # Determine risk level
            if fraud_score >= 80:
                risk_level = "CRITICAL"
            elif fraud_score >= 60:
                risk_level = "HIGH"
            elif fraud_score >= 40:
                risk_level = "MEDIUM"
            elif fraud_score >= 20:
                risk_level = "LOW"
            else:
                risk_level = "MINIMAL"

            is_fraudulent = fraud_score >= 70

            return {
                "success": True,
                "is_fraudulent": is_fraudulent,
                "fraud_score": min(100, fraud_score),
                "risk_level": risk_level,
                "fraud_indicators": fraud_indicators,
                "details": {
                    "faces_detected": detection["faces_detected"],
                    "image_sharpness": round(laplacian_var, 2),
                    "brightness": round(brightness, 2),
                    "contrast": round(contrast, 2),
                    "saturation": round(saturation, 2),
                    "edge_density": round(edge_density, 4),
                    "image_size": f"{img_width}x{img_height}",
                },
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    def detect_fraud_in_video(
        self,
        video_path: str,
        sample_frames: int = 10,
        min_movement_threshold: float = 5.0,
    ) -> Dict:
        """
        Detect fraud attempts in video
        Analyzes: liveness, movement, face consistency, temporal coherence

        Args:
            video_path: Path to video file
            sample_frames: Number of frames to analyze
            min_movement_threshold: Minimum required movement between frames

        Returns:
            {
                'success': bool,
                'is_fraudulent': bool,
                'fraud_score': int (0-100),
                'risk_level': str,
                'fraud_indicators': list,
                'details': dict with video metrics
            }
        """
        try:
            cap = cv2.VideoCapture(video_path)

            if not cap.isOpened():
                return {"success": False, "error": "Failed to open video file"}

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            duration = total_frames / fps if fps > 0 else 0

            fraud_indicators = []
            fraud_score = 0

            # Sample frames evenly throughout video
            frame_indices = np.linspace(
                0, total_frames - 1, min(sample_frames, total_frames), dtype=int
            )

            sampled_frames = []
            face_detections = []
            embeddings = []

            for frame_idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()

                if not ret:
                    continue

                sampled_frames.append(frame)

                # Detect faces
                detection = self.detect_faces(frame, min_confidence=0.8)
                face_detections.append(detection)

                # Generate embedding if single face detected
                if detection["success"] and detection["faces_detected"] == 1:
                    try:
                        emb_result = self.generate_embedding(frame)
                        if emb_result["success"]:
                            embeddings.append(emb_result["embedding"])
                    except:
                        pass

            cap.release()

            # Analysis 1: Face detection consistency
            valid_detections = [
                d for d in face_detections if d["success"] and d["faces_detected"] == 1
            ]
            detection_rate = (
                len(valid_detections) / len(face_detections) if face_detections else 0
            )

            if detection_rate < 0.7:
                fraud_indicators.append(
                    f"inconsistent_face_detection_{detection_rate * 100:.0f}%"
                )
                fraud_score += 30

            # Analysis 2: Multiple faces in frames
            multi_face_frames = sum(
                1 for d in face_detections if d.get("faces_detected", 0) > 1
            )
            if multi_face_frames > 0:
                fraud_indicators.append(f"multiple_faces_in_{multi_face_frames}_frames")
                fraud_score += 25

            # Analysis 3: Movement detection (liveness)
            if len(sampled_frames) >= 2:
                movement_scores = []

                for i in range(len(sampled_frames) - 1):
                    gray1 = cv2.cvtColor(sampled_frames[i], cv2.COLOR_BGR2GRAY)
                    gray2 = cv2.cvtColor(sampled_frames[i + 1], cv2.COLOR_BGR2GRAY)

                    diff = cv2.absdiff(gray1, gray2)
                    movement = np.mean(diff)
                    movement_scores.append(movement)

                avg_movement = np.mean(movement_scores)
                max_movement = np.max(movement_scores)

                if avg_movement < min_movement_threshold:
                    fraud_indicators.append(f"insufficient_movement_{avg_movement:.1f}")
                    fraud_score += 35

                # Check if movement is too uniform (might be looping video)
                movement_std = np.std(movement_scores)
                if movement_std < 2 and len(movement_scores) > 3:
                    fraud_indicators.append(f"uniform_movement_pattern")
                    fraud_score += 20

            # Analysis 4: Face embedding consistency
            if len(embeddings) >= 2:
                distances = []

                for i in range(len(embeddings) - 1):
                    emb1 = np.array(embeddings[i])
                    emb2 = np.array(embeddings[i + 1])

                    # Cosine distance
                    emb1_norm = emb1 / np.linalg.norm(emb1)
                    emb2_norm = emb2 / np.linalg.norm(emb2)
                    distance = 1 - np.dot(emb1_norm, emb2_norm)

                    distances.append(distance)

                avg_distance = np.mean(distances)
                max_distance = np.max(distances)

                # If embeddings vary too much, might be different people or manipulated
                if max_distance > 0.5:
                    fraud_indicators.append(
                        f"inconsistent_face_identity_{max_distance:.3f}"
                    )
                    fraud_score += 40
                elif avg_distance > 0.3:
                    fraud_indicators.append(f"high_face_variation_{avg_distance:.3f}")
                    fraud_score += 20

            # Analysis 5: Video duration check
            if duration < 2:
                fraud_indicators.append(f"video_too_short_{duration:.1f}s")
                fraud_score += 15

            # Analysis 6: First frame quality check
            if sampled_frames:
                first_frame_fraud = self.detect_fraud_in_image(sampled_frames[0])
                if (
                    first_frame_fraud["success"]
                    and first_frame_fraud["fraud_score"] > 30
                ):
                    fraud_indicators.append("poor_initial_frame_quality")
                    fraud_score += min(25, first_frame_fraud["fraud_score"] // 3)

            # Analysis 7: FPS check (too low might be manipulated)
            if fps < 15:
                fraud_indicators.append(f"low_fps_{fps:.1f}")
                fraud_score += 10

            # Determine risk level
            if fraud_score >= 70:
                risk_level = "CRITICAL"
            elif fraud_score >= 50:
                risk_level = "HIGH"
            elif fraud_score >= 30:
                risk_level = "MEDIUM"
            elif fraud_score >= 15:
                risk_level = "LOW"
            else:
                risk_level = "MINIMAL"

            is_fraudulent = fraud_score >= 50

            return {
                "success": True,
                "is_fraudulent": is_fraudulent,
                "fraud_score": min(100, fraud_score),
                "risk_level": risk_level,
                "fraud_indicators": fraud_indicators,
                "details": {
                    "total_frames": total_frames,
                    "analyzed_frames": len(sampled_frames),
                    "fps": round(fps, 2),
                    "duration_seconds": round(duration, 2),
                    "detection_rate": round(detection_rate * 100, 2),
                    "valid_face_frames": len(valid_detections),
                    "avg_movement": round(avg_movement, 2)
                    if "avg_movement" in locals()
                    else 0,
                    "embedding_consistency": round((1 - avg_distance) * 100, 2)
                    if "avg_distance" in locals()
                    else None,
                },
            }

        except Exception as e:
            return {"success": False, "error": str(e)}


# Usage Examples
if __name__ == "__main__":
    # Initialize service
    service = FaceRecognitionService(model_name="facenet512", detector_backend="mtcnn")

    print("=" * 60)
    print("FACE RECOGNITION SERVICE - EXAMPLES")
    print("=" * 60)

    # Example 1: Detect faces in image
    print("\n1. DETECTING FACES")
    detection = service.detect_faces("photo.jpg", min_confidence=0.9)
    print(f"Faces detected: {detection['faces_detected']}")
    print(f"Success: {detection['success']}")

    # Example 2: Generate face embedding
    print("\n2. GENERATING FACE EMBEDDING")
    embedding = service.generate_embedding("photo.jpg")
    if embedding["success"]:
        print(f"Embedding size: {embedding['embedding_size']}")
        print(f"Model used: {embedding['model_used']}")
        print(f"First 5 values: {embedding['embedding'][:5]}")

    # Example 3: Compare two faces
    print("\n3. COMPARING TWO FACES")
    comparison = service.compare_faces("photo1.jpg", "photo2.jpg")
    if comparison["success"]:
        print(f"Same person: {comparison['is_same_person']}")
        print(f"Confidence: {comparison['confidence']}%")
        print(f"Confidence level: {comparison['confidence_level']}")
        print(f"Distance: {comparison['distance']}")

    # Example 4: Compare embeddings directly
    print("\n4. COMPARING EMBEDDINGS")
    if embedding["success"]:
        emb1 = embedding["embedding"]
        # Simulate another embedding (in real case, get from another image)
        comparison_emb = service.compare_embeddings(emb1, emb1)  # Comparing with itself
        print(f"Same person: {comparison_emb['is_same_person']}")
        print(f"Confidence: {comparison_emb['confidence']}%")

    # Example 5: Detect fraud in image
    print("\n5. FRAUD DETECTION IN IMAGE")
    fraud_img = service.detect_fraud_in_image("suspicious_photo.jpg")
    if fraud_img["success"]:
        print(f"Is fraudulent: {fraud_img['is_fraudulent']}")
        print(f"Fraud score: {fraud_img['fraud_score']}/100")
        print(f"Risk level: {fraud_img['risk_level']}")
        print(f"Indicators: {fraud_img['fraud_indicators']}")
        print(f"Details: {fraud_img['details']}")

    # Example 6: Detect fraud in video
    print("\n6. FRAUD DETECTION IN VIDEO")
    fraud_vid = service.detect_fraud_in_video(
        "verification_video.mp4", sample_frames=15, min_movement_threshold=5.0
    )
    if fraud_vid["success"]:
        print(f"Is fraudulent: {fraud_vid['is_fraudulent']}")
        print(f"Fraud score: {fraud_vid['fraud_score']}/100")
        print(f"Risk level: {fraud_vid['risk_level']}")
        print(f"Indicators: {fraud_vid['fraud_indicators']}")
        print(f"Video details: {fraud_vid['details']}")

    print("\n" + "=" * 60)
