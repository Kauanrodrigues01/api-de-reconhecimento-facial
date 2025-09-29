import uuid

from django.contrib.auth.models import AbstractUser
from django.db import models


class User(AbstractUser):
    """
    Custom User model with face recognition capabilities
    """

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    email = models.EmailField(unique=True)
    face_embeddings = models.JSONField(default=list, blank=True)
    face_model_used = models.CharField(max_length=50, blank=True, null=True)
    is_face_verified = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    USERNAME_FIELD = "email"
    REQUIRED_FIELDS = ["username"]

    class Meta:
        db_table = "users"

    def __str__(self):
        return self.email


class FaceLoginAttempt(models.Model):
    """
    Track face login attempts for security monitoring
    """

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(
        User,
        on_delete=models.CASCADE,
        related_name="face_login_attempts",
        null=True,
        blank=True,
    )
    ip_address = models.GenericIPAddressField()
    user_agent = models.TextField(blank=True)
    timestamp = models.DateTimeField(auto_now_add=True)
    success = models.BooleanField()
    confidence_score = models.FloatField(null=True, blank=True)
    fraud_score = models.IntegerField(null=True, blank=True)
    fraud_indicators = models.JSONField(default=list, blank=True)
    error_message = models.TextField(blank=True)

    class Meta:
        db_table = "face_login_attempts"
        ordering = ["-timestamp"]

    def __str__(self):
        status = "SUCCESS" if self.success else "FAILED"
        user_info = self.user.email if self.user else "Unknown"
        return f"{status} - {user_info} - {self.timestamp}"


class FaceRegistrationAttempt(models.Model):
    """
    Track face registration attempts
    """

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    email = models.EmailField()
    ip_address = models.GenericIPAddressField()
    user_agent = models.TextField(blank=True)
    timestamp = models.DateTimeField(auto_now_add=True)
    success = models.BooleanField()
    error_message = models.TextField(blank=True)
    fraud_score = models.IntegerField(null=True, blank=True)
    fraud_indicators = models.JSONField(default=list, blank=True)

    class Meta:
        db_table = "face_registration_attempts"
        ordering = ["-timestamp"]

    def __str__(self):
        status = "SUCCESS" if self.success else "FAILED"
        return f"{status} - {self.email} - {self.timestamp}"
