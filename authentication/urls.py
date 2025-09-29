from django.urls import path
from rest_framework_simplejwt.views import TokenRefreshView

from . import views

urlpatterns = [
    # Authentication endpoints
    path("register/", views.register_user, name="register_user"),
    path("login/", views.face_login, name="face_login"),
    path("logout/", views.logout, name="logout"),
    path("token/refresh/", TokenRefreshView.as_view(), name="token_refresh"),
    # User management
    path("profile/", views.user_profile, name="user_profile"),
    path("add-face/", views.add_face_embedding, name="add_face_embedding"),
    # History and monitoring
    path("login-history/", views.login_history, name="login_history"),
    # Utilities
    path("convert-image/", views.convert_image_to_base64, name="convert_image_to_base64"),
]
