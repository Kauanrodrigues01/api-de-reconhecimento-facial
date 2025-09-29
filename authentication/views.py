from rest_framework import status
from rest_framework.decorators import api_view, permission_classes, parser_classes
from rest_framework.permissions import AllowAny, IsAuthenticated
from rest_framework.response import Response
from rest_framework.authtoken.models import Token
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework_simplejwt.tokens import RefreshToken
from django.contrib.auth import get_user_model
from django.db import transaction
from django.utils import timezone
from django.core.files.uploadedfile import UploadedFile
from drf_yasg.utils import swagger_auto_schema
from drf_yasg import openapi
import base64
import mimetypes

from .serializers import (
    UserRegistrationSerializer,
    FaceLoginSerializer,
    UserSerializer,
    FaceLoginAttemptSerializer,
    FaceRegistrationAttemptSerializer
)
from .models import FaceLoginAttempt, FaceRegistrationAttempt
from services.face_recognition_service import FaceRecognitionService

User = get_user_model()


def get_client_ip(request):
    """Get client IP address from request"""
    x_forwarded_for = request.META.get('HTTP_X_FORWARDED_FOR')
    if x_forwarded_for:
        ip = x_forwarded_for.split(',')[0]
    else:
        ip = request.META.get('REMOTE_ADDR')
    return ip


@swagger_auto_schema(
    method='post',
    operation_description="Registra um novo usuário com reconhecimento facial. Aceita upload de arquivo de imagem diretamente.",
    operation_summary="Registro de usuário",
    manual_parameters=[
        openapi.Parameter('email', openapi.IN_FORM, description="Email do usuário", type=openapi.TYPE_STRING, required=True),
        openapi.Parameter('username', openapi.IN_FORM, description="Nome de usuário", type=openapi.TYPE_STRING, required=True),
        openapi.Parameter('first_name', openapi.IN_FORM, description="Primeiro nome", type=openapi.TYPE_STRING, required=True),
        openapi.Parameter('last_name', openapi.IN_FORM, description="Último nome", type=openapi.TYPE_STRING, required=True),
        openapi.Parameter('password', openapi.IN_FORM, description="Senha", type=openapi.TYPE_STRING, required=True),
        openapi.Parameter('password_confirm', openapi.IN_FORM, description="Confirmação da senha", type=openapi.TYPE_STRING, required=True),
        openapi.Parameter('face_image', openapi.IN_FORM, description="Arquivo de imagem facial", type=openapi.TYPE_FILE, required=True),
    ],
    responses={
        201: openapi.Response(
            description="Usuário registrado com sucesso",
            examples={
                "application/json": {
                    "success": True,
                    "message": "User registered successfully with face recognition",
                    "user": {
                        "id": 1,
                        "username": "usuario123",
                        "email": "usuario@example.com",
                        "first_name": "João",
                        "last_name": "Silva"
                    },
                    "access_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
                    "refresh_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9..."
                }
            }
        ),
        400: openapi.Response(
            description="Dados inválidos",
            examples={
                "application/json": {
                    "success": False,
                    "errors": {
                        "email": ["Este campo é obrigatório."]
                    }
                }
            }
        ),
        500: openapi.Response(
            description="Erro interno do servidor",
            examples={
                "application/json": {
                    "success": False,
                    "message": "Registration failed",
                    "error": "Database connection failed"
                }
            }
        )
    },
    tags=['Authentication']
)
@api_view(['POST'])
@permission_classes([AllowAny])
@parser_classes([MultiPartParser, FormParser])
def register_user(request):
    """
    Register a new user with face recognition

    Expected form-data:
    - email: user@example.com
    - username: username
    - first_name: First
    - last_name: Last
    - password: password123
    - password_confirm: password123
    - face_image: [IMAGE FILE UPLOAD]
    """
    serializer = UserRegistrationSerializer(data=request.data)

    # Track registration attempt
    attempt_data = {
        'email': request.data.get('email', ''),
        'ip_address': get_client_ip(request),
        'user_agent': request.META.get('HTTP_USER_AGENT', ''),
        'success': False
    }

    try:
        if serializer.is_valid():
            with transaction.atomic():
                user = serializer.save()

                # Create JWT tokens
                refresh = RefreshToken.for_user(user)
                access_token = refresh.access_token

                # Mark attempt as successful
                attempt_data['success'] = True
                FaceRegistrationAttempt.objects.create(**attempt_data)

                return Response({
                    'success': True,
                    'message': 'User registered successfully with face recognition',
                    'user': UserSerializer(user).data,
                    'access_token': str(access_token),
                    'refresh_token': str(refresh)
                }, status=status.HTTP_201_CREATED)
        else:
            # Record failed attempt with errors
            attempt_data['error_message'] = str(serializer.errors)
            FaceRegistrationAttempt.objects.create(**attempt_data)

            return Response({
                'success': False,
                'errors': serializer.errors
            }, status=status.HTTP_400_BAD_REQUEST)

    except Exception as e:
        # Record failed attempt with exception
        attempt_data['error_message'] = str(e)
        FaceRegistrationAttempt.objects.create(**attempt_data)

        return Response({
            'success': False,
            'message': 'Registration failed',
            'error': str(e)
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@swagger_auto_schema(
    method='post',
    operation_description="Realiza login do usuário usando email + reconhecimento facial. O sistema gera um novo embedding da face fornecida e compara com os embeddings salvos no banco de dados do usuário. Aceita upload de arquivo de imagem diretamente.",
    operation_summary="Login com email e reconhecimento facial",
    manual_parameters=[
        openapi.Parameter('email', openapi.IN_FORM, description="Email do usuário", type=openapi.TYPE_STRING, required=True),
        openapi.Parameter('face_image', openapi.IN_FORM, description="Arquivo de imagem facial", type=openapi.TYPE_FILE, required=True),
    ],
    responses={
        200: openapi.Response(
            description="Login realizado com sucesso",
            examples={
                "application/json": {
                    "success": True,
                    "message": "Login successful",
                    "user": {
                        "id": 1,
                        "username": "usuario123",
                        "email": "usuario@example.com",
                        "first_name": "João",
                        "last_name": "Silva"
                    },
                    "access_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
                    "refresh_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9..."
                }
            }
        ),
        401: openapi.Response(
            description="Falha na autenticação facial ou usuário não encontrado",
            examples={
                "application/json": {
                    "success": False,
                    "message": "Face recognition failed. Please try again."
                }
            }
        ),
        400: openapi.Response(
            description="Dados inválidos ou usuário sem verificação facial",
            examples={
                "application/json": {
                    "success": False,
                    "errors": {
                        "email": ["User with this email does not exist"],
                        "face_image": ["Invalid face image format"]
                    }
                }
            }
        )
    },
    tags=['Authentication']
)
@api_view(['POST'])
@permission_classes([AllowAny])
@parser_classes([MultiPartParser, FormParser])
def face_login(request):
    """
    Login user using email + face recognition

    The system:
    1. Validates the user exists with the provided email
    2. Generates a new face embedding from the provided image
    3. Compares the new embedding with stored embeddings in the database
    4. Returns JWT tokens if match is found

    Expected form-data:
    - email: user@example.com
    - face_image: [IMAGE FILE UPLOAD]
    """
    serializer = FaceLoginSerializer(data=request.data)

    # Initialize attempt tracking
    attempt_data = {
        'ip_address': get_client_ip(request),
        'user_agent': request.META.get('HTTP_USER_AGENT', ''),
        'success': False,
        'user': None
    }

    try:
        if serializer.is_valid():
            # Authenticate user using face recognition
            user = serializer.authenticate_user(request)

            if user:
                # Successful login - Generate JWT tokens
                refresh = RefreshToken.for_user(user)
                access_token = refresh.access_token

                # Get confidence and fraud analysis from the authentication process
                face_confidence = getattr(request, '_face_confidence', 0)
                face_distance = getattr(request, '_face_distance', 0)

                # Get fraud score for logging
                face_service = FaceRecognitionService(model_name="facenet512", detector_backend="mtcnn")
                face_array = serializer._image_to_numpy(request.FILES['face_image'])
                fraud_result = face_service.detect_fraud_in_image(face_array)

                # Update attempt data
                attempt_data.update({
                    'user': user,
                    'success': True,
                    'fraud_score': fraud_result.get('fraud_score', 0),
                    'fraud_indicators': fraud_result.get('fraud_indicators', [])
                })

                FaceLoginAttempt.objects.create(**attempt_data)

                return Response({
                    'success': True,
                    'message': 'Login successful',
                    'user': UserSerializer(user).data,
                    'access_token': str(access_token),
                    'refresh_token': str(refresh)
                }, status=status.HTTP_200_OK)
            else:
                # Failed login
                attempt_data['error_message'] = 'Face recognition failed'
                FaceLoginAttempt.objects.create(**attempt_data)

                return Response({
                    'success': False,
                    'message': 'Face recognition failed. Please try again.'
                }, status=status.HTTP_401_UNAUTHORIZED)
        else:
            # Invalid data
            attempt_data['error_message'] = str(serializer.errors)
            FaceLoginAttempt.objects.create(**attempt_data)

            return Response({
                'success': False,
                'errors': serializer.errors
            }, status=status.HTTP_400_BAD_REQUEST)

    except Exception as e:
        # Record failed attempt with exception
        attempt_data['error_message'] = str(e)
        FaceLoginAttempt.objects.create(**attempt_data)

        return Response({
            'success': False,
            'message': 'Login failed',
            'error': str(e)
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@swagger_auto_schema(
    method='get',
    operation_description="Obtém o perfil do usuário autenticado",
    operation_summary="Perfil do usuário",
    responses={
        200: openapi.Response(
            description="Perfil obtido com sucesso",
            examples={
                "application/json": {
                    "success": True,
                    "user": {
                        "id": 1,
                        "username": "usuario123",
                        "email": "usuario@example.com",
                        "first_name": "João",
                        "last_name": "Silva",
                        "is_face_verified": True
                    }
                }
            }
        ),
        401: openapi.Response(
            description="Não autenticado",
            examples={
                "application/json": {
                    "detail": "Token de autenticação não fornecido."
                }
            }
        )
    },
    tags=['User Management'],
    manual_parameters=[
        openapi.Parameter(
            'Authorization',
            openapi.IN_HEADER,
            description="JWT token no formato: Bearer <access_token>",
            type=openapi.TYPE_STRING,
            required=True
        )
    ]
)
@api_view(['GET'])
@permission_classes([IsAuthenticated])
def user_profile(request):
    """
    Get current user profile
    """
    try:
        serializer = UserSerializer(request.user)
        return Response({
            'success': True,
            'user': serializer.data
        }, status=status.HTTP_200_OK)

    except Exception as e:
        return Response({
            'success': False,
            'message': 'Failed to get user profile',
            'error': str(e)
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@swagger_auto_schema(
    method='post',
    operation_description="Adiciona um embedding facial adicional para o usuário autenticado. Aceita upload de arquivo de imagem diretamente.",
    operation_summary="Adicionar embedding facial",
    manual_parameters=[
        openapi.Parameter('face_image', openapi.IN_FORM, description="Arquivo de imagem facial", type=openapi.TYPE_FILE, required=True),
        openapi.Parameter(
            'Authorization',
            openapi.IN_HEADER,
            description="JWT token no formato: Bearer <access_token>",
            type=openapi.TYPE_STRING,
            required=True
        )
    ],
    responses={
        200: openapi.Response(
            description="Embedding adicionado com sucesso",
            examples={
                "application/json": {
                    "success": True,
                    "message": "Face embedding added successfully",
                    "total_embeddings": 2
                }
            }
        ),
        400: openapi.Response(
            description="Dados inválidos ou imagem fraudulenta",
            examples={
                "application/json": {
                    "success": False,
                    "message": "Fraudulent image detected. Risk level: HIGH"
                }
            }
        ),
        401: openapi.Response(
            description="Não autenticado"
        )
    },
    tags=['User Management']
)
@api_view(['POST'])
@permission_classes([IsAuthenticated])
@parser_classes([MultiPartParser, FormParser])
def add_face_embedding(request):
    """
    Add additional face embedding for user

    Expected form-data:
    - face_image: [IMAGE FILE UPLOAD]
    """
    try:
        face_image = request.FILES.get('face_image')
        if not face_image:
            return Response({
                'success': False,
                'message': 'Face image is required'
            }, status=status.HTTP_400_BAD_REQUEST)

        # Initialize face recognition service with improved settings
        face_service = FaceRecognitionService(model_name="facenet512", detector_backend="mtcnn")

        # Convert uploaded image to numpy array
        serializer = FaceLoginSerializer()
        face_array = serializer._image_to_numpy(face_image)

        # Check for fraud
        fraud_result = face_service.detect_fraud_in_image(face_array)
        if not fraud_result["success"]:
            return Response({
                'success': False,
                'message': f"Face image analysis failed: {fraud_result.get('error', 'Unknown error')}"
            }, status=status.HTTP_400_BAD_REQUEST)

        if fraud_result["is_fraudulent"]:
            return Response({
                'success': False,
                'message': f"Fraudulent image detected. Risk level: {fraud_result['risk_level']}"
            }, status=status.HTTP_400_BAD_REQUEST)

        # Generate face embedding
        embedding_result = face_service.generate_embedding(face_array)
        if not embedding_result["success"]:
            return Response({
                'success': False,
                'message': f"Failed to generate face embedding: {embedding_result.get('error', 'Unknown error')}"
            }, status=status.HTTP_400_BAD_REQUEST)

        # Add embedding to user
        user = request.user
        if not user.face_embeddings:
            user.face_embeddings = []

        user.face_embeddings.append(embedding_result["embedding"])
        user.face_model_used = embedding_result["model_used"]
        user.is_face_verified = True
        user.save()

        return Response({
            'success': True,
            'message': 'Face embedding added successfully',
            'total_embeddings': len(user.face_embeddings)
        }, status=status.HTTP_200_OK)

    except Exception as e:
        return Response({
            'success': False,
            'message': 'Failed to add face embedding',
            'error': str(e)
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@swagger_auto_schema(
    method='get',
    operation_description="Obtém o histórico de tentativas de login do usuário autenticado",
    operation_summary="Histórico de login",
    responses={
        200: openapi.Response(
            description="Histórico obtido com sucesso",
            examples={
                "application/json": {
                    "success": True,
                    "login_history": [
                        {
                            "id": 1,
                            "timestamp": "2024-01-01T10:00:00Z",
                            "success": True,
                            "ip_address": "192.168.1.100",
                            "user_agent": "Mozilla/5.0...",
                            "fraud_score": 0.1
                        }
                    ]
                }
            }
        ),
        401: openapi.Response(
            description="Não autenticado"
        )
    },
    tags=['User Management'],
    manual_parameters=[
        openapi.Parameter(
            'Authorization',
            openapi.IN_HEADER,
            description="JWT token no formato: Bearer <access_token>",
            type=openapi.TYPE_STRING,
            required=True
        )
    ]
)
@api_view(['GET'])
@permission_classes([IsAuthenticated])
def login_history(request):
    """
    Get user's login history
    """
    try:
        attempts = FaceLoginAttempt.objects.filter(user=request.user).order_by('-timestamp')[:20]
        serializer = FaceLoginAttemptSerializer(attempts, many=True)

        return Response({
            'success': True,
            'login_history': serializer.data
        }, status=status.HTTP_200_OK)

    except Exception as e:
        return Response({
            'success': False,
            'message': 'Failed to get login history',
            'error': str(e)
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@swagger_auto_schema(
    method='post',
    operation_description="Faz logout do usuário removendo o token de autenticação",
    operation_summary="Logout do usuário",
    responses={
        200: openapi.Response(
            description="Logout realizado com sucesso",
            examples={
                "application/json": {
                    "success": True,
                    "message": "Logout successful"
                }
            }
        ),
        401: openapi.Response(
            description="Não autenticado"
        ),
        500: openapi.Response(
            description="Erro interno do servidor",
            examples={
                "application/json": {
                    "success": False,
                    "message": "Logout failed",
                    "error": "Token deletion failed"
                }
            }
        )
    },
    tags=['Authentication'],
    manual_parameters=[
        openapi.Parameter(
            'Authorization',
            openapi.IN_HEADER,
            description="JWT token no formato: Bearer <access_token>",
            type=openapi.TYPE_STRING,
            required=True
        )
    ]
)
@api_view(['POST'])
@permission_classes([IsAuthenticated])
def logout(request):
    """
    Logout user by deleting token
    """
    try:
        request.user.auth_token.delete()
        return Response({
            'success': True,
            'message': 'Logout successful'
        }, status=status.HTTP_200_OK)

    except Exception as e:
        return Response({
            'success': False,
            'message': 'Logout failed',
            'error': str(e)
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@swagger_auto_schema(
    method='post',
    operation_description="Converte uma imagem enviada via upload para formato base64",
    operation_summary="Converter imagem para base64",
    manual_parameters=[
        openapi.Parameter(
            'image',
            openapi.IN_FORM,
            description="Arquivo de imagem para conversão",
            type=openapi.TYPE_FILE,
            required=True
        )
    ],
    responses={
        200: openapi.Response(
            description="Imagem convertida com sucesso",
            examples={
                "application/json": {
                    "success": True,
                    "message": "Image converted successfully",
                    "data": {
                        "base64": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQ...",
                        "mime_type": "image/jpeg",
                        "size": 2048,
                        "filename": "example.jpg"
                    }
                }
            }
        ),
        400: openapi.Response(
            description="Dados inválidos ou arquivo não fornecido",
            examples={
                "application/json": {
                    "success": False,
                    "message": "No image file provided"
                }
            }
        ),
        500: openapi.Response(
            description="Erro interno do servidor",
            examples={
                "application/json": {
                    "success": False,
                    "message": "Failed to convert image",
                    "error": "File processing error"
                }
            }
        )
    },
    tags=['Utilities']
)
@api_view(['POST'])
@permission_classes([AllowAny])
@parser_classes([MultiPartParser, FormParser])
def convert_image_to_base64(request):
    """
    Convert uploaded image to base64 format

    This endpoint accepts an image file upload and returns it as a base64 string
    along with metadata about the file.
    """
    try:
        # Check if image file was provided
        if 'image' not in request.FILES:
            return Response({
                'success': False,
                'message': 'No image file provided'
            }, status=status.HTTP_400_BAD_REQUEST)

        image_file = request.FILES['image']

        # Validate file type
        allowed_types = ['image/jpeg', 'image/jpg', 'image/png', 'image/gif', 'image/bmp', 'image/webp']
        mime_type = mimetypes.guess_type(image_file.name)[0]

        if mime_type not in allowed_types:
            return Response({
                'success': False,
                'message': 'Invalid file type. Allowed types: JPEG, PNG, GIF, BMP, WebP'
            }, status=status.HTTP_400_BAD_REQUEST)

        # Read file content
        image_content = image_file.read()

        # Convert to base64
        base64_string = base64.b64encode(image_content).decode('utf-8')

        # Create data URL format
        data_url = f"data:{mime_type};base64,{base64_string}"

        return Response({
            'success': True,
            'message': 'Image converted successfully',
            'data': {
                'base64': data_url,
                'mime_type': mime_type,
                'size': len(image_content),
                'filename': image_file.name
            }
        }, status=status.HTTP_200_OK)

    except Exception as e:
        return Response({
            'success': False,
            'message': 'Failed to convert image',
            'error': str(e)
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@swagger_auto_schema(
    method='get',
    operation_description="Lista todos os usuários cadastrados no sistema",
    operation_summary="Listar todos os usuários",
    responses={
        200: openapi.Response(
            description="Lista de usuários obtida com sucesso",
            examples={
                "application/json": {
                    "success": True,
                    "users": [
                        {
                            "id": 1,
                            "username": "usuario1",
                            "email": "usuario1@example.com",
                            "first_name": "João",
                            "last_name": "Silva",
                            "is_face_verified": True,
                            "created_at": "2024-01-01T10:00:00Z"
                        },
                        {
                            "id": 2,
                            "username": "usuario2",
                            "email": "usuario2@example.com",
                            "first_name": "Maria",
                            "last_name": "Santos",
                            "is_face_verified": True,
                            "created_at": "2024-01-02T10:00:00Z"
                        }
                    ],
                    "total_users": 2
                }
            }
        ),
        500: openapi.Response(
            description="Erro interno do servidor",
            examples={
                "application/json": {
                    "success": False,
                    "message": "Failed to list users",
                    "error": "Database connection failed"
                }
            }
        )
    },
    tags=['User Management']
)
@api_view(['GET'])
@permission_classes([AllowAny])
def list_all_users(request):
    """
    List all registered users in the system
    """
    try:
        users = User.objects.all().order_by('-date_joined')
        serializer = UserSerializer(users, many=True)

        return Response({
            'success': True,
            'users': serializer.data,
            'total_users': users.count()
        }, status=status.HTTP_200_OK)

    except Exception as e:
        return Response({
            'success': False,
            'message': 'Failed to list users',
            'error': str(e)
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@swagger_auto_schema(
    method='delete',
    operation_description="Remove todos os usuários cadastrados no sistema",
    operation_summary="Remover todos os usuários",
    responses={
        200: openapi.Response(
            description="Todos os usuários foram removidos com sucesso",
            examples={
                "application/json": {
                    "success": True,
                    "message": "All users deleted successfully",
                    "deleted_count": 5
                }
            }
        ),
        500: openapi.Response(
            description="Erro interno do servidor",
            examples={
                "application/json": {
                    "success": False,
                    "message": "Failed to delete all users",
                    "error": "Database error"
                }
            }
        )
    },
    tags=['User Management']
)
@api_view(['DELETE'])
@permission_classes([AllowAny])
def delete_all_users(request):
    """
    Delete all registered users from the system
    """
    try:
        with transaction.atomic():
            # Count users before deletion
            user_count = User.objects.count()

            # Delete all users
            User.objects.all().delete()

            return Response({
                'success': True,
                'message': 'All users deleted successfully',
                'deleted_count': user_count
            }, status=status.HTTP_200_OK)

    except Exception as e:
        return Response({
            'success': False,
            'message': 'Failed to delete all users',
            'error': str(e)
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@swagger_auto_schema(
    method='delete',
    operation_description="Remove um usuário específico pelo ID",
    operation_summary="Remover usuário específico",
    manual_parameters=[
        openapi.Parameter('user_id', openapi.IN_PATH, description="ID do usuário a ser removido", type=openapi.TYPE_INTEGER, required=True),
    ],
    responses={
        200: openapi.Response(
            description="Usuário removido com sucesso",
            examples={
                "application/json": {
                    "success": True,
                    "message": "User deleted successfully",
                    "deleted_user": {
                        "id": 1,
                        "username": "usuario123",
                        "email": "usuario@example.com"
                    }
                }
            }
        ),
        404: openapi.Response(
            description="Usuário não encontrado",
            examples={
                "application/json": {
                    "success": False,
                    "message": "User not found"
                }
            }
        ),
        500: openapi.Response(
            description="Erro interno do servidor",
            examples={
                "application/json": {
                    "success": False,
                    "message": "Failed to delete user",
                    "error": "Database error"
                }
            }
        )
    },
    tags=['User Management']
)
@api_view(['DELETE'])
@permission_classes([AllowAny])
def delete_user(request, user_id):
    """
    Delete a specific user by ID
    """
    try:
        # Get user by ID
        user = User.objects.get(id=user_id)

        # Store user data for response
        deleted_user_data = {
            'id': user.id,
            'username': user.username,
            'email': user.email
        }

        # Delete user
        with transaction.atomic():
            user.delete()

        return Response({
            'success': True,
            'message': 'User deleted successfully',
            'deleted_user': deleted_user_data
        }, status=status.HTTP_200_OK)

    except User.DoesNotExist:
        return Response({
            'success': False,
            'message': 'User not found'
        }, status=status.HTTP_404_NOT_FOUND)

    except Exception as e:
        return Response({
            'success': False,
            'message': 'Failed to delete user',
            'error': str(e)
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
