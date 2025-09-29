# Docker Setup - API de Reconhecimento Facial

Este documento contém instruções para executar a API de Reconhecimento Facial usando Docker.

## 📋 Pré-requisitos

- Docker (v20.10+)
- Docker Compose (v2.0+)

## 🚀 Execução Rápida

### 1. Clone e Configure

```bash
# Clone o repositório (se ainda não fez)
git clone <repository-url>
cd reconhecimento-facial-com-deepface

# Copie e configure as variáveis de ambiente
cp .env.example .env
# Edite o .env conforme necessário
```

### 2. Execute com Docker Compose

```bash
# Build e execute todos os serviços
docker-compose up --build

# Ou execute em background
docker-compose up -d --build
```

### 3. Acesse a aplicação

- **API**: http://localhost:8000
- **Swagger UI**: http://localhost:8000/swagger/
- **ReDoc**: http://localhost:8000/redoc/
- **Admin**: http://localhost:8000/admin/

## 🐳 Comandos Docker Úteis

### Build apenas a aplicação
```bash
docker build -t face-recognition-api .
```

### Execute apenas o container da aplicação
```bash
docker run -p 8000:8000 face-recognition-api
```

### Acesse o shell do container
```bash
docker-compose exec web bash
```

### Execute comandos Django no container
```bash
# Migrations
docker-compose exec web python manage.py makemigrations
docker-compose exec web python manage.py migrate

# Criar superuser
docker-compose exec web python manage.py createsuperuser

# Collect static files
docker-compose exec web python manage.py collectstatic

# Shell Django
docker-compose exec web python manage.py shell
```

### Logs
```bash
# Ver logs de todos os serviços
docker-compose logs

# Ver logs apenas da aplicação web
docker-compose logs web

# Seguir logs em tempo real
docker-compose logs -f web
```

## 🛠️ Configuração Avançada

### Variáveis de Ambiente

Edite o arquivo `.env` para configurar:

```bash
# Desenvolvimento
DEBUG=True
SECRET_KEY=your-secret-key

# Produção
DEBUG=False
SECRET_KEY=your-very-secure-secret-key
ALLOWED_HOSTS=yourdomain.com,www.yourdomain.com

# Database
DATABASE_URL=postgresql://user:pass@host:port/db
```

### Volumes

- `./media`: Arquivos de mídia (uploads de imagens)
- `./logs`: Logs da aplicação
- `postgres_data`: Dados do PostgreSQL

### Portas

- `8000`: Django application
- `5432`: PostgreSQL database
- `6379`: Redis (opcional)

## 🔧 Desenvolvimento

### Hot reload durante desenvolvimento

```bash
# Monte o código fonte como volume para desenvolvimento
docker-compose -f docker-compose.yml -f docker-compose.dev.yml up
```

### Executar testes

```bash
docker-compose exec web python manage.py test
```

## 📦 Produção

### Docker Compose para produção

```bash
# Use um arquivo docker-compose específico para produção
docker-compose -f docker-compose.prod.yml up -d
```

### Configurações importantes para produção:

1. **Variáveis de ambiente**: Configure `DEBUG=False` e `SECRET_KEY` segura
2. **HTTPS**: Configure reverse proxy (nginx/traefik)
3. **Volumes**: Use volumes externos para dados persistentes
4. **Backup**: Configure backup automático do PostgreSQL
5. **Monitoring**: Adicione ferramentas de monitoramento

## 🔍 Troubleshooting

### Container não inicia
```bash
# Verifique os logs
docker-compose logs web

# Reconstrua as imagens
docker-compose build --no-cache
```

### Problemas de dependências
```bash
# Limpe o cache do Docker
docker system prune -a

# Reconstrua tudo do zero
docker-compose down
docker-compose build --no-cache
docker-compose up
```

### Problemas de banco de dados
```bash
# Reset do banco de dados
docker-compose down
docker volume rm reconhecimento-facial-com-deepface_postgres_data
docker-compose up -d postgres
docker-compose exec web python manage.py migrate
```

### Performance

O container inclui todas as dependências do OpenCV e TensorFlow, então:
- **Build inicial**: Pode levar 10-15 minutos
- **RAM necessária**: Mínimo 4GB, recomendado 8GB
- **Espaço em disco**: ~3-4GB para a imagem

## 📚 Recursos Adicionais

- [Docker Documentation](https://docs.docker.com/)
- [Docker Compose Documentation](https://docs.docker.com/compose/)
- [Django Deployment Checklist](https://docs.djangoproject.com/en/stable/howto/deployment/checklist/)