# Usar una imagen base de Python
FROM python:3.11-slim

# Establecer el directorio de trabajo
WORKDIR /app

# Instalar dependencias del sistema necesarias para 'unstructured' y otros
# (Ajusta según las necesidades exactas de tus dependencias)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpoppler-cpp-dev \
    pkg-config \
    poppler-utils \
    tesseract-ocr \
    libmagic-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copiar el archivo de requerimientos
COPY requirements.txt .

# Instalar dependencias de Python
# Usar --no-cache-dir para reducir el tamaño de la imagen
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copiar el código de la aplicación al contenedor
COPY ./app /app/app

# (Opcional) Copiar datos iniciales si quieres que estén en la imagen
# COPY ./data /app/data

# Exponer el puerto si planeas añadir una API más adelante (ej: FastAPI, Flask)
# EXPOSE 8000

# Comando por defecto al iniciar el contenedor (puedes ejecutar main.py con argumentos)
# Aquí lo dejamos listo para ejecutar comandos `docker-compose exec`
CMD ["tail", "-f", "/dev/null"]
# Alternativa si quieres que ejecute algo al iniciar (menos flexible para CLI):
# CMD ["python", "app/main.py"]