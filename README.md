# Asistente RAG Local para Lineamientos de Proyecto

Este proyecto implementa un asistente basado en Retrieval-Augmented Generation (RAG)
utilizando LangChain, Ollama (para LLMs locales), Embeddings locales (HuggingFace)
y una base de datos vectorial local (ChromaDB). Todo está dockerizado para portabilidad.

El asistente responde preguntas basándose *exclusivamente* en el contenido de los
documentos proporcionados en la carpeta `data/`.

## Prerrequisitos

*   Git: [https://git-scm.com/](https://git-scm.com/)
*   Docker: [https://www.docker.com/get-started](https://www.docker.com/get-started)
*   Docker Compose: (Normalmente incluido con Docker Desktop)

## Configuración Inicial

1.  **Clonar el repositorio:**
    ```bash
    git clone <URL-de-tu-repositorio>
    cd tu-proyecto-rag
    ```

2.  **Colocar Documentos:**
    *   Añade tu documento o documentos de lineamientos (PDF, MD, TXT) dentro de la carpeta `data/`.

3.  **Revisar Configuración (Opcional):**
    *   Puedes ajustar los modelos o parámetros en `app/config.py`. Asegúrate de que los modelos (`LLM_MODEL`, `EMBEDDING_MODEL_NAME`) coincidan con los que usarás/descargarás en Ollama y HuggingFace.

## Ejecución con Docker Compose

1.  **Construir e Iniciar los Contenedores:**
    *   Este comando construirá la imagen de tu aplicación (`rag_app`) y descargará/iniciará la imagen de `ollama`.
    ```bash
    docker-compose up --build -d
    ```
    *   `-d` ejecuta los contenedores en segundo plano.

2.  **Descargar Modelos en Ollama (Primera vez):**
    *   Necesitas descargar el modelo LLM y, si usas embeddings de Ollama (no es el caso por defecto aquí, usamos HuggingFace), también el de embeddings. Ejecuta esto en otra terminal mientras `docker-compose` está corriendo:
    ```bash
    # Descargar el modelo de lenguaje (ej: llama3:8b) - ajusta según config.py
    docker-compose exec ollama ollama pull llama3:8b

    # (No necesario por defecto, pero si usaras embeddings de Ollama)
    # docker-compose exec ollama ollama pull nomic-embed-text
    ```
    *   Espera a que la descarga se complete. Puedes verificar los modelos descargados con: `docker-compose exec ollama ollama list`

3.  **Indexar los Documentos:**
    *   Ejecuta el script `main.py` dentro del contenedor `rag_app` con el flag `--index`. Esto cargará los documentos de `data/`, los procesará y creará la base de datos vectorial en `vector_store/`.
    ```bash
    docker-compose exec rag_app python app/main.py --index
    ```
    *   Si quieres forzar la reindexación (borrar y recrear la base de datos), usa:
    ```bash
    docker-compose exec rag_app python app/main.py --index --force-reindex
    ```

4.  **Realizar Preguntas:**
    *   Una vez indexado, puedes hacer preguntas usando el flag `--query`:
    ```bash
    docker-compose exec rag_app python app/main.py --query "¿Cuál es la convención para nombrar variables?"
    ```
    *   Otro ejemplo:
    ```bash
    docker-compose exec rag_app python app/main.py --query "Descríbeme la estructura de carpetas para los componentes React."
    ```
    *   El asistente responderá basándose únicamente en el contenido de tus documentos en `data/`. Si la información no está, indicará que no se encuentra.

## Detener los Contenedores

*   Para detener los servicios:
    ```bash
    docker-compose down
    ```
*   Si quieres eliminar también los volúmenes (¡perderás los modelos descargados de Ollama y la base de datos vectorial!):
    ```bash
    docker-compose down -v
    ```

## Desarrollo

*   Gracias a los volúmenes montados en `docker-compose.yml`, puedes editar el código en `app/` en tu máquina local y los cambios se reflejarán dentro del contenedor `rag_app` sin necesidad de reconstruir la imagen (`docker-compose build`). Simplemente vuelve a ejecutar los comandos `docker-compose exec ...`.
*   Si cambias `requirements.txt` o el `Dockerfile`, necesitarás reconstruir: `docker-compose up --build -d`.


docker-compose exec rag_app python -m app.main --index --force-reindex