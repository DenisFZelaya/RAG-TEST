import os
import logging
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import (
    PyPDFLoader,
    DirectoryLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter

from app.config import (
    LLM_MODEL, EMBEDDING_MODEL_NAME, VECTOR_STORE_PATH, COLLECTION_NAME,
    DOCUMENT_SOURCE_DIR, CHUNK_SIZE, CHUNK_OVERLAP, SIMILARITY_TOP_K,
    OLLAMA_BASE_URL
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Inicialización ---

def get_embedding_model():
    """Carga el modelo de embeddings local desde HuggingFace."""
    logging.info(f"Cargando modelo de embeddings: {EMBEDDING_MODEL_NAME}")
    # Usar 'trust_remote_code=True' para modelos como nomic-embed-text
    model_kwargs = {'device': 'cpu', 'trust_remote_code': True}
    encode_kwargs = {'normalize_embeddings': False} # Ajustar según modelo si es necesario
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )
        logging.info("Modelo de embeddings cargado exitosamente.")
        return embeddings
    except Exception as e:
        logging.error(f"Error al cargar el modelo de embeddings: {e}")
        raise

def get_vector_store(embeddings):
    """Obtiene o crea la base de datos vectorial Chroma."""
    logging.info(f"Accediendo a la base de datos vectorial en: {VECTOR_STORE_PATH}")
    vector_store = Chroma(
        persist_directory=VECTOR_STORE_PATH,
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME
    )
    return vector_store

def get_llm():
    """Obtiene el modelo de lenguaje local desde Ollama."""
    logging.info(f"Conectando al LLM: {LLM_MODEL} en {OLLAMA_BASE_URL}")
    try:
        llm = ChatOllama(
            model=LLM_MODEL,
            base_url=OLLAMA_BASE_URL,
            temperature=0.1, # Baja temperatura para respuestas más deterministas/basadas en hechos
            # Puedes añadir más parámetros si lo necesitas (top_k, top_p, etc.)
        )
        # Pequeña prueba de conexión
        llm.invoke("Hola")
        logging.info("Conexión con LLM establecida.")
        return llm
    except Exception as e:
        logging.error(f"Error al conectar con Ollama ({OLLAMA_BASE_URL}): {e}")
        logging.error("Asegúrate de que Ollama esté corriendo y el modelo esté descargado (`ollama pull {LLM_MODEL}`).")
        raise

# --- Procesamiento de Documentos ---

def load_documents(source_dir: str):
    """Carga documentos desde el directorio especificado."""
    logging.info(f"Cargando documentos desde: {source_dir}")
    # Configurar loaders para diferentes tipos de archivo
    loaders = {
        '.pdf': PyPDFLoader,
        '.md': UnstructuredMarkdownLoader,
        '.txt': TextLoader,
    }
    all_docs = []
    # Usar DirectoryLoader para simplificar
    for ext, loader_cls in loaders.items():
        try:
            loader = DirectoryLoader(
                source_dir,
                glob=f"**/*{ext}",
                loader_cls=loader_cls,
                show_progress=True,
                use_multithreading=True # Puede acelerar la carga
            )
            docs = loader.load()
            if docs:
                 logging.info(f"Cargados {len(docs)} documentos con extensión {ext}")
                 all_docs.extend(docs)
            else:
                logging.info(f"No se encontraron documentos con extensión {ext}")

        except Exception as e:
             logging.warning(f"Error cargando archivos {ext}: {e}. Asegúrate de tener las dependencias correctas (ej: pypdf para PDF, unstructured para MD).")


    if not all_docs:
        logging.warning(f"No se cargaron documentos desde {source_dir}. Verifica que haya archivos soportados.")
    return all_docs

def split_documents(documents):
    """Divide los documentos en fragmentos."""
    logging.info(f"Dividiendo {len(documents)} documentos en fragmentos (Chunk Size: {CHUNK_SIZE}, Overlap: {CHUNK_OVERLAP})")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        add_start_index=True, # Ayuda a identificar el origen del fragmento
    )
    chunks = text_splitter.split_documents(documents)
    logging.info(f"Documentos divididos en {len(chunks)} fragmentos.")
    return chunks

def index_documents(force_reindex=False):
    """Carga, divide e indexa los documentos en la base de datos vectorial."""
    vector_store_exists = os.path.exists(VECTOR_STORE_PATH) and os.listdir(VECTOR_STORE_PATH)

    if vector_store_exists and not force_reindex:
        logging.info("La base de datos vectorial ya existe y no se forzó la reindexación. Saltando indexación.")
        embeddings = get_embedding_model()
        vector_store = get_vector_store(embeddings)
        return vector_store

    logging.info("Iniciando proceso de indexación...")
    if not os.path.exists(DOCUMENT_SOURCE_DIR) or not os.listdir(DOCUMENT_SOURCE_DIR):
         logging.error(f"El directorio de documentos '{DOCUMENT_SOURCE_DIR}' está vacío o no existe.")
         raise FileNotFoundError(f"Directorio de documentos no encontrado o vacío: {DOCUMENT_SOURCE_DIR}")

    documents = load_documents(DOCUMENT_SOURCE_DIR)
    if not documents:
        logging.error("No se encontraron documentos para indexar.")
        return None # O manejar el error como prefieras

    chunks = split_documents(documents)
    embeddings = get_embedding_model()

    logging.info(f"Creando/Actualizando base de datos vectorial en: {VECTOR_STORE_PATH}")
    # Si es la primera vez o forzamos, Chroma creará/sobreescribirá los datos al añadir
    # Si el directorio existe pero queremos reindexar, podríamos borrarlo antes,
    # pero Chroma maneja la adición a colecciones existentes o la creación.
    # Si la colección ya existe, add_documents puede añadir duplicados si no se gestiona.
    # Por simplicidad aquí, si forzamos reindexación o no existe, creamos/añadimos.
    # Una estrategia más robusta implicaría borrar la colección o el directorio si force_reindex=True.

    if force_reindex and vector_store_exists:
        logging.warning(f"Forzando reindexación. Se intentará limpiar la colección '{COLLECTION_NAME}' si existe o recrear.")
        # Nota: La limpieza directa de Chroma es compleja. Es más simple borrar y recrear.
        # O gestionar IDs únicos si se añaden incrementalmente.
        # Por ahora, asumimos que crear de nuevo con los mismos datos es aceptable o
        # que Chroma maneja la adición de forma idempotente si los IDs son consistentes (no garantizado aquí).
        # Para una reindexación limpia garantizada, borra el contenido de VECTOR_STORE_PATH antes.
        # import shutil
        # if os.path.exists(VECTOR_STORE_PATH):
        #     shutil.rmtree(VECTOR_STORE_PATH)
        # os.makedirs(VECTOR_STORE_PATH, exist_ok=True)


    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=VECTOR_STORE_PATH,
        collection_name=COLLECTION_NAME
    )
    logging.info("Persistiendo base de datos vectorial...")
    vector_store.persist()
    logging.info("Indexación completada.")
    return vector_store

# --- Consulta RAG ---

def setup_rag_chain(vector_store, llm):
    """Configura la cadena RetrievalQA."""
    logging.info("Configurando la cadena RAG...")

    # Plantilla de Prompt ESTRICTA - ¡Clave para tu requerimiento!
    prompt_template = """
        Eres un asistente experto en la creación de elementos de trabajo (cards) para Azure DevOps, siguiendo estrictamente las plantillas y ejemplos proporcionados en el CONTEXTO.

        TAREA PRINCIPAL: Generar el contenido detallado para una tarjeta de Azure DevOps basándote en la SOLICITUD del usuario y utilizando la ESTRUCTURA/PLANTILLA encontrada en el CONTEXTO recuperado.

        CONTEXTO (Contiene la plantilla/ejemplo de estructura y campos obligatorios):
        {context}

        SOLICITUD DEL USUARIO (Describe qué card se necesita crear):
        {question}

        INSTRUCCIONES DETALLADAS:
        1.  **Identifica la Plantilla:** Examina el CONTEXTO para encontrar la estructura, los campos y el formato definidos para el tipo de tarjeta mencionada o implícita en la SOLICITUD (Task, Bug, PBI, etc.).
        2.  **Adhiérete a la Estructura:** Tu salida DEBE seguir *exactamente* la estructura y los nombres de los campos presentes en la plantilla del CONTEXTO. NO inventes campos nuevos. NO omitas campos obligatorios de la plantilla.
        3.  **Genera Contenido Relevante:** Rellena los campos de la plantilla (Título, Descripción, Criterios de Aceptación, Pasos Técnicos, etc.) con información específica y detallada, inferida a partir de la SOLICITUD del usuario. Debes "pensar" en los detalles necesarios para cumplir la solicitud dentro del marco de la plantilla. ¡Sé creativo pero realista!
        4.  **Formato de Salida:** Presenta la tarjeta completa en formato MARKDOWN, utilizando encabezados (## o ###) para los nombres de los campos principales (como Título) y listas o texto normal para el contenido, imitando cómo se vería en Azure DevOps. Asegúrate de incluir TODOS los campos de la plantilla recuperada.
        5.  **Manejo de Plantilla Ausente:** Si el CONTEXTO recuperado no contiene una plantilla o ejemplo claro y relevante para la SOLICITUD, responde únicamente: "No se encontró una plantilla adecuada en el documento de lineamientos para crear este tipo de tarjeta."

        SALIDA (Tarjeta Azure DevOps en formato Markdown):
        """
    QA_CHAIN_PROMPT = PromptTemplate.from_template(prompt_template)

    # Configurar el Retriever
    retriever = vector_store.as_retriever(
        search_type="similarity", # Tipo de búsqueda
        search_kwargs={'k': SIMILARITY_TOP_K} # Número de fragmentos a recuperar
    )

    # Crear la cadena RetrievalQA
    rag_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff", # Pone todos los fragmentos recuperados en el prompt (adecuado para pocos fragmentos)
                            # Otros tipos: map_reduce, refine, map_rerank si el contexto es muy grande
        retriever=retriever,
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
        return_source_documents=True # Opcional: para ver qué fragmentos se usaron
    )
    logging.info("Cadena RAG configurada exitosamente.")
    return rag_chain

def query_rag(question: str, rag_chain):
    """Realiza una consulta a la cadena RAG."""
    if not rag_chain:
        logging.error("La cadena RAG no está inicializada. Ejecuta la indexación primero.")
        return "Error: El sistema RAG no está listo."

    logging.info(f"Recibida pregunta: {question}")
    logging.info("Realizando búsqueda de similitud y generando respuesta...")

    # IMPORTANTE: No pasamos historial de chat aquí. Cada llamada es independiente.
    response = rag_chain.invoke({"query": question})

    answer = response.get('result', "No se pudo obtener una respuesta.")
    source_docs = response.get('source_documents', [])

    logging.info(f"Respuesta generada: {answer}")
    if source_docs:
        logging.info(f"Basado en {len(source_docs)} fragmentos recuperados:")
        # for i, doc in enumerate(source_docs):
        #     logging.debug(f"  Fragmento {i+1}: {doc.page_content[:150]}...") # Muestra inicio del fragmento
    else:
        logging.warning("No se recuperaron fragmentos de contexto para esta pregunta.")

    return answer