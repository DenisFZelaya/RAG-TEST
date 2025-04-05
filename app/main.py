import argparse
import logging
import sys
from app.rag_pipeline import index_documents, get_embedding_model, get_vector_store, get_llm, setup_rag_chain, query_rag
from app.config import DOCUMENT_SOURCE_DIR

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    parser = argparse.ArgumentParser(description="Asistente RAG local para lineamientos de proyecto.")
    parser.add_argument("--index", action="store_true", help=f"Indexa o reindexa los documentos en '{DOCUMENT_SOURCE_DIR}'.")
    parser.add_argument("--force-reindex", action="store_true", help="Fuerza la reindexación aunque ya exista una base de datos.")
    parser.add_argument("--query", type=str, help="Realiza una pregunta al asistente RAG.")

    args = parser.parse_args()

    rag_chain = None # Inicializar

    # Inicializar componentes comunes
    try:
        embeddings = get_embedding_model()
        vector_store = get_vector_store(embeddings) # Intenta cargarla si existe
        llm = get_llm()
        # Intentar configurar la cadena RAG si la tienda de vectores ya existe
        if vector_store and vector_store._collection.count() > 0: # Verifica si la colección tiene documentos
             logging.info(f"Base de datos vectorial encontrada con {vector_store._collection.count()} elementos.")
             rag_chain = setup_rag_chain(vector_store, llm)
        else:
            logging.warning("La base de datos vectorial está vacía o no se pudo cargar. Necesitas indexar documentos.")

    except Exception as e:
        logging.error(f"Error al inicializar los componentes RAG: {e}")
        sys.exit(1)


    if args.index or args.force_reindex:
        logging.info("Iniciando proceso de indexación...")
        try:
            vector_store = index_documents(force_reindex=args.force_reindex)
            if vector_store:
                 logging.info("Indexación completada. Reconfigurando cadena RAG...")
                 rag_chain = setup_rag_chain(vector_store, llm) # Reconfigurar la cadena con la nueva/actualizada DB
            else:
                 logging.error("La indexación falló o no se encontraron documentos.")
        except Exception as e:
            logging.error(f"Falló la indexación: {e}")
            sys.exit(1)
        # Si solo se pidió indexar, termina aquí.
        if not args.query:
             print("Indexación finalizada.")
             sys.exit(0)


    if args.query:
        if not rag_chain:
            print("\nError: La base de datos vectorial está vacía o no se ha inicializado correctamente.")
            print(f"Por favor, ejecuta primero el comando de indexación: python app/main.py --index")
            sys.exit(1)

        print(f"\nPregunta: {args.query}")
        answer = query_rag(args.query, rag_chain)
        print(f"\nRespuesta:\n{answer}")
    elif not args.index:
        # Si no se dio ninguna acción
        parser.print_help()

if __name__ == "__main__":
    main()