from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import os

DIRECTORIO_DOCS = "documentos"
DIRECTORIO_DB = "vectordb"

def cargar_documentos():
    documentos = []
    for archivo in os.listdir(DIRECTORIO_DOCS):
        ruta = os.path.join(DIRECTORIO_DOCS, archivo)
        if archivo.endswith(".pdf"):
            loader = PyPDFLoader(ruta)
        elif archivo.endswith(".txt"):
            loader = TextLoader(ruta, encoding="utf-8")
        else:
            continue
        documentos.extend(loader.load())
        print(f"Cargado: {archivo}")
    return documentos
def crear_chunks(documentos):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(documentos)
    print(f"Total de chunks creados: {len(chunks)}")
    return chunks

def crear_vectordb(chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DIRECTORIO_DB
    )
    print("Base de datos vectorial creada")
    return db

def cargar_vectordb():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    db = Chroma(
        persist_directory=DIRECTORIO_DB,
        embedding_function=embeddings
    )
    return db

def buscar_chunks_relevantes(db, pregunta, k=3):
    resultados = db.similarity_search(pregunta, k=k)
    return resultados

def procesar_documentos():
    print("Cargando documentos...")
    documentos = cargar_documentos()
    if not documentos:
        print("No hay PDFs en la carpeta 'documentos'")
        return False
    print("Creando chunks...")
    chunks = crear_chunks(documentos)
    print("Creando base de datos vectorial...")
    crear_vectordb(chunks)
    return True