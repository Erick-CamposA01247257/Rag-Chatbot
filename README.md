# 📄 RAG Chatbot

Chatbot que responde preguntas basándose únicamente en documentos propios usando RAG (Retrieval Augmented Generation).

## ¿Qué hace?

- Lee documentos PDF y TXT de la empresa
- Los procesa y guarda en una base de datos vectorial
- Responde preguntas usando solo la información de los documentos
- Cita las fuentes de donde vino cada respuesta
- No inventa información — si no está en el documento lo dice

## Tecnologías

- Python 3.12
- FastAPI
- Claude API (Anthropic)
- LangChain
- ChromaDB
- HuggingFace Embeddings
- Uvicorn

## Instalación

1. Clona el repositorio:
git clone https://github.com/Erick-CamposA01247257/Rag-Chatbot.git

2. Instala las dependencias:
pip install -r requirements.txt

3. Crea un archivo .env con tu API key:
ANTHROPIC_API_KEY=tu_api_key_aqui

4. Agrega tus documentos PDF o TXT en la carpeta documentos/

5. Corre la app:
uvicorn app:app --reload

6. Abre en tu navegador:
http://localhost:8000

## Autor

Erick Campos — Tec de Monterrey