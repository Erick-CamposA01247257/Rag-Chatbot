from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import anthropic
from dotenv import load_dotenv
import os
from rag import procesar_documentos, cargar_vectordb, buscar_chunks_relevantes

load_dotenv()

app = FastAPI()

client = anthropic.Anthropic()
db = None

class Pregunta(BaseModel):
    mensaje: str

@app.on_event("startup")
async def startup():
    global db
    print("Iniciando servidor...")
    
    if not os.path.exists("vectordb"):
        print("Procesando documentos por primera vez...")
        exito = procesar_documentos()
        if not exito:
            print("Error: no hay documentos en la carpeta 'documentos'")
            return
    
    print("Cargando base de datos vectorial...")
    db = cargar_vectordb()
    print("Sistema listo")

@app.get("/", response_class=HTMLResponse)
async def index():
    with open("templates/index.html", "r", encoding="utf-8") as f:
        return f.read()

@app.post("/chat")
async def chat(pregunta: Pregunta):
    global db
    
    if db is None:
        raise HTTPException(status_code=500, detail="Base de datos no disponible")
    
    chunks = buscar_chunks_relevantes(db, pregunta.mensaje, k=3)
    
    contexto = "\n\n".join([chunk.page_content for chunk in chunks])
    
    prompt = f"""Eres un asistente experto que responde preguntas basándose ÚNICAMENTE en el siguiente contexto. 
Si la respuesta no está en el contexto, di claramente que no tienes esa información.
No inventes información.

CONTEXTO:
{contexto}

PREGUNTA:
{pregunta.mensaje}

RESPUESTA:"""

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    
    return {
        "respuesta": response.content[0].text,
        "fuentes": [chunk.page_content[:100] + "..." for chunk in chunks]
    }

@app.post("/reprocesar")
async def reprocesar():
    global db
    import shutil
    if os.path.exists("vectordb"):
        shutil.rmtree("vectordb")
    exito = procesar_documentos()
    if exito:
        db = cargar_vectordb()
        return {"status": "ok", "mensaje": "Documentos reprocesados correctamente"}
    return {"status": "error", "mensaje": "No hay documentos para procesar"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)