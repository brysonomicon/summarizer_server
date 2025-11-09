from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
import os

app = FastAPI()

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
MODEL = "gemma2:9b"

class SummarizeRequest(BaseModel):
    input: str
    max_tokens: int = 32768
    temperature: float = 0.1

class SummarizeResponse(BaseModel):
    summary: str

@app.post("/summarize")
async def summarize(request: SummarizeRequest) -> SummarizeResponse:
    if not request.input.strip():
        raise HTTPException(status_code=400, detail="there is no input")

    prompt = f"""Your job is to summarize the following content into a set of comprehensive markdown notes. The goal is verbosity. Identify the topics that are covered, then provide
    a summary. You should augment the summary with important information. Be as verbose as possible. Complete your answer in steps. First summarize everything. Then go over the summary
    and expand on each point to help pad out the response.
Content:
{request.input}"""

    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "num_predict": request.max_tokens,
                    "temperature": request.temperature
                }
            },
            timeout=600
        )
        response.raise_for_status()

        result = response.json()
        summary = result.get("response", "")

        return SummarizeResponse(summary=summary)
    
    except requests.exceptions.Timeout:
        raise HTTPException(status_code=504, detail="timeout")
    except Exception as err:
        raise HTTPException(status_code=500, detail=f"error: {str(err)}")

@app.get("/health")
async def health():
    return {"status": "healthy", "model": MODEL}