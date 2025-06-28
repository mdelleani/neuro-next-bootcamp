import httpx

# === CONFIGURATION ===
HOST = "34.76.8.222"  # <-- Insert IP or hostname
HOST_MED = "35.240.108.178"
PORT = 8000

VLLM_ENDPOINT = f"http://{HOST}:{PORT}/v1/chat/completions"
VLLM_ENDPOINT_MED = f"http://{HOST_MED}:{PORT}/v1/chat/completions"

GENERALIST_MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"
MEDGEMMA_MODEL_NAME = "unsloth/medgemma-4b-it-bnb-4bit"

# === GENERALIST LLM CALL ===
async def ask_generalist_llm(messages, response_format = None):
    

    if response_format:
        payload = {
                "model": GENERALIST_MODEL_NAME,
                "messages": messages,
                "max_tokens": 500,
                "temperature": 0.6,
                'response_format': response_format
            }
    else:
        payload = {
            "model": GENERALIST_MODEL_NAME,
            "messages": messages,
            "max_tokens": 500,
            "temperature": 0.6
        }

    try:
        async with httpx.AsyncClient(timeout=50) as client:
            response = await client.post(VLLM_ENDPOINT, json=payload)
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"]
    except httpx.ConnectError:
        return "Errore di connessione al server vLLM. Assicurati che l'endpoint sia corretto e il server sia attivo."
    except httpx.HTTPStatusError as e:
        return f"Errore HTTP dal server vLLM: {e}. Controlla il modello e i parametri."
    except Exception as e:
        return f"Si è verificato un errore inatteso: {e}"

# === SPECIALIZED LLM CALL ===
async def ask_specialized_llm(messages, response_format = None):

    if response_format:
        payload = {
                "model": MEDGEMMA_MODEL_NAME,
                "messages": messages,
                "max_tokens": 500,
                "temperature": 0.6,
                'response_format': response_format
            }
    else:
        
        payload = {
            "model": MEDGEMMA_MODEL_NAME,
            "messages": messages,
            "max_tokens": 500,
            "temperature": 0.6
        }

    try:
        async with httpx.AsyncClient(timeout=50) as client:
            response = await client.post(VLLM_ENDPOINT_MED, json=payload)
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"]
    except httpx.ConnectError:
        return "Errore di connessione al server vLLM. Assicurati che l'endpoint sia corretto e il server sia attivo."
    except httpx.HTTPStatusError as e:
        return f"Errore HTTP dal server vLLM: {e}. Controlla il modello e i parametri."
    except Exception as e:
        return f"Si è verificato un errore inatteso: {e}"
