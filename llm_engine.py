from llama_cpp import Llama

class LLMEngine:
    def __init__(self, model_path):
        self.model_path = model_path

        # CONFIG ESTABLE PARA WINDOWS (evita access violation)
        self.llm = Llama(
            model_path=model_path,
            n_ctx=2048,         # evita overflow
            n_batch=256,        # Windows necesita batches pequeños
            n_threads=6,        # estable para CPUs modernas
            flash_attn=False,   # evita crashes
            rope_scaling={"type": "linear", "factor": 1.0},
            verbose=False
        )

    def generate(self, context, query, max_tokens=200):
        prompt = f"""
Eres un asistente experto en análisis laboral. Usa el contexto para responder.

CONTEXT:
{context}

PREGUNTA:
{query}

RESPUESTA:
"""
        out = self.llm(prompt, max_tokens=max_tokens, temperature=0.1)
        return out["choices"][0]["text"].strip()
