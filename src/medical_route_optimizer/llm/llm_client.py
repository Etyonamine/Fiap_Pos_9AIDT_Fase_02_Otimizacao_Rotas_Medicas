"""
Adaptador de integração com LLM (Large Language Model) usando apenas Groq.

Características:
- Carregamento automático do .env se ainda não estiver carregado.
- Timeout/retentativa simples (3 tentativas).
- Retorno opcional de metadados (provider, model, prompt).
- Tratamento robusto do retorno da API Groq (message.content ou text).
"""

import os
import time
import asyncio
from typing import Optional, Any
from dotenv import load_dotenv

# ---- Garante que variáveis do .env estejam disponíveis ----
if not os.getenv("LLM_PROVIDER"):
    load_dotenv()
    print("[llm_client] Variáveis de ambiente carregadas do .env")
# -----------------------------------------------------------


DEFAULT_MODEL = "llama-3.3-70b-versatile"
RETRY_ATTEMPTS = 3
RETRY_DELAY = 1.0  # segundos


def _get_model() -> str:
    """Obtém o modelo configurado no ambiente, com fallback para o padrão Groq."""
    return os.getenv("LLM_MODEL", DEFAULT_MODEL).strip()


def _chamar_groq(prompt: str, model: str, max_tokens: int) -> str:
    """Chama o modelo Groq."""
    try:
        from groq import Groq  # type: ignore
    except ImportError as e:
        raise ImportError("Pacote 'groq' não encontrado. Instale com: pip install groq") from e

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError("Variável de ambiente GROQ_API_KEY não configurada.")

    client = Groq(api_key=api_key)
    last_exc = None
    for attempt in range(1, RETRY_ATTEMPTS + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0.3,
                timeout=60,  # aumenta tolerância
            )
            choice = response.choices[0]

            # ✅ Compatibilidade com diferentes formatos de retorno
            if hasattr(choice, "message") and choice.message and choice.message.content:
                return choice.message.content.strip()
            elif hasattr(choice, "text") and choice.text:
                return choice.text.strip()
            else:
                return "⚠️ Nenhum conteúdo retornado pela API Groq."
        except Exception as exc:
            last_exc = exc
            if attempt < RETRY_ATTEMPTS:
                time.sleep(RETRY_DELAY)
            else:
                raise RuntimeError(f"Erro na chamada Groq: {exc}") from exc
    raise RuntimeError("Falha desconhecida na chamada Groq.") from last_exc


def chamar_llm(
    prompt: str,
    max_tokens: int = 1024,
    model: Optional[str] = None,
    return_metadata: bool = False
) -> Any:
    """
    Envia um prompt para o LLM Groq configurado e retorna a resposta em texto.

    Parâmetros:
    - prompt: texto do prompt a enviar ao modelo
    - max_tokens: limite de tokens na resposta (padrão 1024)
    - model: modelo a usar; usa LLM_MODEL ou padrão se omitido
    - return_metadata: se True, retorna um dict com 'text' e 'meta' (model, prompt)

    Retorno:
    - se return_metadata False: string com a resposta
    - se return_metadata True: dict {"text": str, "meta": {"model":..., "prompt":...}}

    Exceções:
    - EnvironmentError: se a chave de API não estiver configurada
    - ImportError: se o pacote 'groq' não estiver instalado
    - RuntimeError: erros de rede/timeout após retentativas
    """
    model = model or _get_model()
    text = _chamar_groq(prompt, model, max_tokens)

    if return_metadata:
        return {
            "text": text,
            "meta": {
                "provider": "groq",
                "model": model,
                "prompt": prompt[:10000]  # truncar se muito grande
            }
        }
    return text
