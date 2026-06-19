"""
Adaptador de integração com LLM (Large Language Model).

Melhorias:
- Validação de provedor/modelo.
- Mensagens de erro mais informativas.
- Timeout/retentativa simples (3 tentativas).
- Retorno opcional de metadados (provider, model, prompt).
- Documentação clara das exceções possíveis.
"""

import os
import time
from typing import Optional, Tuple, Dict, Any

DEFAULT_MODELS = {
    "openai": "gpt-4o-mini",
    "groq": "llama-3.3-70b-versatile",
}

RETRY_ATTEMPTS = 3
RETRY_DELAY = 1.0  # segundos


def _get_provider() -> str:
    return os.getenv("LLM_PROVIDER", "openai").lower()


def _get_model(provider: str) -> str:
    return os.getenv("LLM_MODEL", DEFAULT_MODELS.get(provider, DEFAULT_MODELS["openai"]))


def _chamar_openai(prompt: str, model: str, max_tokens: int) -> str:
    try:
        from openai import OpenAI  # type: ignore
    except ImportError as e:
        raise ImportError("Pacote 'openai' não encontrado. Instale com: pip install openai") from e

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("Variável de ambiente OPENAI_API_KEY não configurada.")

    client = OpenAI(api_key=api_key)
    # Retentativa simples
    last_exc = None
    for attempt in range(1, RETRY_ATTEMPTS + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0.3,
            )
            return (response.choices[0].message.content or "").strip()
        except Exception as exc:
            last_exc = exc
            if attempt < RETRY_ATTEMPTS:
                time.sleep(RETRY_DELAY)
            else:
                raise RuntimeError(f"Erro na chamada OpenAI: {exc}") from exc
    raise RuntimeError("Falha desconhecida na chamada OpenAI.") from last_exc


def _chamar_groq(prompt: str, model: str, max_tokens: int) -> str:
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
            )
            return (response.choices[0].message.content or "").strip()
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
    provider: Optional[str] = None,
    model: Optional[str] = None,
    return_metadata: bool = False
) -> str:
    """
    Envia um prompt para o LLM configurado e retorna a resposta em texto.

    Parâmetros:
    - prompt: texto do prompt a enviar ao modelo
    - max_tokens: limite de tokens na resposta (padrão 1024)
    - provider: provedor a usar ("openai" ou "groq"); usa LLM_PROVIDER se omitido
    - model: modelo a usar; usa LLM_MODEL ou padrão do provedor se omitido
    - return_metadata: se True, retorna um dict com 'text' e 'meta' (provider, model, prompt)

    Retorno:
    - se return_metadata False: string com a resposta
    - se return_metadata True: dict {"text": str, "meta": {"provider":..., "model":..., "prompt":...}}

    Exceções:
    - EnvironmentError: se a chave de API não estiver configurada
    - ImportError: se o pacote do provedor não estiver instalado
    - ValueError: se o provedor não for suportado
    - RuntimeError: erros de rede/timeout após retentativas
    """
    provider = (provider or _get_provider()).lower()
    model = model or _get_model(provider)

    if provider not in DEFAULT_MODELS:
        raise ValueError(f"Provedor LLM '{provider}' não suportado. Use 'openai' ou 'groq'.")

    if provider == "openai":
        text = _chamar_openai(prompt, model, max_tokens)
    elif provider == "groq":
        text = _chamar_groq(prompt, model, max_tokens)
    else:
        raise ValueError(f"Provedor LLM '{provider}' não suportado.")

    if return_metadata:
        return {
            "text": text,
            "meta": {
                "provider": provider,
                "model": model,
                "prompt": prompt[:10000]  # truncar se muito grande
            }
        }
    return text
