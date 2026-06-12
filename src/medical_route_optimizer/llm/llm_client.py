"""
Adaptador de integração com LLM (Large Language Model).

Suporta dois provedores configuráveis via variável de ambiente LLM_PROVIDER:
    - "openai"  → OpenAI API (gpt-4o-mini por padrão)
    - "groq"    → Groq API (llama-3.3-70b-versatile por padrão, gratuito)

Configuração via variáveis de ambiente:
    LLM_PROVIDER    → "openai" ou "groq" (padrão: "openai")
    OPENAI_API_KEY  → chave de API OpenAI
    GROQ_API_KEY    → chave de API Groq
    LLM_MODEL       → nome do modelo a usar (opcional, usa o padrão do provedor)

O adaptador é desacoplado do restante do sistema:
trocar de provedor exige apenas alterar LLM_PROVIDER, sem alterar código.
"""

import os
from typing import Optional


# Modelos padrão por provedor
DEFAULT_MODELS = {
    "openai": "gpt-4o-mini",
    "groq": "llama-3.3-70b-versatile",
}


def _get_provider() -> str:
    return os.getenv("LLM_PROVIDER", "openai").lower()


def _get_model(provider: str) -> str:
    return os.getenv("LLM_MODEL", DEFAULT_MODELS.get(provider, "gpt-4o-mini"))


def _chamar_openai(prompt: str, model: str, max_tokens: int) -> str:
    """Chama a API da OpenAI e retorna o texto da resposta."""
    try:
        from openai import OpenAI  # type: ignore
    except ImportError:
        raise ImportError(
            "Pacote 'openai' não encontrado. Instale com: pip install openai"
        )

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "Variável de ambiente OPENAI_API_KEY não configurada."
        )

    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=0.3,
    )
    return response.choices[0].message.content.strip()


def _chamar_groq(prompt: str, model: str, max_tokens: int) -> str:
    """Chama a API do Groq e retorna o texto da resposta."""
    try:
        from groq import Groq  # type: ignore
    except ImportError:
        raise ImportError(
            "Pacote 'groq' não encontrado. Instale com: pip install groq"
        )

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "Variável de ambiente GROQ_API_KEY não configurada."
        )

    client = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=0.3,
    )
    return response.choices[0].message.content.strip()


def chamar_llm(
    prompt: str,
    max_tokens: int = 1024,
    provider: Optional[str] = None,
    model: Optional[str] = None,
) -> str:
    """
    Envia um prompt para o LLM configurado e retorna a resposta em texto.

    Parâmetros:
    - prompt: texto do prompt a enviar ao modelo
    - max_tokens: limite de tokens na resposta (padrão 1024)
    - provider: provedor a usar ("openai" ou "groq"); usa LLM_PROVIDER se omitido
    - model: modelo a usar; usa LLM_MODEL ou padrão do provedor se omitido

    Retorno:
    - texto da resposta do modelo

    Exceções:
    - EnvironmentError: se a chave de API não estiver configurada
    - ImportError: se o pacote do provedor não estiver instalado
    - ValueError: se o provedor não for suportado
    """
    provider = provider or _get_provider()
    model = model or _get_model(provider)

    if provider == "openai":
        return _chamar_openai(prompt, model, max_tokens)
    elif provider == "groq":
        return _chamar_groq(prompt, model, max_tokens)
    else:
        raise ValueError(
            f"Provedor LLM '{provider}' não suportado. "
            "Use 'openai' ou 'groq'."
        )
