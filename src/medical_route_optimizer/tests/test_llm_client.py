import builtins
import importlib
import importlib.util
import sys
import types
import uuid
from pathlib import Path

import pytest

MODULE_PATH = "medical_route_optimizer.llm.llm_client"
MODULE_FILE = Path(__file__).resolve().parents[1] / "llm" / "llm_client.py"


def _load_temp_module(monkeypatch, *, dotenv_available=True, llm_provider=None):
    if llm_provider is None:
        monkeypatch.delenv("LLM_PROVIDER", raising=False)
    else:
        monkeypatch.setenv("LLM_PROVIDER", llm_provider)

    calls = []
    if dotenv_available:
        fake_dotenv = types.ModuleType("dotenv")

        def fake_load_dotenv(dotenv_path=None):
            calls.append(dotenv_path)

        fake_dotenv.load_dotenv = fake_load_dotenv
        monkeypatch.setitem(sys.modules, "dotenv", fake_dotenv)
    else:
        monkeypatch.delitem(sys.modules, "dotenv", raising=False)
        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "dotenv":
                raise ImportError("dotenv ausente")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)

    module_name = f"temp_llm_client_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(module_name, MODULE_FILE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module, calls


def _install_fake_groq(monkeypatch, create_fn):
    class FakeGroq:
        def __init__(self, api_key):
            self.api_key = api_key
            self.chat = types.SimpleNamespace(
                completions=types.SimpleNamespace(create=create_fn)
            )

    monkeypatch.setitem(sys.modules, "groq", types.SimpleNamespace(Groq=FakeGroq))


def test_module_importable():
    importlib.import_module(MODULE_PATH)


def test_import_loads_dotenv_when_provider_missing(monkeypatch, capsys):
    module, calls = _load_temp_module(monkeypatch, dotenv_available=True, llm_provider=None)

    assert module._DOTENV_AVAILABLE is True
    assert calls
    assert Path(calls[0]).name == ".env"
    assert "Variáveis de ambiente carregadas do .env" in capsys.readouterr().out


def test_import_warns_when_dotenv_missing(monkeypatch, capsys):
    module, calls = _load_temp_module(monkeypatch, dotenv_available=False, llm_provider=None)

    assert module._DOTENV_AVAILABLE is False
    assert calls == []
    assert "python-dotenv não instalado" in capsys.readouterr().out


def test_get_model_uses_default_and_strip(monkeypatch):
    module, _ = _load_temp_module(monkeypatch, llm_provider="groq")

    monkeypatch.delenv("LLM_MODEL", raising=False)
    assert module._get_model() == module.DEFAULT_MODEL

    monkeypatch.setenv("LLM_MODEL", " modelo-customizado ")
    assert module._get_model() == "modelo-customizado"


def test_chamar_groq_raises_import_error_without_package(monkeypatch):
    module, _ = _load_temp_module(monkeypatch, llm_provider="groq")
    monkeypatch.delitem(sys.modules, "groq", raising=False)

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "groq":
            raise ImportError("groq ausente")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(ImportError, match="Pacote 'groq' não encontrado"):
        module._chamar_groq("oi", "modelo", 10, api_key="key")


def test_chamar_groq_requires_api_key(monkeypatch):
    module, _ = _load_temp_module(monkeypatch, llm_provider="groq")
    _install_fake_groq(monkeypatch, lambda **_: None)

    with pytest.raises(EnvironmentError, match="Chave de API não fornecida"):
        module._chamar_groq("oi", "modelo", 10, api_key="")


def test_chamar_groq_returns_message_content(monkeypatch):
    module, _ = _load_temp_module(monkeypatch, llm_provider="groq")
    called = {}

    def fake_create(**kwargs):
        called.update(kwargs)
        choice = types.SimpleNamespace(message=types.SimpleNamespace(content=" resposta "), text=None)
        return types.SimpleNamespace(choices=[choice])

    _install_fake_groq(monkeypatch, fake_create)

    resposta = module._chamar_groq("prompt", "modelo", 55, api_key="key")

    assert resposta == "resposta"
    assert called["model"] == "modelo"
    assert called["messages"] == [{"role": "user", "content": "prompt"}]
    assert called["max_tokens"] == 55
    assert called["temperature"] == 0.3
    assert called["timeout"] == 60


def test_chamar_groq_returns_text_fallback(monkeypatch):
    module, _ = _load_temp_module(monkeypatch, llm_provider="groq")

    def fake_create(**kwargs):
        choice = types.SimpleNamespace(message=None, text=" texto ")
        return types.SimpleNamespace(choices=[choice])

    _install_fake_groq(monkeypatch, fake_create)

    assert module._chamar_groq("prompt", "modelo", 10, api_key="key") == "texto"


def test_chamar_groq_returns_warning_when_empty(monkeypatch):
    module, _ = _load_temp_module(monkeypatch, llm_provider="groq")

    def fake_create(**kwargs):
        choice = types.SimpleNamespace(message=types.SimpleNamespace(content=""), text="")
        return types.SimpleNamespace(choices=[choice])

    _install_fake_groq(monkeypatch, fake_create)

    assert module._chamar_groq("prompt", "modelo", 10, api_key="key") == "⚠️ Nenhum conteúdo retornado pela API Groq."


def test_chamar_groq_retries_and_raises_runtime_error(monkeypatch):
    module, _ = _load_temp_module(monkeypatch, llm_provider="groq")
    tentativas = {"total": 0}

    def fake_create(**kwargs):
        tentativas["total"] += 1
        raise ValueError("timeout")

    _install_fake_groq(monkeypatch, fake_create)
    monkeypatch.setattr(module.time, "sleep", lambda *_: None)

    with pytest.raises(RuntimeError, match="Erro na chamada Groq: timeout"):
        module._chamar_groq("prompt", "modelo", 10, api_key="key")

    assert tentativas["total"] == module.RETRY_ATTEMPTS


def test_chamar_llm_requires_api_key(monkeypatch):
    module, _ = _load_temp_module(monkeypatch, llm_provider="groq")

    with pytest.raises(EnvironmentError, match="api_key é obrigatório"):
        module.chamar_llm("prompt", api_key="")


def test_chamar_llm_returns_text_and_metadata(monkeypatch):
    module, _ = _load_temp_module(monkeypatch, llm_provider="groq")
    monkeypatch.setattr(module, "_get_model", lambda: "modelo-padrao")
    monkeypatch.setattr(module, "_chamar_groq", lambda prompt, model, max_tokens, api_key: "resultado")

    resposta = module.chamar_llm("prompt de teste", api_key="key", return_metadata=True)

    assert resposta == {
        "text": "resultado",
        "meta": {
            "provider": "groq",
            "model": "modelo-padrao",
            "prompt": "prompt de teste",
        },
    }


def test_chamar_llm_respects_explicit_model(monkeypatch):
    module, _ = _load_temp_module(monkeypatch, llm_provider="groq")
    called = {}

    def fake_chamar_groq(prompt, model, max_tokens, api_key):
        called.update(
            prompt=prompt,
            model=model,
            max_tokens=max_tokens,
            api_key=api_key,
        )
        return "ok"

    monkeypatch.setattr(module, "_chamar_groq", fake_chamar_groq)

    resposta = module.chamar_llm("pergunta", api_key="segredo", max_tokens=7, model="modelo-x")

    assert resposta == "ok"
    assert called == {
        "prompt": "pergunta",
        "model": "modelo-x",
        "max_tokens": 7,
        "api_key": "segredo",
    }
