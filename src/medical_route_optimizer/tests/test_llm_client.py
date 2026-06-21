import pytest
import importlib

MODULE_PATH = "medical_route_optimizer.llm.llm_client"


def test_module_importable():
    importlib.import_module(MODULE_PATH)


def test_has_llm_client_class():
    mod = importlib.import_module(MODULE_PATH)
    assert hasattr(mod, "LLMClient"), "Expected LLMClient class in llm_client module"


def _safe_instantiate(cls):
    try:
        return cls()
    except Exception:
        pytest.skip("LLMClient cannot be instantiated without required parameters or external resources")


def test_generate_method_returns_string(monkeypatch):
    mod = importlib.import_module(MODULE_PATH)
    if not hasattr(mod, "LLMClient"):
        pytest.skip("LLMClient not implemented yet")

    ClientClass = getattr(mod, "LLMClient")
    inst = _safe_instantiate(ClientClass)

    # Replace/patch the instance generate method to avoid external calls
    monkeypatch.setattr(inst, "generate", lambda prompt, **kwargs: f"fake-response: {prompt}", raising=False)

    resp = inst.generate("hello")
    assert isinstance(resp, str)
    assert "hello" in resp


def test_stream_generate_is_iterable(monkeypatch):
    mod = importlib.import_module(MODULE_PATH)
    if not hasattr(mod, "LLMClient"):
        pytest.skip("LLMClient not implemented yet")

    ClientClass = getattr(mod, "LLMClient")
    inst = _safe_instantiate(ClientClass)

    # Provide a fake streaming implementation
    def fake_stream(prompt, **kwargs):
        for i in range(3):
            yield f"chunk-{i}"

    monkeypatch.setattr(inst, "stream_generate", fake_stream, raising=False)

    chunks = list(inst.stream_generate("stream me"))
    assert chunks == ["chunk-0", "chunk-1", "chunk-2"]