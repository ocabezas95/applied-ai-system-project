import sys
import types

import pytest

from src.recommender import GeminiLLM, create_conversational_recommender


class FakeModels:
    def __init__(self):
        self.calls = []

    def generate_content(self, model, contents):
        self.calls.append({"model": model, "contents": contents})
        return types.SimpleNamespace(text="Gemini says these songs fit.")


class FakeClient:
    last_instance = None

    def __init__(self):
        self.models = FakeModels()
        FakeClient.last_instance = self


class FakeLegacyGenerativeModel:
    last_instance = None

    def __init__(self, model):
        self.model = model
        self.calls = []
        FakeLegacyGenerativeModel.last_instance = self

    def generate_content(self, prompt):
        self.calls.append(prompt)
        return types.SimpleNamespace(text="Legacy Gemini says these songs fit.")


class FakeLegacyGenAI(types.ModuleType):
    def __init__(self):
        super().__init__("google.generativeai")
        self.configured_key = None

    def configure(self, api_key):
        self.configured_key = api_key

    GenerativeModel = FakeLegacyGenerativeModel


def install_fake_google_genai(monkeypatch):
    fake_google = types.ModuleType("google")
    fake_genai = types.SimpleNamespace(Client=FakeClient)
    fake_google.genai = fake_genai
    monkeypatch.setitem(sys.modules, "google", fake_google)
    monkeypatch.setitem(sys.modules, "google.genai", fake_genai)


def install_fake_legacy_google_generativeai(monkeypatch):
    fake_google = types.ModuleType("google")
    fake_google.__path__ = []
    fake_legacy_genai = FakeLegacyGenAI()
    fake_google.generativeai = fake_legacy_genai
    monkeypatch.setitem(sys.modules, "google", fake_google)
    monkeypatch.delitem(sys.modules, "google.genai", raising=False)
    monkeypatch.setitem(sys.modules, "google.generativeai", fake_legacy_genai)


def test_gemini_llm_generates_text_with_default_model(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    install_fake_google_genai(monkeypatch)

    llm = GeminiLLM()
    response = llm.generate("Explain these recommendations")

    assert response == "Gemini says these songs fit."
    assert FakeClient.last_instance.models.calls == [
        {
            "model": "gemini-2.5-flash",
            "contents": "Explain these recommendations",
        }
    ]


def test_gemini_llm_falls_back_to_legacy_google_generativeai(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    install_fake_legacy_google_generativeai(monkeypatch)

    llm = GeminiLLM()
    response = llm.generate("Explain these recommendations")

    assert response == "Legacy Gemini says these songs fit."
    assert FakeLegacyGenerativeModel.last_instance.model == "gemini-2.5-flash"
    assert FakeLegacyGenerativeModel.last_instance.calls == ["Explain these recommendations"]


def test_gemini_llm_reports_missing_api_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    llm = GeminiLLM()

    assert llm.generate("hello").startswith("[Gemini Error: GEMINI_API_KEY is not set")


def test_gemini_llm_sanitizes_busy_model_error():
    llm = GeminiLLM.__new__(GeminiLLM)

    message = llm.sanitize_explanation(
        "[Gemini Error: 503 UNAVAILABLE. {'error': {'message': 'This model is currently experiencing high demand.'}}]"
    )

    assert message == (
        "AI explanation is temporarily unavailable because the model is busy. "
        "Showing recommendations based on similarity search."
    )
    assert "503" not in message
    assert "Gemini Error" not in message


def test_factory_selects_gemini_llm(monkeypatch):
    class FakeHybrid:
        pass

    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setattr("src.recommender.create_hybrid_recommender", lambda *args, **kwargs: FakeHybrid())
    install_fake_google_genai(monkeypatch)

    recommender = create_conversational_recommender(
        "data/songs.csv",
        use_llm=True,
        llm_type="gemini",
    )

    assert isinstance(recommender.llm, GeminiLLM)
