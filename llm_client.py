from datetime import datetime
from pathlib import Path

import streamlit as st
from openai import OpenAI

# Verzeichnisse / Dateien
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
ERROR_LOG_FILE = DATA_DIR / "errors.log"


def now_iso() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def log_error(context: str, error: str, session_id: str = None):
    """Einfache Fehler-Logdatei, damit LLM-Probleme nachvollziehbar sind."""
    try:
        with open(ERROR_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(f"{now_iso()}\t{session_id or ''}\t{context}\t{error}\n")
    except Exception:
        pass


# Konfiguration aus Streamlit-Secrets
LLM_BASE_URL = st.secrets.get("LLM_BASE_URL", "https://llm-server.llmhub.t-systems.net/v2")
LLM_API_KEY = st.secrets.get("LLM_API_KEY", "")
LLM_MODEL = st.secrets.get("LLM_MODEL", "Llama-3.3-70B-Instruct")
PROMPT_VERSION = "v1.0-2026-04-07"

_client = None


def get_client():
    """Singleton-Client für die OpenAI-kompatible API von T-Systems."""
    global _client
    if _client is None:
        _client = OpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY)
    return _client


def call_llm(system_prompt: str, messages: list[str], cond: str, session_id: str = None) -> str:
    """
    Ruft das T-Systems-LLM über eine OpenAI-kompatible Schnittstelle auf.

    system_prompt: kompletter System-Prompt (inkl. Stil-Regeln für die Bedingung)
    messages: Liste kurzer Kontextstrings (z.B. Thema + letzte Eingabe),
              wird zu einem User-Content zusammengefügt.
    cond: Bedingung ("low" / "high") – hier nur zur Info weitergereicht.
    session_id: aktuelle Session-ID für das Fehler-Logging.
    """
    if not LLM_API_KEY:
        log_error("call_llm", "LLM_API_KEY fehlt oder ist leer", session_id=session_id)
        return ""

    try:
        client = get_client()
        resp = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": "\n\n".join(messages)},
            ],
            temperature=0.4,
            max_tokens=150,
        )
        reply = resp.choices[0].message.content
        return reply.strip() if reply else ""
    except Exception as e:
        log_error("call_llm", str(e), session_id=session_id)
        return ""