from datetime import datetime
from pathlib import Path

import streamlit as st
from openai import OpenAI

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
ERROR_LOG_FILE = DATA_DIR / "errors.log"

_client = None


def now_iso() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def log_error(context: str, error: str, session_id: str | None = None):
    try:
        with open(ERROR_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(f"{now_iso()}\t{session_id or ''}\t{context}\t{error}\n")
    except Exception:
        pass


def get_llm_config():
    base_url = st.secrets.get("LLM_BASE_URL", "https://api.openai.com/v1")
    api_key = st.secrets.get("LLM_API_KEY", "")
    model = st.secrets.get("LLM_MODEL", "gpt-4.1-mini")
    return base_url, api_key, model


def get_client():
    global _client
    if _client is None:
        base_url, api_key, _ = get_llm_config()
        _client = OpenAI(base_url=base_url, api_key=api_key)
    return _client


def call_llm(system_prompt: str, messages: list[str], session_id: str | None = None) -> str:
    base_url, api_key, model = get_llm_config()

    if not api_key:
        log_error("call_llm", "LLM_API_KEY fehlt oder ist leer", session_id=session_id)
        return ""

    try:
        client = get_client()
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": "\n\n".join(messages)},
            ],
            temperature=0.4,
            max_tokens=150,
        )
        reply = (resp.choices[0].message.content or "").strip()
        return reply

    except Exception as e:
        log_error("call_llm", f"EXC: {type(e).__name__}: {e}", session_id=session_id)
        return ""
