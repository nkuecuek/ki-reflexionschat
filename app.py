import csv
import string
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict
from urllib.parse import quote

import pandas as pd
import streamlit as st
from openai import OpenAI

st.set_page_config(
    page_title="KI-Reflexionschat",
    page_icon="💬",
    layout="centered",
    initial_sidebar_state="expanded",
)

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
LOG_FILE = DATA_DIR / "chat_logs.csv"
SUMMARY_FILE = DATA_DIR / "chat_sessions.csv"

TEMPERATURE = 0.3
MAX_RETRIES = 3
PROD_ROUNDS = 6
DEFAULT_ROUNDS = PROD_ROUNDS

SAFETY_KEYWORDS = [
    "suizid",
    "ich will sterben",
    "ich möchte sterben",
    "nicht mehr leben",
    "ich will nicht mehr leben",
    "mich umbringen",
    "ich bringe mich um",
    "bring mich um",
    "selbst verletzen",
    "selbstverletzung",
    "ich verletze mich selbst",
    "jemanden umbringen",
    "jemanden verletzen",
]

FORBIDDEN_PHRASES = [
    "ich fühle", "ich fuehle",
    "ich bin für dich da", "ich bin fuer dich da",
    "danke für dein vertrauen", "danke fuer dein vertrauen",
    "es tut mir leid",
    "du solltest", "du musst",
    "ich verstehe dich", "ich fuehle mit dir",
    "du bist nicht allein",
    "ich begleite dich",
    "mein rat",
    "am besten wäre",
    "versuche doch",
    "das klingt belastend",
    "vielleicht steckt",
    "es könnte sein",
    "vermutlich",
    "deutet darauf hin",
]

QUESTION_START_WORDS = ["Was", "Wie", "Woran"]
FORBIDDEN_QUESTION_STARTS = [
    "Warum", "Wieso", "Weshalb", "Wann", "Wer", "Welche", "Inwiefern"
]

PSYCH_TERMS = [
    "depression", "depressiv",
    "angststörung", "angststoerung",
    "trauma", "traumatisch",
    "diagnose",
    "psychodynamisch",
    "bindungsstil",
    "vermeidungsmuster",
    "symptom",
    "störung", "stoerung",
]

INTRO_TEXT = """
Willkommen zur KI-Reflexionssession.

In dieser kurzen Session reflektierst du ein aktuelles studienbezogenes Thema. Das System ist ein KI-basiertes Reflexionstool und keine Beratung oder Therapie. Es unterstützt dabei, ein studienbezogenes Anliegen kurz zu ordnen und in einer begrenzten Reflexion zu betrachten.

Die Reflexion umfasst insgesamt sechs kurze Antwortschritte. Danach endet der Chatteil automatisch. Bitte kehre anschließend zum Fragebogen zurück.
"""


@st.cache_resource
def get_openai_client() -> OpenAI:
    api_key = None
    for key_name in ["OPENAI_API_KEY", "LLM_API_KEY"]:
        try:
            candidate = st.secrets[key_name]
            if candidate and str(candidate).strip():
                api_key = candidate
                break
        except Exception:
            pass

    if not api_key:
        raise RuntimeError(
            "Kein API-Key gefunden. Erwartet wird OPENAI_API_KEY oder LLM_API_KEY in st.secrets."
        )

    base_url = st.secrets.get("LLM_BASE_URL", "[api.openai.com](https://api.openai.com/v1)")
    return OpenAI(api_key=api_key, base_url=base_url)


def get_model_name() -> str:
    return st.secrets.get("LLM_MODEL", "gpt-4.1-mini")


def now_iso() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def ensure_csv_files():
    if not LOG_FILE.exists():
        with open(LOG_FILE, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(
                ["session_id", "pid", "cond", "turn", "role", "text", "timestamp"]
            )

    if not SUMMARY_FILE.exists():
        with open(SUMMARY_FILE, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(
                [
                    "session_id",
                    "pid",
                    "cond",
                    "raw_cond",
                    "session_start",
                    "session_end",
                    "completed_chat",
                    "turns_completed",
                    "user_messages_count",
                    "topic",
                    "safety_triggered",
                    "validation_fail_count",
                    "fallback_count",
                    "closing_validation_fail_count",
                    "closing_fallback_count",
                ]
            )


def get_param(name: str, default: str = "") -> str:
    try:
        value = st.query_params.get(name, default)
        if isinstance(value, list):
            return value[-1] if value else default
        return str(value or default)
    except Exception:
        return default


def get_debug_mode() -> bool:
    raw_debug = get_param("debug", "0").strip().lower()
    return raw_debug in {"1", "true", "yes", "on"}


def validate_topic_input(text: str) -> bool:
    return len(text.strip()) >= 10


def is_very_short_user_input(text: str) -> bool:
    cleaned = (text or "").strip()
    if not cleaned:
        return True
    return len(cleaned.split()) <= 4


def extract_question_part(normalized: str) -> str:
    before_q = normalized[:-1].strip() if normalized.endswith("?") else normalized.strip()
    sentence_parts = [part.strip() for part in before_q.split(".") if part.strip()]
    if sentence_parts:
        return sentence_parts[-1]
    return before_q


def validate_response(text: str) -> bool:
    if not text:
        return False

    raw = (text or "").strip()
    normalized = " ".join(raw.replace("\n", " ").split()).strip()
    if not normalized:
        return False

    if "\n" in raw:
        return False
    if any(sep in raw for sep in ["- ", "•", "* "]):
        return False

    if normalized.count("?") != 1 or not normalized.endswith("?"):
        return False

    words = normalized.split()
    if len(words) < 8 or len(words) > 70:
        return False

    lower = normalized.lower()

    if any(phrase in lower for phrase in FORBIDDEN_PHRASES):
        return False

    if any(term in lower for term in PSYCH_TERMS):
        return False

    question_part = extract_question_part(normalized)
    if not any(question_part.startswith(prefix) for prefix in QUESTION_START_WORDS):
        return False

    if any(question_part.startswith(prefix) for prefix in FORBIDDEN_QUESTION_STARTS):
        return False

    if normalized.startswith(tuple(QUESTION_START_WORDS)):
        return False

    return True


def validate_closing_response(text: str) -> bool:
    if not text:
        return False

    normalized = " ".join((text or "").replace("\n", " ").split()).strip()
    if not normalized:
        return False

    if "?" in normalized:
        return False

    if any(sep in text for sep in ["\n", "- ", "•", "*"]):
        return False

    words = normalized.split()
    if len(words) < 10 or len(words) > 60:
        return False

    lower = normalized.lower()
    if any(phrase in lower for phrase in FORBIDDEN_PHRASES):
        return False

    if any(term in lower for term in PSYCH_TERMS):
        return False

    return True


def fallback_reply(cond: str, user_text: str = "") -> str:
    short = (user_text or "").strip().lower()

    unsure_forms = {
        "ich weiß nicht", "ich weiss nicht", "weiß nicht", "weiss nicht",
        "keine ahnung", "nicht sicher", "kp", "idk", "schwer zu sagen", "unsicher"
    }

    if short in unsure_forms or is_very_short_user_input(user_text):
        if cond == "high":
            return (
                "Gerade bleibt noch unklar, woran du dein studienbezogenes Thema am ehesten festmachst. "
                "Was fällt dir daran im Moment als Erstes auf?"
            )
        return (
            "Im Moment bleibt noch unklar, welcher konkrete Punkt an diesem studienbezogenen Thema zuerst greifbar wird. "
            "Was fällt dazu als Erstes auf?"
        )

    if cond == "high":
        return (
            "Du beschreibst gerade mehrere Punkte rund um dein studienbezogenes Thema. "
            "Was zeigt sich davon im Alltag im Moment am deutlichsten?"
        )
    return (
        "Hier werden mehrere Punkte rund um das studienbezogene Thema beschrieben. "
        "Was zeigt sich davon im Alltag im Moment am deutlichsten?"
    )


def closing_fallback(cond: str) -> str:
    if cond == "high":
        return (
            "Du hast dein studienbezogenes Thema in dieser kurzen Reflexion weiter beschrieben "
            "und einige konkrete Punkte benannt. Damit endet die Reflexion zu deinem Thema."
        )
    return (
        "Das studienbezogene Thema wurde in dieser kurzen Reflexion weiter beschrieben "
        "und durch konkrete Punkte eingegrenzt. Damit endet die Reflexion zu diesem Thema."
    )


def check_safety(user_text: str) -> bool:
    text = (user_text or "").lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return any(kw in text for kw in SAFETY_KEYWORDS)


def build_system_prompt(cond: str, max_rounds: int) -> str:
    base = f"""
Du bist ein KI-basiertes Reflexionstool im Rahmen einer kurzen psychologischen Studie im Hochschulkontext.

ROLLE UND GRENZEN
- Du bist ein transparentes KI-System und keine menschliche Person.
- Du ersetzt keine Therapie, kein Coaching und keine Beratung.
- Du gibst keine Lösungen, keine Handlungsempfehlungen und keine Ziele vor.
- Du stellst keine Diagnosen und verwendest keine klinischen Fachbegriffe (z. B. Angststörung, Trauma, Diagnose, Störung, Symptom, Bindungsstil, Perfektionismus als Fachbegriff, psychodynamisch).
- Du simulierst keine menschliche Beziehung, keine Empathie und keine emotionale Begleitung.

THEMENRAHMEN
- Die Person schreibt über ein studienbezogenes Anliegen.
- Der Schwerpunkt bleibt beim studienbezogenen Thema.
- Wenn andere Lebensbereiche erwähnt werden, greife sie nur auf, sofern die Person sie selbst genannt hat.
- Alltagsnahe Begriffe wie Stress, Überforderung, Druck oder Unsicherheit darfst du aufgreifen und spiegeln – das sind keine verbotenen Begriffe.

ZIEL DER INTERAKTION
- Deine Funktion ist minimale kognitive Strukturierung.
- Du hilfst dabei, diffuse oder allgemeine Beschreibungen in etwas Konkreteres zu überführen.
- Du veränderst nicht das Thema, interpretierst nicht und entwickelst keine Theorie über die Person.
- Du bleibst möglichst auf derselben Bedeutungsebene wie die Aussage der Person.
- Du formulierst näher an der subjektiven Beschreibung als an einer analytischen Einordnung.
- Du kommentierst die Kommunikation mit dem KI-System nicht analytisch als eigenes Thema.

GRUNDREGEL FÜR JEDE ANTWORT
- Jede Antwort greift die letzte Nutzereingabe knapp auf und endet mit genau einer offenen Frage.
- Die Antwort soll kurz, anschlussfähig und natürlich klingen.
- Sie darf das Gesagte leicht ordnen oder fokussieren, aber keine neuen Inhalte hinzufügen.
- Wenn die neue Eingabe klar an den direkt vorherigen Schritt anknüpft, darfst du genau einen solchen Punkt kurz mit aufgreifen.
- Du fasst nicht mehrere frühere Turns zusammen.
- Die Sitzung umfasst insgesamt {max_rounds} Nutzereingaben.
- Die letzte Nutzereingabe wird mit einer kurzen Abschlussantwort ohne neue Frage beantwortet.

SPIEGELUNG
- Greife aus der letzten Eingabe einen zentralen Punkt oder höchstens zwei eng verbundene Punkte auf.
- Spiegele nicht nur den Wortlaut, sondern den naheliegenden Schwerpunkt, die Unklarheit, den schwierigen Moment oder die Spannung in dem, was beschrieben wurde.
- Bleibe dabei eng an der Formulierung und Logik der Person.
- Füge keine Ursachen, Motive, Bewertungen oder Deutungen hinzu.
- Sprich nicht abstrakt über „die Aussage" oder „die Kommunikation", sondern über den konkret beschriebenen Inhalt.

FRAGE
- Stelle genau eine offene Frage.
- Die Frage steht immer am Ende der Antwort.
- Erlaubte Fragestarter: „Was", „Wie", „Woran" – diese drei Wörter sind erlaubt und ausreichend.
- Nicht erlaubt: Warum, Wieso, Weshalb, Wann, Wer, Welche, Inwiefern.
- Die Frage soll leicht beantwortbar sein und an konkrete Beobachtungen, Situationen, Abläufe oder unmittelbare Gedanken anknüpfen.
- Die Frage darf keinen neuen Themenbereich einführen.
- Die Frage soll keine Analyse verlangen, sondern den nächsten greifbaren Schritt der Reflexion öffnen.

UMGANG MIT KURZEN ODER UNKLAREN ANTWORTEN
- Auch sehr kurze Antworten wie „Ich weiß nicht", „Keine Ahnung" oder einzelne Stichworte sind ernst zu nehmen.
- Greife in solchen Fällen die Unklarheit oder das Feststecken direkt auf.
- Stelle eine kleine, einfache Frage, die hilft, einen ersten konkreten Punkt sichtbar zu machen.

UMGANG MIT SCHWIERIGKEITEN IN DER INTERAKTION
- Wenn die Person Unsicherheit, Frustration oder Schwierigkeiten mit dem Chat äußert, greife die konkret genannte Schwierigkeit auf.
- Analysiere nicht die Gesprächsdynamik.
- Bleibe bei dem Problem, das die Person gerade benennt.

VERBOTENE FORMULIERUNGEN – diese Sätze oder sinngemäß sehr ähnliche Wendungen sind nicht erlaubt:
- „Das klingt belastend"
- „Ich kann verstehen, dass ..."
- „Das muss schwer für dich sein"
- „Vielleicht steckt dahinter ..."
- „Es könnte sein, dass ..."
- „Vermutlich ..."
- „Ich bin für dich da"
- „Danke für dein Vertrauen"
- „Es tut mir leid"
- „Du bist nicht allein"
- „Ich begleite dich"
- „Du solltest ..." / „Du musst ..."
- „Mein Rat wäre ..."
- „Am besten wäre ..."
- „Versuche doch ..."

FORMATREGELN
- Du antwortest auf Deutsch.
- Deine Antwort ist genau ein zusammenhängender Fließtextabschnitt.
- Du verwendest keine Bulletpoints, keine Listen und keine mehreren Absätze.
- Deine Antwort enthält genau ein Fragezeichen, das am Ende steht.
- Deine Antwort umfasst ungefähr 15 bis 70 Wörter.
"""

    low_style = """
STILREGELN – BEDINGUNG A (sachlich)
- Formuliere sachlich, ruhig und inhaltsbezogen.
- Du-Ansprache ist möglich, aber setze sie sparsam ein – eher seltener als häufiger.
- Bevorzuge im Zweifel eine neutrale, inhaltsbezogene Formulierung gegenüber einer persönlich adressierten.
- Die Sprache soll nüchtern wirken, aber trotzdem flüssig und verständlich klingen.
- Nicht mechanisch, nicht steif, nicht akademisch.
- Kein Smalltalk, keine Wärme, keine fürsorgliche Rahmung.
"""

    high_style = """
STILREGELN – BEDINGUNG B (natürlich)
- Formuliere etwas natürlicher und leicht personenbezogener als in Bedingung A.
- Du-Ansprache ist selbstverständlich und darf regelmäßig verwendet werden.
- Die Sprache darf etwas gesprächsnäher wirken, aber nicht warm, fürsorglich oder therapeutisch.
- Nicht locker, nicht casual, nicht emotionalisierend.
- Kein Smalltalk, keine fürsorgliche Rahmung.
"""

    if cond == "high":
        return base + "\n" + high_style
    return base + "\n" + low_style


def build_closing_prompt(cond: str, max_rounds: int) -> str:
    base = f"""
Du bist ein KI-basiertes Reflexionstool im Rahmen einer kurzen psychologischen Studie im Hochschulkontext.

Dies ist die letzte Antwort der Reflexionsinteraktion (Runde {max_rounds} von {max_rounds}).

AUFGABE
- Formuliere eine kurze Abschlussantwort.
- Greife den zuletzt genannten Punkt der Person knapp auf – nicht mehr als einen.
- Füge keine neuen Inhalte, Deutungen, Ratschläge oder Zukunftsaussagen hinzu.
- Stelle keine neue Frage.
- Markiere klar, dass die Reflexion jetzt endet.

VERBOTENE FORMULIERUNGEN
- Keine Sätze wie „Das klingt belastend", „Ich bin für dich da", „Du hast das gut gemacht", „Das war mutig".
- Keine Bewertungen der Person oder der Interaktion.
- Keine therapeutische, tröstende oder beratende Sprache.
- Alltagsnahe Begriffe wie Stress, Überforderung oder Druck darfst du aufgreifen, wenn die Person sie selbst verwendet hat.

FORMATREGELN
- Deutsch.
- Ein kurzer Fließtextabschnitt.
- Kein Fragezeichen.
- Keine Bulletpoints, keine Listen.
- 10 bis 60 Wörter.
"""

    low_style = """
STILREGELN – BEDINGUNG A (sachlich)
- Sachlich, nüchtern, inhaltsbezogen.
- Du-Ansprache möglich, aber sparsam einsetzen.
- Beispiel: „Der zuletzt beschriebene Punkt wurde in dieser Reflexion aufgegriffen und etwas weiter eingegrenzt. Damit endet die Reflexion zu diesem Thema."
"""

    high_style = """
STILREGELN – BEDINGUNG B (natürlich)
- Natürlich, leicht personenbezogen, aber nicht empathisch oder fürsorglich.
- Du-Ansprache selbstverständlich.
- Beispiel: „Du hast den zuletzt genannten Punkt in dieser kurzen Reflexion weiter beschrieben. Damit endet die Reflexion zu deinem Thema."
"""

    if cond == "high":
        return base + "\n" + high_style
    return base + "\n" + low_style


def get_recent_context(messages: List[Dict[str, str]], max_pairs: int = 3) -> str:
    pairs = []
    i = 0
    while i < len(messages):
        if messages[i]["role"] == "user":
            user_msg = messages[i]["content"]
            asst_msg = (
                messages[i + 1]["content"]
                if i + 1 < len(messages) and messages[i + 1]["role"] == "assistant"
                else ""
            )
            pairs.append((user_msg, asst_msg))
            i += 2
        else:
            i += 1

    recent = pairs[-max_pairs:]
    lines = []
    for u, a in recent:
        lines.append(f"User: {u}")
        if a:
            lines.append(f"Assistent: {a}")
    return "\n".join(lines)


def build_api_messages(
    system_prompt: str,
    topic: str,
    turn: int,
    max_rounds: int,
    user_text: str
) -> List[Dict[str, str]]:
    recent_context = get_recent_context(st.session_state.messages, max_pairs=3)

    user_payload = (
        f"Studienbezogenes Hauptthema der Person: {topic}\n"
        f"Aktuelle Rundenzahl: {turn} von {max_rounds}\n"
    )

    if recent_context:
        user_payload += (
            "Bisheriger Gesprächsverlauf (die letzten Schritte, User und Assistent abwechselnd). "
            "Beziehe dich primär auf die unmittelbar letzte Nutzereingabe. "
            "Die früheren Schritte dienen nur als Hintergrund – fasse sie nicht zusammen:\n"
            f"{recent_context}\n"
        )

    user_payload += (
        f"Unmittelbar letzte Eingabe der Person: {user_text}\n"
        "Formuliere jetzt genau eine Antwort gemäß allen Regeln."
    )

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_payload},
    ]


def call_llm(
    system_prompt: str,
    topic: str,
    turn: int,
    max_rounds: int,
    user_text: str,
    temperature: float,
) -> str:
    client = get_openai_client()
    model_name = get_model_name()

    messages = build_api_messages(
        system_prompt=system_prompt,
        topic=topic,
        turn=turn,
        max_rounds=max_rounds,
        user_text=user_text,
    )

    st.session_state.last_prompt_excerpt = messages[-1]["content"]

    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=temperature,
        max_tokens=180,
    )

    content = response.choices[0].message.content
    if content is None:
        raise RuntimeError("LLM-Antwort enthält keinen Textinhalt.")

    return content.strip()


def generate_llm_reply(user_text: str, cond: str, topic: str, turn: int, max_rounds: int) -> str:
    system_prompt = build_system_prompt(cond=cond, max_rounds=max_rounds)

    st.session_state.last_llm_error = ""
    st.session_state.last_llm_raw_reply = ""
    st.session_state.last_llm_status = ""

    temperatures = [0.3, 0.2, 0.1][:MAX_RETRIES]

    for attempt, temp in enumerate(temperatures, start=1):
        try:
            raw_reply = call_llm(
                system_prompt=system_prompt,
                topic=topic,
                turn=turn,
                max_rounds=max_rounds,
                user_text=user_text,
                temperature=temp,
            )

            st.session_state.last_llm_raw_reply = raw_reply

            if validate_response(raw_reply):
                st.session_state.last_llm_status = f"LLM ok in Versuch {attempt}"
                return " ".join(raw_reply.split())

            st.session_state.validation_fail_count += 1
            st.session_state.last_llm_error = (
                f"Validierung fehlgeschlagen in Versuch {attempt}. Antwort: {raw_reply}"
            )
            st.session_state.last_llm_status = f"Validierung fehlgeschlagen in Versuch {attempt}"
            time.sleep(0.4)

        except Exception as e:
            st.session_state.last_llm_error = f"{type(e).__name__}: {e}"
            st.session_state.last_llm_status = f"Fehler in Versuch {attempt}"
            time.sleep(0.7)

    st.session_state.fallback_count += 1
    st.session_state.last_llm_status = "Fallback ausgelöst"
    return fallback_reply(cond, user_text=user_text)


def generate_closing_reply(user_text: str, cond: str, topic: str, turn: int, max_rounds: int) -> str:
    system_prompt = build_closing_prompt(cond=cond, max_rounds=max_rounds)

    st.session_state.last_llm_error = ""
    st.session_state.last_llm_raw_reply = ""
    st.session_state.last_llm_status = ""

    temperatures = [0.2, 0.1, 0.0][:MAX_RETRIES]

    for attempt, temp in enumerate(temperatures, start=1):
        try:
            raw_reply = call_llm(
                system_prompt=system_prompt,
                topic=topic,
                turn=turn,
                max_rounds=max_rounds,
                user_text=user_text,
                temperature=temp,
            )

            st.session_state.last_llm_raw_reply = raw_reply

            if validate_closing_response(raw_reply):
                st.session_state.last_llm_status = f"Closing ok in Versuch {attempt}"
                return " ".join(raw_reply.split())

            st.session_state.closing_validation_fail_count += 1
            st.session_state.last_llm_error = (
                f"Closing-Validierung fehlgeschlagen in Versuch {attempt}. Antwort: {raw_reply}"
            )
            st.session_state.last_llm_status = f"Closing-Validierung fehlgeschlagen in Versuch {attempt}"
            time.sleep(0.4)

        except Exception as e:
            st.session_state.last_llm_error = f"{type(e).__name__}: {e}"
            st.session_state.last_llm_status = f"Closing-Fehler in Versuch {attempt}"
            time.sleep(0.7)

    st.session_state.closing_fallback_count += 1
    st.session_state.last_llm_status = "Closing-Fallback ausgelöst"
    return closing_fallback(cond)


def get_condition_label(cond: str) -> str:
    if cond == "high":
        return "high-anthropomorph"
    return "low-anthropomorph"


def init_state():
    ensure_csv_files()

    pid = get_param("pid", "").strip()
    raw_cond = get_param("cond", "1").strip().lower()

    if raw_cond == "1":
        cond = "low"
    elif raw_cond == "2":
        cond = "high"
    elif raw_cond in {"low", "high"}:
        cond = raw_cond
    else:
        cond = "low"

    if not pid:
        pid = f"test_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"

    return_url = get_param("return_url", "")
    debug_mode = get_debug_mode()

    if debug_mode:
        max_rounds_param = get_param("rounds", str(PROD_ROUNDS))
        try:
            max_rounds_int = max(1, min(int(max_rounds_param), 10))
        except ValueError:
            max_rounds_int = PROD_ROUNDS
    else:
        max_rounds_int = PROD_ROUNDS

    defaults = {
        "phase": "intro",
        "pid": pid,
        "cond": cond,
        "raw_cond": raw_cond,
        "return_url": return_url,
        "max_rounds": max_rounds_int,
        "debug_mode": debug_mode,
        "messages": [],
        "turn": 0,
        "topic": "",
        "session_id": f"{pid}_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}",
        "session_start": now_iso(),
        "session_end": "",
        "chat_completed": False,
        "safety_triggered": False,
        "closing_logged": False,
        "pending_finish": False,
        "user_messages_count": 0,
        "validation_fail_count": 0,
        "fallback_count": 0,
        "closing_validation_fail_count": 0,
        "closing_fallback_count": 0,
        "last_llm_error": "",
        "last_llm_raw_reply": "",
        "last_llm_status": "",
        "last_prompt_excerpt": "",
    }

    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def log_message(role: str, text: str):
    with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(
            [
                st.session_state.session_id,
                st.session_state.pid,
                st.session_state.cond,
                st.session_state.turn,
                role,
                text,
                now_iso(),
            ]
        )


def write_summary_once():
    if st.session_state.session_end:
        return

    st.session_state.session_end = now_iso()

    with open(SUMMARY_FILE, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(
            [
                st.session_state.session_id,
                st.session_state.pid,
                st.session_state.cond,
                st.session_state.raw_cond,
                st.session_state.session_start,
                st.session_state.session_end,
                "yes" if st.session_state.chat_completed else "no",
                st.session_state.turn,
                st.session_state.user_messages_count,
                st.session_state.topic,
                "yes" if st.session_state.safety_triggered else "no",
                st.session_state.validation_fail_count,
                st.session_state.fallback_count,
                st.session_state.closing_validation_fail_count,
                st.session_state.closing_fallback_count,
            ]
        )


def render_debug_sidebar():
    if not st.session_state.debug_mode:
        return

    with st.sidebar:
        st.markdown("### Debug")
        st.write(
            {
                "pid": st.session_state.pid,
                "cond": st.session_state.cond,
                "raw_cond": st.session_state.raw_cond,
                "cond_label": get_condition_label(st.session_state.cond),
                "rounds": st.session_state.max_rounds,
                "turn": st.session_state.turn,
                "phase": st.session_state.phase,
                "pending_finish": st.session_state.pending_finish,
                "session_id": st.session_state.session_id,
                "model": get_model_name(),
                "validation_fail_count": st.session_state.validation_fail_count,
                "fallback_count": st.session_state.fallback_count,
                "closing_validation_fail_count": st.session_state.closing_validation_fail_count,
                "closing_fallback_count": st.session_state.closing_fallback_count,
            }
        )

        found_key_name = None
        for key_name in ["OPENAI_API_KEY", "LLM_API_KEY"]:
            try:
                candidate = st.secrets[key_name]
                if candidate and str(candidate).strip():
                    found_key_name = key_name
                    break
            except Exception:
                pass

        if found_key_name:
            api_key = st.secrets[found_key_name]
            masked = f"{str(api_key)[:7]}...{str(api_key)[-4:]}" if len(str(api_key)) >= 12 else "vorhanden"
            st.success(f"{found_key_name} gefunden: {masked}", icon="✅")
        else:
            st.error("Kein API-Key gefunden. Erwartet wird OPENAI_API_KEY oder LLM_API_KEY.", icon="🚨")

        if st.session_state.last_llm_status:
            st.info(st.session_state.last_llm_status)

        if st.session_state.last_llm_error:
            st.error("Letzter LLM-Fehler", icon="🚨")
            st.code(st.session_state.last_llm_error)

        if st.session_state.last_llm_raw_reply:
            st.write("Letzte rohe Modellantwort:")
            st.code(st.session_state.last_llm_raw_reply)

        if st.session_state.last_prompt_excerpt:
            st.write("Letzter Prompt-Ausschnitt:")
            st.code(st.session_state.last_prompt_excerpt)


# === Streamlit App Flow ===

init_state()
render_debug_sidebar()

st.title("KI-Reflexionschat")
st.caption("Technischer Prototyp für die Masterarbeit")

if st.session_state.phase == "intro":
    st.markdown(INTRO_TEXT)

    topic = st.text_area(
        "Mit welchem studienbezogenen Thema oder welcher studienbezogenen Herausforderung möchtest du dich in dieser kurzen Reflexion beschäftigen?",
        value=st.session_state.topic,
        placeholder="Zum Beispiel: Prüfungsdruck, Stress mit der Masterarbeit, Zukunftsunsicherheit im Studium, Überforderung, Motivation, Zeitmanagement …",
        height=140,
    )
    st.session_state.topic = topic

    st.markdown(
        """
**Hinweis für den Einstieg:**  
Hilfreich ist, wenn du dein Thema kurz so beschreibst, dass der Chat deine Situation nachvollziehen kann — zum Beispiel worum es geht, welche Gedanken dich dazu beschäftigen und warum das Thema im Moment relevant ist.
"""
    )

    if st.button("Reflexion starten", type="primary"):
        if not validate_topic_input(topic):
            st.warning("Bitte beschreibe dein studienbezogenes Thema etwas genauer, bevor du die Reflexion startest.")
        else:
            intro_msg = "Beschreibe, was dich an deinem Thema gerade beschäftigt."
            st.session_state.messages.append({"role": "assistant", "content": intro_msg})
            log_message("assistant", intro_msg)
            st.session_state.phase = "chat"
            st.rerun()

elif st.session_state.phase == "chat":
    st.subheader(f"Reflexion zum Thema: {st.session_state.topic}")
    st.write(f"Nachricht {st.session_state.turn + 1} von {st.session_state.max_rounds}")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    if st.session_state.pending_finish:
        st.session_state.phase = "finished"
        st.rerun()

    if st.session_state.turn >= st.session_state.max_rounds and not st.session_state.pending_finish:
        st.session_state.chat_completed = True
        st.session_state.pending_finish = True
        st.rerun()

    user_input = st.chat_input("Schreibe hier deine Antwort …")

    if user_input:
        if check_safety(user_input):
            st.session_state.safety_triggered = True
            st.session_state.messages.append({"role": "user", "content": user_input})
            log_message("user", user_input)
            st.session_state.user_messages_count += 1

            safety_msg = (
                "Dein Text enthält Hinweise auf starke Belastung oder eine mögliche Krisensituation. "
                "Dieses KI-System kann in solchen Situationen keine Hilfe leisten. "
                "Bitte wende dich jetzt an eine vertraute Person oder an professionelle Hilfe. "
                "Bei akuter Gefahr rufe bitte den Notruf 112 an. "
                "Wenn du mit jemandem sprechen möchtest, kannst du dich zum Beispiel an die TelefonSeelsorge "
                "oder an die psychologische Beratungsstelle deiner Hochschule wenden. "
                "Du kannst die Teilnahme hier beenden."
            )
            st.session_state.messages.append({"role": "assistant", "content": safety_msg})
            log_message("assistant", safety_msg)

            st.session_state.phase = "finished"
            st.rerun()

        st.session_state.messages.append({"role": "user", "content": user_input})
        log_message("user", user_input)
        st.session_state.user_messages_count += 1

        with st.chat_message("assistant"):
            with st.spinner("Antwort wird erzeugt …"):
                is_last_turn = st.session_state.turn >= st.session_state.max_rounds - 1

                if is_last_turn:
                    reply = generate_closing_reply(
                        user_text=user_input,
                        cond=st.session_state.cond,
                        topic=st.session_state.topic,
                        turn=st.session_state.turn + 1,
                        max_rounds=st.session_state.max_rounds,
                    )
                else:
                    reply = generate_llm_reply(
                        user_text=user_input,
                        cond=st.session_state.cond,
                        topic=st.session_state.topic,
                        turn=st.session_state.turn + 1,
                        max_rounds=st.session_state.max_rounds,
                    )

                st.write(reply)

        st.session_state.messages.append({"role": "assistant", "content": reply})
        log_message("assistant", reply)
        st.session_state.turn += 1

        if is_last_turn:
            st.session_state.chat_completed = True
            st.session_state.closing_logged = True
            st.session_state.pending_finish = True

        st.rerun()

elif st.session_state.phase == "finished":
    write_summary_once()
    st.success("Der Chatteil ist beendet.")
    st.write(
        "Vielen Dank für deine Teilnahme an diesem Chatteil. "
        "Im nächsten Schritt geht es im Fragebogen mit einigen Fragen zu deiner Erfahrung weiter."
    )
    st.write(
        "Bitte kehre dazu zum Fragebogen-Tab zurück. "
        "Falls unten ein Button angezeigt wird, kannst du auch darauf klicken."
    )

    if st.session_state.return_url:
        safe_url = quote(st.session_state.return_url, safe=":/?&=%#")
        st.markdown(
            f'<a href="{safe_url}" target="_self" '
            'style="text-decoration:none;color:#111111;display:inline-block;">'
            '<div style="display:inline-block;padding:0.7rem 1rem;'
            'background:#e9dfcf;color:#111111;border-radius:0.6rem;'
            'font-weight:600;border:1px solid #cbbda8;">Zurück zum Fragebogen</div></a>',
            unsafe_allow_html=True,
        )
    else:
        st.info(
            "Es wurde kein automatischer Rücksprunglink übermittelt. "
            "Bitte wechsle manuell zurück zum Fragebogen-Tab in deinem Browser und fahre dort fort."
        )

    if st.session_state.debug_mode:
        st.markdown("### Sitzungsdaten (lokale Vorschau)")
        if LOG_FILE.exists():
            df = pd.read_csv(LOG_FILE)
            session_df = df[df["session_id"] == st.session_state.session_id]
            st.dataframe(session_df, use_container_width=True)

        if st.button("Neue Testsitzung starten"):
            for key in [
                "phase",
                "messages",
                "turn",
                "session_id",
                "session_start",
                "session_end",
                "chat_completed",
                "topic",
                "safety_triggered",
                "closing_logged",
                "pending_finish",
                "user_messages_count",
                "validation_fail_count",
                "fallback_count",
                "closing_validation_fail_count",
                "closing_fallback_count",
                "last_llm_error",
                "last_llm_raw_reply",
                "last_llm_status",
                "last_prompt_excerpt",
            ]:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()
