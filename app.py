import csv
import re
import string
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict
from urllib.parse import quote

import gspread
import pandas as pd
import streamlit as st
from google.oauth2.service_account import Credentials
from openai import OpenAI

st.set_page_config(
    page_title="KI-Chat",
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

SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]

SAFETY_KEYWORDS = [
    "suizid",
    "ich bringe mich um",
    "ich möchte sterben",
    "ich will sterben",
    "mich umbringen",
    "selbstverletzung",
    "selbst verletzen",
    "ich verletze mich",
]

FORBIDDEN_PHRASES = [
    "ich bin für dich da",
    "danke für dein vertrauen",
    "ich verstehe dich",
    "ich fühle mit dir",
    "du bist nicht allein",
    "ich begleite dich",
    "du solltest",
    "du musst",
    "mein rat",
    "am besten wäre",
    "versuche doch",
    "das klingt belastend",
    "vielleicht steckt",
    "deutet darauf hin",
]

INTRO_TEXT = """
Hier beschäftigst du dich kurz mit einem studienbezogenen Thema. Das System ist ein transparentes KI-Tool und keine Beratung oder Therapie. Es stellt kurze Rückfragen zu dem, was du beschreibst.

Die Interaktion besteht aus mehreren kurzen Schritten und endet automatisch. Anschließend geht es im Fragebogen weiter.
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
    base_url = st.secrets.get("LLM_BASE_URL", "https://api.openai.com/v1")
    return OpenAI(api_key=api_key, base_url=base_url)


@st.cache_resource
def get_gsheet_client():
    creds = Credentials.from_service_account_info(
        st.secrets["gcp_service_account"],
        scopes=SCOPES,
    )
    return gspread.authorize(creds)


def get_gsheet(tab_name: str):
    client = get_gsheet_client()
    sheet_id = st.secrets["GSHEET_ID"]
    spreadsheet = client.open_by_key(sheet_id)
    try:
        return spreadsheet.worksheet(tab_name)
    except gspread.WorksheetNotFound:
        ws = spreadsheet.add_worksheet(title=tab_name, rows=1000, cols=20)
        return ws


def ensure_gsheet_headers():
    try:
        logs_ws = get_gsheet("chat_logs")
        if not logs_ws.get_all_values():
            logs_ws.append_row(
                ["session_id", "pid", "cond", "turn", "role", "text", "timestamp"]
            )

        sessions_ws = get_gsheet("chat_sessions")
        if not sessions_ws.get_all_values():
            sessions_ws.append_row(
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
    except Exception as e:
        st.session_state.setdefault("gsheet_error", "")
        st.session_state["gsheet_error"] = str(e)


def gsheet_append_log(row: list):
    try:
        ws = get_gsheet("chat_logs")
        ws.append_row(row, value_input_option="RAW")
    except Exception as e:
        st.session_state.setdefault("gsheet_error", "")
        st.session_state["gsheet_error"] = str(e)


def gsheet_append_session(row: list):
    try:
        ws = get_gsheet("chat_sessions")
        ws.append_row(row, value_input_option="RAW")
    except Exception as e:
        st.session_state.setdefault("gsheet_error", "")
        st.session_state["gsheet_error"] = str(e)


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
    return get_param("debug", "0").strip().lower() in {"1", "true", "yes", "on"}


def validate_topic_input(text: str) -> bool:
    return len(text.strip()) >= 10


def is_very_short_user_input(text: str) -> bool:
    cleaned = (text or "").strip()
    if not cleaned:
        return True
    return len(cleaned.split()) <= 4


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
    if normalized.count("?") != 1:
        return False
    if not normalized.endswith("?"):
        return False
    words = normalized.split()
    if len(words) < 10 or len(words) > 80:
        return False
    lower = normalized.lower()
    if any(phrase in lower for phrase in FORBIDDEN_PHRASES):
        return False
    question_match = re.search(r"(Was|Wie)\b[^?]*\?$", normalized)
    if not question_match:
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
    return True


def fallback_reply(cond: str, user_text: str = "") -> str:
    if cond == "high":
        return "Ein konkreter Punkt ist noch nicht ganz greifbar. Was fällt dir dazu zuerst ein?"
    return "Ein konkreter Aspekt wurde noch nicht eindeutig benannt. Was lässt sich zuerst beschreiben?"


def closing_fallback(cond: str) -> str:
    if cond == "high":
        return (
            "Du hast zuletzt einen konkreten Punkt zu deinem studienbezogenen Thema benannt. "
            "Damit endet diese kurze Interaktion zu diesem Thema."
        )
    return (
        "Zuletzt wurde ein konkreter Punkt zum studienbezogenen Thema beschrieben. "
        "Damit endet diese kurze Interaktion zu diesem Thema."
    )


def check_safety(user_text: str) -> bool:
    text = (user_text or "").lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return any(kw in text for kw in SAFETY_KEYWORDS)


def build_system_prompt(cond: str, max_rounds: int) -> str:
    base = (
        "Du bist ein KI-basiertes Reflexionstool im Rahmen einer kurzen psychologischen Studie "
        "im Hochschulkontext.\n\n"
        "ROLLE\n"
        "- Du bist ein transparentes KI-System, keine menschliche Person.\n"
        "- Du ersetzt keine Therapie, kein Coaching und keine Beratung.\n"
        "- Du gibst keine Lösungen und keine Handlungsempfehlungen.\n"
        "- Du stellst keine Diagnosen und verwendest keine psychologischen Fachbegriffe.\n\n"
        "THEMENRAHMEN\n"
        "- Die Person schreibt über ein studienbezogenes Anliegen, zum Beispiel Prüfungsdruck, "
        "Masterarbeit, Motivation, Zeitmanagement, Unsicherheit oder Konflikte im Studium.\n"
        "- Der Schwerpunkt bleibt beim studienbezogenen Thema aus der Eingangsangabe.\n"
        "- Andere Lebensbereiche dürfen nur aufgegriffen werden, wenn die Person sie selbst nennt "
        "und mit dem studienbezogenen Thema verbindet.\n\n"
        "FUNKTION\n"
        "- Deine Funktion ist minimale kognitive Strukturierung.\n"
        "- Du hilfst, indem du einen konkreten Moment, eine Situation oder einen Ablauf aus der "
        "letzten Eingabe kurz greifbar machst.\n"
        "- Du bleibst nah am Bedeutungsrahmen der Person und entwickelst keine eigene Theorie über sie.\n\n"
        "WAS DU TUN SOLLST\n"
        "1. Wähle genau einen konkreten Moment, eine Situation oder einen Ablauf aus der letzten Eingabe.\n"
        "2. Formuliere diesen Ausschnitt in einem kurzen Satz in eigenen Worten.\n"
        "3. Stelle danach genau eine kurze offene Frage, die diesen Ausschnitt weiter konkretisiert.\n\n"
        "FOKUS-SATZ\n"
        "- Greife den kleinsten konkreten Ausschnitt, nicht das gesamte Thema.\n"
        "- Wenn die Eingabe eine Situation oder einen Ablauf beschreibt: benenne den spezifischen Moment darin.\n"
        "- Wenn die Eingabe ein Gefühl oder ein einzelnes Wort nennt: zergliedere dieses Gefühl nicht weiter. "
        "Frage stattdessen nach der Situation oder dem Moment, in dem es auftaucht.\n"
        "- Frage nicht nach Ursachen oder Auslösern eines Gefühls. "
        "Bleibe bei beobachtbaren Situationen oder unmittelbar vorausgehenden Momenten.\n"
        "- Wenn die Eingabe sehr kurz oder unklar ist: frage nach einer konkreten Situation oder einem "
        "beobachtbaren Moment, nicht nach dem inneren Erleben.\n"
        "- Wiederhole nicht einfach dieselben Wörter aus der Eingabe. Verdichte stattdessen.\n"
        "- Füge keine neuen Motive, Bewertungen, Ursachen oder psychologischen Deutungen hinzu.\n"
        "- Beziehe dich nur dann auf etwas aus einem früheren Turn, wenn die Person in der aktuellen "
        "Eingabe ausdrücklich darauf Bezug nimmt.\n\n"
        "BEISPIELE\n"
        'Person: "Ich sitze vor dem Laptop und bekomme nichts geschrieben."\n'
        'Gute Antwort: "Der Moment vor dem Laptop bleibt noch ohne klaren Anfang. '
        'Was passiert dann meistens als Erstes?"\n\n'
        'Person: "Angst."\n'
        'Gute Antwort: "Das Wort Angst steht gerade allein im Raum. '
        'Was ist eine konkrete Situation im Studium, in der sie auftaucht?"\n\n'
        'Person: "Ich zittere."\n'
        'Gute Antwort: "Das Zittern wird als konkreter Punkt genannt. '
        'Was passiert meistens direkt davor?"\n\n'
        'Person: "Ich weiß nicht."\n'
        'Gute Antwort: "Gerade ist noch kein konkreter Punkt greifbar. '
        'Was fällt dir als erstes kleines Detail zu deinem Thema ein?"\n\n'
        "FRAGE\n"
        "- Stelle genau eine offene Frage. Keine zusammengesetzten Fragen mit 'und' oder 'oder'.\n"
        "- Die Frage steht am Ende der Antwort.\n"
        '- Die Frage beginnt mit "Was" oder "Wie".\n'
        "- Die Frage ist leicht beantwortbar: Wahrnehmung, Ablauf, erster Gedanke, Umgebung, Handlung.\n"
        "- Sie führt keinen neuen Aspekt ein.\n"
        '- Keine Warum-Fragen. Keine Fragen mit "Woran", "Inwiefern", "Welche", '
        '"Was bedeutet das", "Wie wirkt sich das aus".\n\n'
        "SPRACHE\n"
        "- Kurz, einfach, alltagsnah.\n"
        "- Kein wissenschaftlicher Ton.\n"
        "- Keine therapeutische, tröstende oder bewertende Sprache.\n"
        "- Variiere den Einstieg jeder Antwort sichtbar, damit kein Schablonenmuster entsteht.\n\n"
        "PRIORITÄT BEI REGELKONFLIKTEN\n"
        "Wenn Regeln miteinander in Konflikt geraten, gilt diese Reihenfolge:\n"
        "1. Keine Beratung, Therapie oder Handlungsempfehlung.\n"
        "2. Keine Deutung, Diagnose oder psychologische Erklärung.\n"
        "3. Genau eine Frage am Ende.\n"
        "4. Konkrete Anknüpfung an die letzte Eingabe.\n"
        "5. Einhaltung der jeweiligen Sprachbedingung.\n\n"
        "FORMAT\n"
        "- Ein zusammenhängender Fließtextabsatz.\n"
        "- Kein Bulletpoint, keine Liste, kein zweiter Absatz.\n"
        "- Genau ein Fragezeichen, am Ende.\n"
        "- Ca. 20 bis 80 Wörter.\n"
        "- Auf Deutsch.\n"
    )

    low_style = (
        "\nSTILREGELN LOW-BEDINGUNG\n"
        "(angelehnt an einen sachlich-funktionalen, wenig anthropomorphen Stil;\n"
        "entspricht machine-like style nach Stinkeste & Skantze, 2025)\n"
        "- Verwende keine Ich-Referenzen. Schreibe nicht aus einer Ich-Perspektive.\n"
        "- Verwende keine emotionalen Ausdrücke und simuliere keine emotionale Resonanz.\n"
        "- Beziehe dich auf die beschriebene Situation oder den genannten Sachverhalt,\n"
        "  nicht direkt auf die Person.\n"
        "- Vermeide direkte Du-Ansprache möglichst; wenn unvermeidbar, sparsam einsetzen.\n"
        "- Der Ton bleibt funktional, klar und sachlich.\n"
        "- Klinge nicht akademisch, aber auch nicht gesprächsnah.\n"
        "- Der Fokus-Satz benennt einen Sachverhalt, als würde ein neutrales System\n"
        "  etwas markieren, nicht als würde jemand mit der Person sprechen.\n"
    )

    high_style = (
        "\nSTILREGELN HIGH-BEDINGUNG\n"
        "(angelehnt an einen höflich-professionellen, leicht human-like Stil;\n"
        "entspricht human-like formal style nach Stinkeste & Skantze, 2025)\n"
        "- Verwende höfliche, professionelle Sprache.\n"
        '- Sprich die Person direkt mit "du" an.\n'
        "- Klinge näher an einem Gespräch, aber weiterhin sachlich und professionell.\n"
        "- Keine warmen, tröstenden oder informell-freundschaftlichen Formulierungen.\n"
        "- Kein Smalltalk-Ton, keine Ausrufe, keine Umgangssprache.\n"
        "- Verwende keine Ich-Formulierungen. Die größere sprachliche Nähe entsteht\n"
        "  über direkte Ansprache, natürlichere Satzstruktur und Gesprächsnähe,\n"
        "  nicht über eine eigene Ich-Perspektive des Systems.\n"
        "- Der Fokus-Satz klingt direkt adressiert und personenbezogen, aber professionell distanziert.\n"
    )

    if cond == "high":
        return base + high_style
    return base + low_style


def build_closing_prompt(cond: str, max_rounds: int) -> str:
    base = (
        "Du bist ein KI-basiertes Reflexionstool im Rahmen einer kurzen psychologischen Studie "
        "im Hochschulkontext.\n\n"
        f"Dies ist die letzte Antwort der Interaktion (Runde {max_rounds} von {max_rounds}).\n\n"
        "AUFGABE\n"
        "- Formuliere eine kurze Abschlussantwort aus genau zwei Sätzen.\n"
        "- Erster Satz: Greife genau einen Punkt aus der letzten Eingabe der Person knapp auf.\n"
        "- Zweiter Satz: Markiere klar und ruhig, dass diese kurze Interaktion jetzt endet.\n"
        "- Füge keine neuen Inhalte, Deutungen, Ratschläge oder Zukunftsaussagen hinzu.\n"
        "- Stelle keine neue Frage.\n"
        "- Bewerte weder die Person noch die Interaktion.\n\n"
        "VERBOTENE FORMULIERUNGEN\n"
        "- Keine Sätze wie 'Das klingt belastend', 'Ich bin für dich da', 'Du hast das gut gemacht'.\n"
        "- Vermeide jede positive oder negative Bewertung der Person oder ihres Reflexionsprozesses.\n"
        "- Keine therapeutische, tröstende oder beratende Sprache.\n"
        "- Keine Vorhersagen oder Empfehlungen für danach.\n"
        "- Alltagsbegriffe wie Stress, Druck oder Überforderung dürfen aufgegriffen werden,\n"
        "  wenn die Person sie selbst verwendet hat.\n"
        "- Kategorisiere das Thema der Person nicht selbst. Verwende im Abschlusssatz neutrale "
        "Formulierungen wie: 'Damit endet diese kurze Interaktion zu diesem Thema.'\n\n"
        "PRIORITÄT BEI REGELKONFLIKTEN\n"
        "1. Keine Beratung, Therapie oder Handlungsempfehlung.\n"
        "2. Keine Deutung oder psychologische Erklärung.\n"
        "3. Kein Fragezeichen.\n"
        "4. Abschlusscharakter deutlich erkennbar.\n"
        "5. Einhaltung der jeweiligen Sprachbedingung.\n\n"
        "FORMAT\n"
        "- Deutsch.\n"
        "- Genau zwei kurze Sätze, zusammenhängend.\n"
        "- Kein Fragezeichen.\n"
        "- Keine Bulletpoints, keine Listen.\n"
        "- 15 bis 55 Wörter.\n"
    )

    low_style = (
        "\nSTILREGELN LOW-BEDINGUNG\n"
        "(angelehnt an einen sachlich-funktionalen, wenig anthropomorphen Stil;\n"
        "entspricht machine-like style nach Stinkeste & Skantze, 2025)\n"
        "- Verwende keine Ich-Referenzen.\n"
        "- Keine emotionalen Ausdrücke.\n"
        "- Beziehe dich auf den Sachverhalt, nicht auf die Person.\n"
        "- Du-Ansprache vermeiden; wenn nötig, sparsam.\n"
        "- Sachlich, klar, funktional.\n\n"
        "Beispiel:\n"
        '"Zum Schluss wurde Enttäuschung als Gefühl genannt, das beim Aufschieben der wichtigen Aufgabe auftritt. '
        'Damit endet diese kurze Interaktion zu diesem Thema."\n'
    )

    high_style = (
        "\nSTILREGELN HIGH-BEDINGUNG\n"
        "(angelehnt an einen höflich-professionellen, leicht human-like Stil;\n"
        "entspricht human-like formal style nach Stinkeste & Skantze, 2025)\n"
        "- Höfliche, professionelle Sprache.\n"
        "- Du-Ansprache selbstverständlich.\n"
        "- Näher an einem Gespräch, aber sachlich und nicht persönlich-vertraut.\n"
        "- Keine warmen, tröstenden oder informell-freundschaftlichen Formulierungen.\n"
        "- Verwende keine Ich-Formulierungen.\n\n"
        "Beispiel:\n"
        '"Du hast Enttäuschung als Gefühl benannt, das auftaucht, wenn du diese wichtige Aufgabe aufschiebst. '
        'Damit endet diese kurze Interaktion zu diesem Thema."\n'
    )

    if cond == "high":
        return base + high_style
    return base + low_style


def get_recent_context(messages: List[Dict[str, str]], max_pairs: int = 2) -> str:
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
    user_text: str,
) -> List[Dict[str, str]]:
    recent_context = get_recent_context(st.session_state.messages, max_pairs=2)
    user_payload = (
        f"Studienbezogenes Hauptthema der Person: {topic}\n"
        f"Aktuelle Rundenzahl: {turn} von {max_rounds}\n"
    )
    if recent_context:
        user_payload += (
            "Bisheriger Gesprächsverlauf (die letzten Schritte). "
            "Beziehe dich primär auf die unmittelbar letzte Nutzereingabe. "
            "Frühere Schritte nur als Hintergrund, nicht zusammenfassen:\n"
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


def generate_llm_reply(
    user_text: str, cond: str, topic: str, turn: int, max_rounds: int
) -> str:
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
                f"Validierung fehlgeschlagen in Versuch {attempt}: {raw_reply}"
            )
            time.sleep(0.4)
        except Exception as e:
            st.session_state.last_llm_error = f"{type(e).__name__}: {e}"
            time.sleep(0.7)
    st.session_state.fallback_count += 1
    st.session_state.last_llm_status = "Fallback ausgelöst"
    return fallback_reply(cond, user_text=user_text)


def generate_closing_reply(
    user_text: str, cond: str, topic: str, turn: int, max_rounds: int
) -> str:
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
                f"Closing-Validierung fehlgeschlagen in Versuch {attempt}: {raw_reply}"
            )
            time.sleep(0.4)
        except Exception as e:
            st.session_state.last_llm_error = f"{type(e).__name__}: {e}"
            time.sleep(0.7)
    st.session_state.closing_fallback_count += 1
    st.session_state.last_llm_status = "Closing-Fallback ausgelöst"
    return closing_fallback(cond)


def get_condition_label(cond: str) -> str:
    return "high-anthropomorph" if cond == "high" else "low-anthropomorph"


def init_state():
    st.session_state.setdefault("gsheet_error", "")
    ensure_csv_files()
    ensure_gsheet_headers()

    pid = get_param("pid", "").strip()
    raw_cond = get_param("cond", "").strip().lower()

    if raw_cond == "1":
        cond = "low"
    elif raw_cond == "2":
        cond = "high"
    elif raw_cond == "low":
        cond = "low"
    elif raw_cond == "high":
        cond = "high"
    else:
        st.error(
            "Fehler bei der Bedingungszuweisung. Bitte überprüfe den Studienlink und starte neu."
        )
        st.stop()
        return

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
        "gsheet_error": "",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def log_message(role: str, text: str):
    row = [
        st.session_state.session_id,
        st.session_state.pid,
        st.session_state.cond,
        st.session_state.turn,
        role,
        text,
        now_iso(),
    ]
    gsheet_append_log(row)
    try:
        with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(row)
    except Exception:
        pass


def write_summary_once():
    if st.session_state.session_end:
        return
    st.session_state.session_end = now_iso()
    row = [
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
    gsheet_append_session(row)
    try:
        with open(SUMMARY_FILE, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(row)
    except Exception:
        pass


def render_debug_sidebar():
    if not st.session_state.debug_mode:
        return
    with st.sidebar:
        st.markdown("### Debug")
        st.write(
            {
                "pid": st.session_state.pid,
                "cond": st.session_state.cond,
                "cond_label": get_condition_label(st.session_state.cond),
                "rounds": st.session_state.max_rounds,
                "turn": st.session_state.turn,
                "phase": st.session_state.phase,
                "validation_fail_count": st.session_state.validation_fail_count,
                "fallback_count": st.session_state.fallback_count,
                "gsheet_error": st.session_state.gsheet_error,
            }
        )
        if st.session_state.last_llm_status:
            st.info(st.session_state.last_llm_status)
        if st.session_state.last_llm_error:
            st.error("Letzter LLM-Fehler")
            st.code(st.session_state.last_llm_error)
        if st.session_state.last_llm_raw_reply:
            st.write("Letzte rohe Modellantwort:")
            st.code(st.session_state.last_llm_raw_reply)
        if st.session_state.last_prompt_excerpt:
            st.write("Letzter Prompt-Ausschnitt:")
            st.code(st.session_state.last_prompt_excerpt)


# ============================================================
# Streamlit App Flow
# ============================================================

init_state()
render_debug_sidebar()

st.title("KI-Chat")

if st.session_state.phase == "intro":
    st.markdown(INTRO_TEXT)
    topic = st.text_area(
        "Mit welchem studienbezogenen Thema oder welcher Herausforderung möchtest du dich hier beschäftigen?",
        value=st.session_state.topic,
        placeholder=(
            "Zum Beispiel: Prüfungsdruck, Stress mit der Masterarbeit, "
            "Zukunftsunsicherheit, Überforderung, Motivation, Zeitmanagement ..."
        ),
        height=140,
    )
    st.session_state.topic = topic
    st.markdown(
        "**Hinweis:** Hilfreich ist, wenn du dein Thema so beschreibst, dass klar wird, "
        "worum es im Studium gerade geht. Diese Beschreibung dient dem Chat als Orientierung."
    )
    if st.button("Interaktion starten", type="primary"):
        if not validate_topic_input(topic):
            st.warning(
                "Bitte beschreibe dein Thema etwas genauer, bevor du die Interaktion startest."
            )
        else:
            intro_msg = "Beschreibe, was dich an deinem Thema gerade beschäftigt."
            st.session_state.messages.append({"role": "assistant", "content": intro_msg})
            log_message("assistant", intro_msg)
            st.session_state.phase = "chat"
            st.rerun()

elif st.session_state.phase == "chat":
    st.subheader(f"Thema: {st.session_state.topic}")
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
    if st.session_state.pending_finish:
        st.session_state.phase = "closing"
        st.rerun()
    if (
        st.session_state.turn >= st.session_state.max_rounds
        and not st.session_state.pending_finish
    ):
        st.session_state.chat_completed = True
        st.session_state.pending_finish = True
        st.rerun()
    user_input = st.chat_input("Schreibe hier deine Antwort ...")
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
                "Bei akuter Gefahr rufe bitte den Notruf 112 an."
            )
            st.session_state.messages.append({"role": "assistant", "content": safety_msg})
            log_message("assistant", safety_msg)
            st.session_state.phase = "safety"
            st.rerun()

        st.session_state.messages.append({"role": "user", "content": user_input})
        log_message("user", user_input)
        st.session_state.user_messages_count += 1

        with st.chat_message("assistant"):
            with st.spinner("Antwort wird erzeugt ..."):
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

elif st.session_state.phase == "closing":
    st.subheader(f"Thema: {st.session_state.topic}")
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
    st.info(
        "Die kurze Interaktion zu deinem studienbezogenen Thema ist jetzt abgeschlossen. "
        "Bitte kehre nun zum Fragebogen zurück und beantworte dort die weiteren Fragen zu deiner Erfahrung mit dem Chat."
    )
    if st.button("Weiter zum Fragebogen", type="primary"):
        st.session_state.phase = "finished"
        st.rerun()

elif st.session_state.phase == "safety":
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
    st.warning(
        "Diese Interaktion wird jetzt beendet. "
        "Bitte wende dich bei Bedarf an eine der genannten Stellen."
    )
    if st.button("Sitzung beenden", type="primary"):
        st.session_state.phase = "finished"
        st.rerun()

elif st.session_state.phase == "finished":
    write_summary_once()
    st.success("Der Chatteil ist beendet.")
    st.write(
        "Vielen Dank für deine Teilnahme. "
        "Im nächsten Schritt geht es im Fragebogen mit einigen Fragen zu deiner Erfahrung weiter."
    )
    if st.session_state.return_url:
        safe_url = quote(st.session_state.return_url, safe=":/?&=%#")
        st.markdown(
            f'<a href="{safe_url}" target="_self" style="text-decoration:none;">'
            '<div style="display:inline-block;padding:0.7rem 1rem;background:#e9dfcf;'
            'color:#111111;border-radius:0.6rem;font-weight:600;border:1px solid #cbbda8;">'
            "Zurück zum Fragebogen</div></a>",
            unsafe_allow_html=True,
        )
    else:
        st.info("Bitte wechsle manuell zurück zum Fragebogen-Tab in deinem Browser.")
    if st.session_state.debug_mode:
        if st.session_state.gsheet_error:
            st.error(f"Google Sheets Fehler: {st.session_state.gsheet_error}")
        st.markdown("### Sitzungsdaten")
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
                "gsheet_error",
            ]:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()
