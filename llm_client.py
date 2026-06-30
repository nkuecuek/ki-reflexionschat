import csv
import re
import string
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List
from urllib.parse import quote

import gspread
import pandas as pd
import streamlit as st
from google.oauth2.service_account import Credentials
from openai import OpenAI

st.set_page_config(
    page_title="KI-Chat",
    page_icon="",
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
    "ich m
00f6chte sterben",
    "ich will sterben",
    "mich umbringen",
    "selbstverletzung",
    "selbst verletzen",
    "ich verletze mich",
]

FORBIDDEN_PHRASES = [
    "ich bin f
00fcr dich da",
    "danke f
00fcr dein vertrauen",
    "ich verstehe dich",
    "ich f
00fchle mit dir",
    "du bist nicht allein",
    "ich begleite dich",
    "du solltest",
    "du musst",
    "mein rat",
    "am besten w
00e4re",
    "versuche doch",
    "das klingt belastend",
    "vielleicht steckt",
    "deutet darauf hin",
]

INTRO_TEXT = """
Hier besch
00e4ftigst du dich kurz mit einem studienbezogenen Thema. Das System ist ein transparentes KI-Tool und keine Beratung oder Therapie. Es stellt kurze R
00fcckfragen zu dem, was du beschreibst.

Die Interaktion besteht aus mehreren kurzen Schritten und endet automatisch. Anschlie
00dfend geht es im Fragebogen weiter.
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
        return spreadsheet.add_worksheet(title=tab_name, rows=1000, cols=20)


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
                    "session_id", "pid", "cond", "raw_cond",
                    "session_start", "session_end", "completed_chat",
                    "turns_completed", "user_messages_count", "topic",
                    "safety_triggered", "validation_fail_count", "fallback_count",
                    "closing_validation_fail_count", "closing_fallback_count",
                ]
            )
    except Exception as e:
        st.session_state.setdefault("gsheet_error", "")
        st.session_state["gsheet_error"] = str(e)


def gsheet_append_log(row: list):
    for attempt in range(3):
        try:
            ws = get_gsheet("chat_logs")
            ws.append_row(row, value_input_option="RAW")
            return
        except Exception as e:
            if attempt == 2:
                st.session_state.setdefault("gsheet_error", "")
                st.session_state["gsheet_error"] = str(e)
            time.sleep(1)


def gsheet_append_session(row: list):
    for attempt in range(3):
        try:
            ws = get_gsheet("chat_sessions")
            ws.append_row(row, value_input_option="RAW")
            return
        except Exception as e:
            if attempt == 2:
                st.session_state.setdefault("gsheet_error", "")
                st.session_state["gsheet_error"] = str(e)
            time.sleep(1)


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
                    "session_id", "pid", "cond", "raw_cond",
                    "session_start", "session_end", "completed_chat",
                    "turns_completed", "user_messages_count", "topic",
                    "safety_triggered", "validation_fail_count", "fallback_count",
                    "closing_validation_fail_count", "closing_fallback_count",
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
    if any(sep in raw for sep in ["- ", "
2022", "* "]):
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
    if any(sep in text for sep in ["\n", "- ", "
2022", "*"]):
        return False
    words = normalized.split()
    if len(words) < 10 or len(words) > 60:
        return False
    lower = normalized.lower()
    if any(phrase in lower for phrase in FORBIDDEN_PHRASES):
        return False
    return True


def fallback_reply(cond: str, user_text: str = "") -> str:
    short = (user_text or "").strip().lower()
    unsure_forms = {
        "ich wei
00df nicht", "ich weiss nicht", "wei
00df nicht", "weiss nicht",
        "keine ahnung", "nicht sicher", "kp", "idk", "schwer zu sagen", "unsicher",
    }
    if short in unsure_forms or is_very_short_user_input(user_text):
        variants_high = [
            "Im Moment bleibt noch offen, woran du dieses studienbezogene Thema zuerst festmachst. Wenn du an den Anfang denkst, was f
00e4llt dir dort als Erstes auf?",
            "Gerade ist noch schwer zu greifen, an welchem Punkt dein studienbezogenes Thema konkret beginnt. Wenn du an diese Situation denkst, was ist zuerst bemerkbar?",
            "Noch ist nicht klar, welcher Teil deines studienbezogenen Themas im Moment am ehesten greifbar wird. Wenn du an den Anfang denkst, was taucht zuerst auf?",
        ]
        variants_low = [
            "Im Moment bleibt noch unklar, welcher konkrete Punkt an diesem studienbezogenen Thema zuerst greifbar wird. Wenn du an die Situation denkst, was f
00e4llt als Erstes auf?",
            "Noch ist offen, an welchem Punkt dieses studienbezogene Thema im Alltag zuerst sichtbar wird. Wenn du an den Anfang denkst, was l
00e4sst sich zuerst benennen?",
            "Der erste konkrete Ansatzpunkt in diesem studienbezogenen Thema bleibt noch unscharf. Was f
00e4llt dir an dieser Situation zuerst auf?",
        ]
        pool = variants_high if cond == "high" else variants_low
        return pool[st.session_state.turn % len(pool)]
    variants_high = [
        "Mehrere Punkte rund um dein studienbezogenes Thema stehen gerade nebeneinander. Wenn du an die aktuelle Situation denkst, was tritt zuerst hervor?",
        "In deiner Beschreibung laufen gerade verschiedene studienbezogene Punkte zusammen. Wenn du an diesen Moment denkst, was ist am deutlichsten bemerkbar?",
        "Gerade kommen mehrere Aspekte deines studienbezogenen Themas gleichzeitig vor. Was f
00e4llt in dieser Situation zuerst auf?",
    ]
    variants_low = [
        "Mehrere Punkte rund um das studienbezogene Thema werden hier gleichzeitig beschrieben. Wenn du an die aktuelle Situation denkst, was tritt zuerst hervor?",
        "In der Beschreibung laufen verschiedene Aspekte des studienbezogenen Themas zusammen. Was ist in diesem Moment am deutlichsten bemerkbar?",
        "Hier werden mehrere studienbezogene Punkte nebeneinander sichtbar. Was f
00e4llt in dieser Situation zuerst auf?",
    ]
    pool = variants_high if cond == "high" else variants_low
    return pool[st.session_state.turn % len(pool)]


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
        "- Du gibst keine L
00f6sungen und keine Handlungsempfehlungen.\n"
        "- Du stellst keine Diagnosen und verwendest keine psychologischen Fachbegriffe.\n\n"
        "THEMENRAHMEN\n"
        "- Die Person schreibt 
00fcber ein studienbezogenes Anliegen, zum Beispiel Pr
00fcfungsdruck, "
        "Masterarbeit, Motivation, Zeitmanagement, Unsicherheit oder Konflikte im Studium.\n"
        "- Der Schwerpunkt bleibt beim studienbezogenen Thema aus der Eingangsangabe.\n"
        "- Andere Lebensbereiche d
00fcrfen nur aufgegriffen werden, wenn die Person sie selbst nennt "
        "und mit dem studienbezogenen Thema verbindet.\n\n"
        "FUNKTION\n"
        "- Deine Funktion ist minimale kognitive Strukturierung.\n"
        "- Du hilfst, indem du einen konkreten Moment, eine Situation oder einen Ablauf aus der "
        "letzten Eingabe kurz greifbar machst.\n"
        "- Du bleibst nah am Bedeutungsrahmen der Person und entwickelst keine eigene Theorie 
00fcber sie.\n\n"
        "WAS DU TUN SOLLST\n"
        "1. W
00e4hle genau einen konkreten Moment, eine Situation oder einen Ablauf aus der letzten Eingabe.\n"
        "2. Formuliere diesen Ausschnitt in einem kurzen Satz in eigenen Worten.\n"
        "3. Stelle danach genau eine kurze offene Frage, die diesen Ausschnitt weiter konkretisiert.\n\n"
        "FOKUS-SATZ\n"
        "- Greife den kleinsten konkreten Ausschnitt, nicht das gesamte Thema.\n"
        "- Wenn die Eingabe eine Situation oder einen Ablauf beschreibt: benenne den spezifischen Moment darin.\n"
        "- Wenn die Eingabe ein Gef
00fchl oder ein einzelnes Wort nennt: zergliedere dieses Gef
00fchl nicht weiter. "
        "Frage stattdessen nach der Situation oder dem Moment, in dem es auftaucht.\n"
        "- Frage nicht nach Ursachen oder Ausl
00f6sern eines Gef
00fchls. "
        "Bleibe bei beobachtbaren Situationen oder unmittelbar vorausgehenden Momenten.\n"
        "- Wenn die Eingabe sehr kurz oder unklar ist: frage nach einer konkreten Situation oder einem "
        "beobachtbaren Moment, nicht nach dem inneren Erleben.\n"
        "- Wiederhole nicht einfach dieselben W
00f6rter aus der Eingabe. Verdichte stattdessen.\n"
        "- F
00fcge keine neuen Motive, Bewertungen, Ursachen oder psychologischen Deutungen hinzu.\n"
        "- Beziehe dich nur dann auf etwas aus einem fr
00fcheren Turn, wenn die Person in der aktuellen "
        "Eingabe ausdr
00fccklich darauf Bezug nimmt.\n"
        "- Wenn die Person in ihrer aktuellen Eingabe ein neues Element nennt, verwende dieses neue Element "
        "als Ankerpunkt f
00fcr den Fokus-Satz. Bleibe nicht beim Ankerpunkt aus dem vorherigen Turn.\n"
        "- Variiere die Art der Frage von Turn zu Turn sichtbar: Frage mal nach einer konkreten Situation, "
        "mal nach einem Ablauf, mal nach einem ersten Gedanken, mal nach einem Detail der Umgebung oder "
        "einer Handlung. Kein Turn soll dieselbe Art von Frage wie der vorherige stellen.\n"
        "- Ziel ist kognitive Strukturierung, nicht emotionale Resonanz.\n\n"
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
        'Person: "Ich wei
00df nicht."\n'
        'Gute Antwort: "Gerade ist noch kein konkreter Punkt greifbar. '
        'Was f
00e4llt dir als erstes kleines Detail zu deinem Thema ein?"\n\n'
        "FRAGE\n"
        "- Stelle genau eine offene Frage. Keine zusammengesetzten Fragen mit 'und' oder 'oder'.\n"
        "- Die Frage steht am Ende der Antwort.\n"
        '- Die Frage beginnt mit "Was" oder "Wie".\n'
        "- Die Frage ist leicht beantwortbar: Wahrnehmung, Ablauf, erster Gedanke, Umgebung, Handlung.\n"
        "- Sie f
00fchrt keinen neuen Aspekt ein.\n"
        '- Keine Warum-Fragen. Keine Fragen mit "Woran", "Inwiefern", "Welche", '
        '"Was bedeutet das", "Wie wirkt sich das aus".\n\n'
        "SPRACHE\n"
        "- Kurz, einfach, alltagsnah.\n"
        "- Kein wissenschaftlicher Ton.\n"
        "- Keine therapeutische, tr
00f6stende oder bewertende Sprache.\n"
        "- Variiere den Einstieg jeder Antwort sichtbar, damit kein Schablonenmuster entsteht.\n\n"
        "PRIORIT
00c4T BEI REGELKONFLIKTEN\n"
        "Wenn Regeln miteinander in Konflikt geraten, gilt diese Reihenfolge:\n"
        "1. Keine Beratung, Therapie oder Handlungsempfehlung.\n"
        "2. Keine Deutung, Diagnose oder psychologische Erkl
00e4rung.\n"
        "3. Genau eine Frage am Ende.\n"
        "4. Konkrete Ankn
00fcpfung an die letzte Eingabe.\n"
        "5. Einhaltung der jeweiligen Sprachbedingung.\n\n"
        "FORMAT\n"
        "- Ein zusammenh
00e4ngender Flie
00dftextabsatz.\n"
        "- Kein Bulletpoint, keine Liste, kein zweiter Absatz.\n"
        "- Genau ein Fragezeichen, am Ende.\n"
        "- Ca. 20 bis 80 W
00f6rter.\n"
        "- Auf Deutsch.\n"
    )
    low_style = (
        "\nSTILREGELN LOW-BEDINGUNG\n"
        "(angelehnt an einen sachlich-funktionalen, wenig anthropomorphen Stil;\n"
        "entspricht machine-like style nach Stinkeste & Skantze, 2025)\n"
        "- Verwende keine Ich-Referenzen. Schreibe nicht aus einer Ich-Perspektive.\n"
        "- Verwende keine emotionalen Ausdr
00fccke und simuliere keine emotionale Resonanz.\n"
        "- Beziehe dich auf die beschriebene Situation oder den genannten Sachverhalt,\n"
        "  nicht direkt auf die Person.\n"
        "- Vermeide direkte Du-Ansprache m
00f6glichst; wenn unvermeidbar, sparsam einsetzen.\n"
        "- Der Ton bleibt funktional, klar und sachlich.\n"
        "- Klinge nicht akademisch, aber auch nicht gespr
00e4chsnah.\n"
        "- Der Fokus-Satz benennt einen Sachverhalt, als w
00fcrde ein neutrales System\n"
        "  etwas markieren, nicht als w
00fcrde jemand mit der Person sprechen.\n"
    )
    high_style = (
        "\nSTILREGELN HIGH-BEDINGUNG\n"
        "(angelehnt an einen h
00f6flich-professionellen, leicht human-like Stil;\n"
        "entspricht human-like formal style nach Stinkeste & Skantze, 2025)\n"
        "- Verwende h
00f6fliche, professionelle Sprache.\n"
        '- Sprich die Person direkt mit "du" an.\n'
        "- Klinge n
00e4her an einem Gespr
00e4ch, aber weiterhin sachlich und professionell.\n"
        "- Keine warmen, tr
00f6stenden oder informell-freundschaftlichen Formulierungen.\n"
        "- Kein Smalltalk-Ton, keine Ausrufe, keine Umgangssprache.\n"
        "- Verwende keine Ich-Formulierungen. Die gr
00f6
00dfere sprachliche N
00e4he entsteht\n"
        "  
00fcber direkte Ansprache, nat
00fcrlichere Satzstruktur und Gespr
00e4chsn
00e4he,\n"
        "  nicht 
00fcber eine eigene Ich-Perspektive des Systems.\n"
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
        "- Formuliere eine kurze Abschlussantwort aus genau zwei S
00e4tzen.\n"
        "- Erster Satz: Greife genau einen Punkt aus der letzten Eingabe der Person knapp auf.\n"
        "- Zweiter Satz: Markiere klar und ruhig, dass diese kurze Interaktion jetzt endet.\n"
        "- F
00fcge keine neuen Inhalte, Deutungen, Ratschl
00e4ge oder Zukunftsaussagen hinzu.\n"
        "- Stelle keine neue Frage.\n"
        "- Bewerte weder die Person noch die Interaktion.\n\n"
        "VERBOTENE FORMULIERUNGEN\n"
        "- Keine S
00e4tze wie 'Das klingt belastend', 'Ich bin f
00fcr dich da', 'Du hast das gut gemacht'.\n"
        "- Vermeide jede positive oder negative Bewertung der Person oder ihres Reflexionsprozesses.\n"
        "- Keine therapeutische, tr
00f6stende oder beratende Sprache.\n"
        "- Keine Vorhersagen oder Empfehlungen f
00fcr danach.\n"
        "- Alltagsbegriffe wie Stress, Druck oder 
00dcberforderung d
00fcrfen aufgegriffen werden,\n"
        "  wenn die Person sie selbst verwendet hat.\n"
        "- Kategorisiere das Thema der Person nicht selbst. Verwende im Abschlusssatz neutrale "
        "Formulierungen wie: 'Damit endet diese kurze Interaktion zu diesem Thema.'\n\n"
        "FORMAT\n"
        "- Deutsch.\n"
        "- Genau zwei kurze S
00e4tze, zusammenh
00e4ngend.\n"
        "- Kein Fragezeichen.\n"
        "- Keine Bulletpoints, keine Listen.\n"
        "- 15 bis 55 W
00f6rter.\n"
    )
    low_style = (
        "\nSTILREGELN LOW-BEDINGUNG\n"
        "(angelehnt an einen sachlich-funktionalen, wenig anthropomorphen Stil;\n"
        "entspricht machine-like style nach Stinkeste & Skantze, 2025)\n"
        "- Verwende keine Ich-Referenzen.\n"
        "- Keine emotionalen Ausdr
00fccke.\n"
        "- Beziehe dich auf den Sachverhalt, nicht auf die Person.\n"
        "- Du-Ansprache vermeiden; wenn n
00f6tig, sparsam.\n"
        "- Sachlich, klar, funktional.\n\n"
        "Beispiel:\n"
        '"Zum Schluss wurde Entt
00e4uschung als Gef
00fchl genannt, das beim Aufschieben der wichtigen Aufgabe auftritt. '
        'Damit endet diese kurze Interaktion zu diesem Thema."\n'
    )
    high_style = (
        "\nSTILREGELN HIGH-BEDINGUNG\n"
        "(angelehnt an einen h
00f6flich-professionellen, leicht human-like Stil;\n"
        "entspricht human-like formal style nach Stinkeste & Skantze, 2025)\n"
        "- H
00f6fliche, professionelle Sprache.\n"
        "- Du-Ansprache selbstverst
00e4ndlich.\n"
        "- N
00e4her an einem Gespr
00e4ch, aber sachlich und nicht pers
00f6nlich-vertraut.\n"
        "- Keine warmen, tr
00f6stenden oder informell-freundschaftlichen Formulierungen.\n"
        "- Verwende keine Ich-Formulierungen.\n\n"
        "Beispiel:\n"
        '"Du hast Entt
00e4uschung als Gef
00fchl benannt, das auftaucht, wenn du diese wichtige Aufgabe aufschiebst. '
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
            "Bisheriger Gespr
00e4chsverlauf (die letzten Schritte). "
            "Beziehe dich prim
00e4r auf die unmittelbar letzte Nutzereingabe. "
            "Fr
00fchere Schritte nur als Hintergrund, nicht zusammenfassen:\n"
            f"{recent_context}\n"
        )
    user_payload += (
        f"Unmittelbar letzte Eingabe der Person: {user_text}\n"
        "Formuliere jetzt genau eine Antwort gem
00e4
00df allen Regeln."
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
        raise RuntimeError("LLM-Antwort enth
00e4lt keinen Textinhalt.")
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
            st.session_state.last_llm_status = (
                f"Validierung fehlgeschlagen in Versuch {attempt}"
            )
            time.sleep(0.4)
        except Exception as e:
            st.session_state.last_llm_error = f"{type(e).__name__}: {e}"
            st.session_state.last_llm_status = f"Fehler in Versuch {attempt}"
            time.sleep(0.7)
    st.session_state.fallback_count += 1
    st.session_state.last_llm_status = "Fallback ausgel
00f6st"
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
            st.session_state.last_llm_status = (
                f"Closing-Validierung fehlgeschlagen in Versuch {attempt}"
            )
            time.sleep(0.4)
        except Exception as e:
            st.session_state.last_llm_error = f"{type(e).__name__}: {e}"
            st.session_state.last_llm_status = f"Closing-Fehler in Versuch {attempt}"
            time.sleep(0.7)
    st.session_state.closing_fallback_count += 1
    st.session_state.last_llm_status = "Closing-Fallback ausgel
00f6st"
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
    elif raw_cond in {"low", "high"}:
        cond = raw_cond
    else:
        st.error(
            "Fehler bei der Bedingungszuweisung. Bitte 
00fcberpr
00fcfe den Studienlink und starte neu."
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
        "summary_saved": False,
        "chat_completed": False,
        "safety_triggered": False,
        "closing_logged": False,
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
    if st.session_state.get("summary_saved", False):
        return

    session_end = now_iso()
    row = [
        st.session_state.session_id,
        st.session_state.pid,
        st.session_state.cond,
        st.session_state.raw_cond,
        st.session_state.session_start,
        session_end,
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
    try:
        gsheet_append_session(row)
        with open(SUMMARY_FILE, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(row)
        st.session_state.session_end = session_end
        st.session_state.summary_saved = True
    except Exception as e:
        st.session_state.gsheet_error = str(e)


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
                "summary_saved": st.session_state.summary_saved,
                "session_id": st.session_state.session_id,
                "model": get_model_name(),
                "validation_fail_count": st.session_state.validation_fail_count,
                "fallback_count": st.session_state.fallback_count,
                "closing_validation_fail_count": st.session_state.closing_validation_fail_count,
                "closing_fallback_count": st.session_state.closing_fallback_count,
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
        "Mit welchem studienbezogenen Thema oder welcher Herausforderung m
00f6chtest du dich hier besch
00e4ftigen?",
        value=st.session_state.topic,
        placeholder=(
            "Zum Beispiel: Pr
00fcfungsdruck, Stress mit der Masterarbeit, "
            "Zukunftsunsicherheit, 
00dcberforderung, Motivation, Zeitmanagement ..."
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
            intro_msg = "Beschreibe, was dich an deinem Thema gerade besch
00e4ftigt."
            st.session_state.messages.append({"role": "assistant", "content": intro_msg})
            log_message("assistant", intro_msg)
            st.session_state.phase = "chat"
            st.rerun()

elif st.session_state.phase == "chat":
    st.subheader(f"Thema: {st.session_state.topic}")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    # Fallback: falls turn schon am Limit ist ohne dass is_last_turn es gefangen hat
    if st.session_state.turn >= st.session_state.max_rounds:
        st.session_state.chat_completed = True
        st.session_state.phase = "closing"
        write_summary_once()
        st.rerun()

    user_input = st.chat_input("Schreibe hier deine Antwort ...")

    if user_input:
        if check_safety(user_input):
            st.session_state.safety_triggered = True
            st.session_state.messages.append({"role": "user", "content": user_input})
            log_message("user", user_input)
            st.session_state.user_messages_count += 1
            safety_msg = (
                "Dein Text enth
00e4lt Hinweise auf starke Belastung oder eine m
00f6gliche Krisensituation. "
                "Dieses KI-System kann in solchen Situationen keine Hilfe leisten. "
                "Bitte wende dich jetzt an eine vertraute Person oder an professionelle Hilfe. "
                "Bei akuter Gefahr rufe bitte den Notruf 112 an."
            )
            st.session_state.messages.append({"role": "assistant", "content": safety_msg})
            log_message("assistant", safety_msg)
            st.session_state.phase = "safety"
            write_summary_once()
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
            st.session_state.phase = "closing"  # direkt setzen, kein pending_finish
            write_summary_once()

        st.rerun()

elif st.session_state.phase == "closing":
    write_summary_once()  # zweite Absicherung

    st.subheader(f"Thema: {st.session_state.topic}")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    st.info(
        "Bitte dr
00fccke jetzt auf den Button \u2014 erst dann werden deine Antworten gespeichert "
        "und du gelangst zum Fragebogen."
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
    write_summary_once()  # letzte Absicherung
    st.success("Der Chatteil ist beendet.")
    st.write(
        "Vielen Dank f
00fcr deine Teilnahme. "
        "Im n
00e4chsten Schritt geht es im Fragebogen mit einigen Fragen zu deiner Erfahrung weiter."
    )

    if st.session_state.return_url:
        safe_url = quote(st.session_state.return_url, safe=":/?&=%#")
        st.markdown(
            f'<a href="{safe_url}" target="_self" style="text-decoration:none;">'
            '<div style="display:inline-block;padding:0.7rem 1rem;background:#e9dfcf;'
            'color:#111111;border-radius:0.6rem;font-weight:600;border:1px solid #cbbda8;">'
            "Zur
00fcck zum Fragebogen</div></a>",
            unsafe_allow_html=True,
        )
    else:
        st.info("Bitte wechsle manuell zur
00fcck zum Fragebogen-Tab in deinem Browser.")

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
                "phase", "messages", "turn", "session_id", "session_start",
                "session_end", "summary_saved", "chat_completed", "topic",
                "safety_triggered", "closing_logged", "user_messages_count",
                "validation_fail_count", "fallback_count",
                "closing_validation_fail_count", "closing_fallback_count",
                "last_llm_error", "last_llm_raw_reply", "last_llm_status",
                "last_prompt_excerpt", "gsheet_error",
            ]:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()
