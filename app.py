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

QUESTION_START_WORDS = ["Was", "Wie"]
FORBIDDEN_QUESTION_STARTS = [
    "Warum", "Wieso", "Weshalb", "Wann", "Wer", "Welche", "Woran", "Inwiefern"
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

    base_url = st.secrets.get("LLM_BASE_URL", "https://api.openai.com/v1")
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


def starts_like_previous_assistant(text: str) -> bool:
    assistant_messages = [
        msg["content"].strip()
        for msg in st.session_state.messages
        if msg["role"] == "assistant" and msg["content"].strip()
    ]
    if not assistant_messages:
        return False

    last_assistant = assistant_messages[-1]
    current_words = text.strip().split()
    previous_words = last_assistant.strip().split()

    if len(current_words) < 2 or len(previous_words) < 2:
        return False

    current_start_2 = " ".join(current_words[:2]).lower()
    previous_start_2 = " ".join(previous_words[:2]).lower()
    current_start_3 = " ".join(current_words[:3]).lower() if len(current_words) >= 3 else ""
    previous_start_3 = " ".join(previous_words[:3]).lower() if len(previous_words) >= 3 else ""

    return current_start_2 == previous_start_2 or (
        current_start_3 and previous_start_3 and current_start_3 == previous_start_3
    )


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
    if len(words) < 10 or len(words) > 80:
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

    if starts_like_previous_assistant(normalized):
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
        variants_high = [
            "Im Moment bleibt noch offen, woran du dieses studienbezogene Thema zuerst festmachst. Wenn du an den Anfang denkst, was fällt dir dort als Erstes auf?",
            "Gerade ist noch schwer zu greifen, an welchem Punkt dein studienbezogenes Thema konkret beginnt. Wenn du an diese Situation denkst, was ist zuerst bemerkbar?",
            "Noch ist nicht klar, welcher Teil deines studienbezogenen Themas im Moment am ehesten greifbar wird. Wenn du an den Anfang denkst, was taucht zuerst auf?",
        ]
        variants_low = [
            "Im Moment bleibt noch unklar, welcher konkrete Punkt an diesem studienbezogenen Thema zuerst greifbar wird. Wenn du an die Situation denkst, was fällt als Erstes auf?",
            "Noch ist offen, an welchem Punkt dieses studienbezogene Thema im Alltag zuerst sichtbar wird. Wenn du an den Anfang denkst, was lässt sich zuerst benennen?",
            "Der erste konkrete Ansatzpunkt in diesem studienbezogenen Thema bleibt noch unscharf. Was fällt dir an dieser Situation zuerst auf?",
        ]
        pool = variants_high if cond == "high" else variants_low
        return pool[st.session_state.turn % len(pool)]

    variants_high = [
        "Mehrere Punkte rund um dein studienbezogenes Thema stehen gerade nebeneinander. Wenn du an die aktuelle Situation denkst, was tritt zuerst hervor?",
        "In deiner Beschreibung laufen gerade verschiedene studienbezogene Punkte zusammen. Wenn du an diesen Moment denkst, was ist am deutlichsten bemerkbar?",
        "Gerade kommen mehrere Aspekte deines studienbezogenen Themas gleichzeitig vor. Was fällt in dieser Situation zuerst auf?",
    ]
    variants_low = [
        "Mehrere Punkte rund um das studienbezogene Thema werden hier gleichzeitig beschrieben. Wenn du an die aktuelle Situation denkst, was tritt zuerst hervor?",
        "In der Beschreibung laufen verschiedene Aspekte des studienbezogenen Themas zusammen. Was ist in diesem Moment am deutlichsten bemerkbar?",
        "Hier werden mehrere studienbezogene Punkte nebeneinander sichtbar. Was fällt in dieser Situation zuerst auf?",
    ]
    pool = variants_high if cond == "high" else variants_low
    return pool[st.session_state.turn % len(pool)]


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
- Du stellst keine Diagnosen, erklärst keine psychologischen Modelle und verwendest keine psychologischen Fachbegriffe.
- Du simulierst keine menschliche Beziehung, keine Empathie und keine emotionale Begleitung.

THEMENRAHMEN
- Die Person schreibt über ein studienbezogenes Anliegen, zum Beispiel Prüfungsdruck, Abschlussarbeit, Motivation, Zeitmanagement, Unsicherheit im Studium oder Konflikte im Hochschulkontext.
- Wenn andere Lebensbereiche erwähnt werden, darfst du sie nur kurz aufgreifen, sofern die Person sie selbst genannt hat.
- Der Schwerpunkt bleibt beim studienbezogenen Thema.

ZIEL DER INTERAKTION
- Deine Funktion ist minimale kognitive Strukturierung.
- Du hilfst dabei, allgemeine oder diffuse Beschreibungen in konkretere Beobachtungen, Situationen oder Abläufe zu überführen.
- Du veränderst das Thema nicht, interpretierst nicht und entwickelst keine Theorie über die Person.
- Du bleibst möglichst auf derselben Bedeutungsebene wie die Aussage der Person.
- Du formulierst näher an der subjektiven Beschreibung als an einer analytischen Einordnung.
- Du kommentierst die Kommunikation mit dem KI-System nicht als eigenes Thema.

GRUNDREGEL FÜR JEDE ANTWORT
- Jede Antwort hat genau zwei Teile: 1) ein kurzer, eigener Fokus-Satz, 2) genau eine offene Frage.
- Der Fokus-Satz greift genau einen greifbaren Moment, einen kleinen Ablauf oder eine unmittelbar sichtbare Situation auf.
- Wiederhole die Eingabe nicht bloß mit anderen Worten und verwende keine feste Einleitungsformel in jeder Antwort.
- Füge keine neuen Motive, Ursachen, Bewertungen oder psychologischen Deutungen hinzu.
- Die Frage knüpft direkt an diesen Moment an und öffnet nur eine Facette weiter: Wahrnehmung, Ablauf, erster Gedanke, Körper oder Umgebung.

REGEL FÜR DEN SATZANFANG
- Variiere den Einstieg sichtbar.
- Vermeide wiederkehrende Standardstarter wie "Du beschreibst gerade ...", "Hier zeigt sich ...", "Es geht gerade ...", "Im Moment bleibt ..." oder "Gerade ist noch ...".
- Inhalt, Funktion und Kürze bleiben gleich; nur die sprachliche Oberfläche variiert.

KONKRETISIERUNG DES FOKUS-SATZES
- Wähle aus der letzten Eingabe genau einen Moment oder Abschnitt, der darin deutlich wird.
- Formuliere diesen Moment in eigenen Worten, aber eng an der Beschreibung der Person.
- Wenn die neue Eingabe erkennbar an einen vorherigen Punkt anknüpft, darfst du genau einen solchen früheren Punkt knapp mit erwähnen, aber du fasst nicht mehrere frühere Turns zusammen.
- Sprich über die Situation oder den Moment, nicht über „die Aussage“, „die Rückmeldung“ oder „die Kommunikation“.

FRAGEREGEL
- Stelle genau eine offene Frage.
- Die Frage steht am Ende der Antwort.
- Die Frage beginnt nur mit "Was" oder "Wie".
- Die Frage soll leicht beantwortbar sein und sich auf Wahrnehmung, Ablauf, unmittelbare Gedanken, Körper oder Umgebung beziehen.
- Sie führt keinen neuen Themenbereich ein.

VERBOTENE FRAGEFORMEN
- "Warum ..."
- "Woran ..."
- "Inwiefern ..."
- "Welche ..."
- "Wann ..."
- "Wer ..."
- "Was bedeutet das ..."
- "Welche tieferen Gründe ..."

UMGANG MIT KURZEN ODER UNKLAREN ANTWORTEN
- Auch sehr kurze Antworten wie "Ich weiß nicht", "Keine Ahnung" oder "gar nicht" sind ernst zu nehmen.
- Greife dann nur einen ersten kleinen Ansatzpunkt oder einen ersten greifbaren Moment auf.
- Stelle eine kleine, konkrete Frage, die hilft, einen ersten Moment, eine kleine Situation oder einen ersten Gedanken sichtbar zu machen.

UMGANG MIT SCHWIERIGKEITEN IN DER INTERAKTION
- Wenn die Person Unsicherheit, Frustration oder Schwierigkeiten mit dem Chat äußert, greife den benannten Inhalt direkt auf.
- Formuliere eine Frage, die hilft, einen konkreten Punkt zu finden, an dem weiter reflektiert werden kann.
- Analysiere nicht die gesamte Gesprächsdynamik.

SPRACHE
- Formuliere einfach, kurz und alltagsnah.
- Klinge nicht wie ein wissenschaftlicher Text.
- Klinge nicht wie eine Therapeutin, ein Coach oder ein psychologischer Berater.
- Verwende keine tröstende Sprache und keine emotionale Resonanz.
- Vermeide sichtbare Schablonen und wiederkehrende feste Satzanfänge.

VERBOTENE FORMULIERUNGEN
- "Das klingt belastend."
- "Ich kann verstehen, dass ..."
- "Das muss schwer für dich sein."
- "Vielleicht steckt dahinter ..."
- "Es könnte sein, dass ..."
- "Vermutlich ..."
- "Ich bin für dich da."
- "Danke für dein Vertrauen."
- "Du bist nicht allein."
- "Ich begleite dich."

FORMATREGELN
- Du antwortest auf Deutsch.
- Deine Antwort ist genau ein zusammenhängender Fließtextabschnitt.
- Du verwendest keine Bulletpoints, keine Listen und keine mehreren Absätze.
- Deine Antwort enthält genau ein Fragezeichen.
- Die Antwort umfasst ungefähr 10 bis 80 Wörter.
- Die Sitzung umfasst insgesamt {max_rounds} Nutzereingaben.
"""

    low_style = """
STILREGELN FÜR DIE LOW-BEDINGUNG
- Formuliere sachlich, ruhig und eher inhaltsbezogen.
- Direkte Du-Ansprache vermeidest du möglichst.
- Die Sprache soll nüchtern wirken, aber trotzdem normal und flüssig klingen.
- Klinge nicht mechanisch, nicht steif und nicht akademisch.
"""

    high_style = """
STILREGELN FÜR DIE HIGH-BEDINGUNG
- Formuliere etwas natürlicher und leicht personenbezogener als in der Low-Bedingung.
- Du darfst Du-Ansprache verwenden.
- Die Sprache darf gesprächsnäher wirken, bleibt aber sachlich und nicht fürsorglich oder tröstend.
- Klinge nicht locker, nicht casual und nicht emotionalisierend.
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
- Keine Sätze wie „Das klingt belastend“, „Ich bin für dich da“, „Du hast das gut gemacht“, „Das war mutig“.
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
"""

    high_style = """
STILREGELN – BEDINGUNG B (natürlich)
- Natürlich, leicht personenbezogen, aber nicht empathisch oder fürsorglich.
- Du-Ansprache selbstverständlich.
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
        st.session_state.phase = "closing"
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

            st.session_state.phase = "safety"
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

elif st.session_state.phase == "closing":
    st.subheader(f"Reflexion zum Thema: {st.session_state.topic}")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    st.info("Die Reflexion ist abgeschlossen. Bitte klicke unten, um zum Fragebogen weiterzugehen.")

    if st.button("Weiter zum Fragebogen", type="primary"):
        st.session_state.phase = "finished"
        st.rerun()

elif st.session_state.phase == "safety":
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    st.warning(
        "Diese Reflexionssitzung wird jetzt beendet. "
        "Bitte wende dich bei Bedarf an eine der genannten Stellen."
    )

    if st.button("Sitzung beenden", type="primary"):
        st.session_state.phase = "finished"
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
