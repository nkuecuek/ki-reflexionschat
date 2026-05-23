import csv
import html
import re
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import quote

import pandas as pd
import streamlit as st

from llm_client import (
    LLM_BASE_URL,
    LLM_MODEL,
    PROMPT_VERSION,
    call_llm,
    log_error,
)

st.set_page_config(page_title="KI-Reflexionschat", page_icon="💬", layout="centered")

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
LOG_FILE = DATA_DIR / "chat_logs.csv"
SUMMARY_FILE = DATA_DIR / "chat_sessions.csv"

MAX_INPUT_CHARS = 1800
DEFAULT_MAX_ROUNDS = 6
MIN_WORDS_REPLY = 12
MAX_WORDS_REPLY = 55
MIN_WORDS_CLOSING = 10
MAX_WORDS_CLOSING = 60

SAFETY_KEYWORDS = [
    "suizid",
    "suizidgedanken",
    "ich will sterben",
    "nicht mehr leben",
    "ich will nicht mehr leben",
    "nicht mehr da sein",
    "wäre besser nicht da zu sein",
    "waere besser nicht da zu sein",
    "wäre besser nicht zu existieren",
    "waere besser nicht zu existieren",
    "mich umbringen",
    "bring mich um",
    "mir etwas antun",
    "mir was antun",
    "mir selber etwas antun",
    "selbst verletzen",
    "selbstverletzung",
]

FORBIDDEN_PHRASES = [
    "ich fühle",
    "ich fuehle",
    "ich bin für dich da",
    "ich bin fuer dich da",
    "danke für dein vertrauen",
    "danke fuer dein vertrauen",
    "es tut mir leid",
    "du solltest",
    "du musst",
    "nächster schritt",
    "naechster schritt",
    "warum",
    "was wirst du tun",
    "bindung",
    "vermeidung",
    "dissonanz",
    "ich verstehe dich",
    "ich fuehle mit dir",
    "ich kann gut verstehen",
    "das klingt so, als",
    "das hört sich so an, als",
    "das hoert sich so an, als",
]

QUESTION_START_WORDS = ["Was", "Wie", "Woran", "Inwiefern", "Welche"]

INTRO_TEXT = """
Willkommen zur KI-Reflexionssession.

In dieser kurzen Session reflektierst du ein aktuelles studienbezogenes Thema in einem textbasierten Dialog. Der Chat reagiert mit kurzen Rückmeldungen und offenen Fragen, um deine Selbstreflexion zu strukturieren.

Der Chat dient nicht der Beratung, Therapie oder dem Coaching und gibt keine konkreten Lösungen oder Handlungsempfehlungen. Stattdessen unterstützt er dabei, Gedanken zu einem studienbezogenen Thema, einer Herausforderung oder einer Unsicherheit im Studienalltag zu ordnen und zu reflektieren.

Bitte denke für die folgende Interaktion an ein Thema aus deinem Studienalltag, das dich aktuell beschäftigt. Das kann zum Beispiel sein:
- Stress oder Überforderung im Studium
- Schwierigkeiten mit Motivation oder Konzentration
- Unsicherheit bezüglich der Zukunft oder des Studienverlaufs
- Druck oder hohe Erwartungen an dich selbst
- Schwierigkeiten, den Überblick zu behalten
- Gedanken zu Entscheidungen oder Prioritäten im Studium

Du musst kein sehr persönliches oder stark belastendes Thema wählen. Entscheidend ist nur, dass es sich um ein Thema handelt, über das du einige Minuten reflektieren kannst.

Bitte nutze den Chat möglichst so, als würdest du deine Gedanken in einem Reflexionsprozess sortieren und beschreiben.

Wichtig:
- Das System ist ein KI-basiertes Reflexionstool und keine menschliche Person.
- Es gibt keine richtigen oder falschen Antworten.
- Bitte beschreibe nur so viel, wie du in diesem Rahmen teilen möchtest.
- Die Reflexion umfasst insgesamt sechs kurze Antwortschritte. Danach endet der Chatteil automatisch und du kehrst zum Fragebogen zurück.

Im ersten Schritt gibst du bitte an, mit welchem studienbezogenen Thema du dich in dieser kurzen Reflexion beschäftigen möchtest.
""".strip()


def now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def utc_stamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%d%H%M%S")


def ensure_csv_files() -> None:
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
                    "session_start",
                    "session_end",
                    "completed_chat",
                    "turns_completed",
                    "topic",
                    "safety_triggered",
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


def sanitize_user_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text[:MAX_INPUT_CHARS].strip()


def validate_topic(text: str) -> tuple[bool, str]:
    cleaned = sanitize_user_text(text)
    if not cleaned:
        return False, "Bitte gib zuerst ein Thema ein, bevor du die Reflexion startest."
    if len(cleaned.split()) < 3:
        return False, "Bitte beschreibe dein Thema etwas konkreter."
    return True, cleaned


def validate_chat_input(text: str) -> tuple[bool, str]:
    cleaned = sanitize_user_text(text)
    if not cleaned:
        return False, ""
    if len(cleaned.split()) < 5:
        return False, (
            "Bitte beschreibe deine Antwort etwas ausführlicher, damit der Chat sinnvoll darauf reagieren kann. "
            "Ein bis zwei Sätze reichen aus."
        )
    return True, cleaned


def validate_response(text: str) -> bool:
    text = text.strip()
    if not text:
        return False
    if text.count("?") != 1:
        return False
    if not text.endswith("?"):
        return False
    if any(line.strip().startswith(("-", "•", "*")) for line in text.splitlines()):
        return False
    if "\n\n" in text:
        return False

    word_count = len(text.split())
    if word_count < MIN_WORDS_REPLY or word_count > MAX_WORDS_REPLY:
        return False

    lower = text.lower()
    for phrase in FORBIDDEN_PHRASES:
        if phrase in lower:
            return False

    match = re.search(r"(Was|Wie|Woran|Inwiefern|Welche)\b.*\?$", text)
    if not match:
        return False

    question_start = match.start()
    reflection_part = text[:question_start].strip()
    question_part = text[question_start:].strip()

    if len(reflection_part.split()) < 4:
        return False
    if not any(question_part.startswith(word) for word in QUESTION_START_WORDS):
        return False
    if reflection_part.startswith(tuple(QUESTION_START_WORDS)):
        return False

    return True


def validate_closing_response(text: str) -> bool:
    text = text.strip()
    if not text:
        return False
    if "?" in text:
        return False
    if any(line.strip().startswith(("-", "•", "*")) for line in text.splitlines()):
        return False
    if "\n\n" in text:
        return False

    word_count = len(text.split())
    if word_count < MIN_WORDS_CLOSING or word_count > MAX_WORDS_CLOSING:
        return False

    lower = text.lower()
    for phrase in FORBIDDEN_PHRASES:
        if phrase in lower:
            return False

    return True


def normalize_for_similarity(text: str) -> list[str]:
    text = text.lower()
    text = re.sub(r"[^\wäöüß\s]", " ", text)
    words = text.split()

    stopwords = {
        "ich",
        "du",
        "der",
        "die",
        "das",
        "und",
        "oder",
        "aber",
        "den",
        "dem",
        "des",
        "ein",
        "eine",
        "einer",
        "einem",
        "einen",
        "ist",
        "sind",
        "war",
        "bin",
        "bist",
        "im",
        "in",
        "am",
        "an",
        "auf",
        "mit",
        "zu",
        "von",
        "für",
        "fuer",
        "dass",
        "es",
        "sich",
        "nicht",
        "noch",
        "wie",
        "was",
        "wird",
        "hier",
        "gerade",
        "aktuell",
        "moment",
        "besonders",
        "dabei",
        "steht",
        "tritt",
        "zeigt",
        "deutlich",
        "zentral",
        "dein",
        "deine",
        "deiner",
        "deinem",
    }
    return [w for w in words if w not in stopwords and len(w) > 2]


def too_similar(user_text: str, reply: str) -> bool:
    user_words = normalize_for_similarity(user_text)
    reply_words = normalize_for_similarity(reply)

    if not user_words or not reply_words:
        return False

    user_set = set(user_words)
    reply_set = set(reply_words)
    overlap_ratio = len(user_set & reply_set) / max(len(reply_set), 1)

    user_bigrams = set(zip(user_words, user_words[1:]))
    reply_bigrams = set(zip(reply_words, reply_words[1:]))
    shared_bigrams = user_bigrams & reply_bigrams

    return overlap_ratio > 0.6 or len(shared_bigrams) >= 2


def fallback_reply(cond: str) -> str:
    if cond == "high":
        return (
            "Du beschreibst, dass dieses Thema im Moment viel Raum einnimmt und dich belastet. "
            "Was daran ist gerade besonders präsent?"
        )
    return (
        "Deutlich wird hier, dass dieses Thema derzeit viel Raum einnimmt und mit Belastung verbunden ist. "
        "Was ist daran aktuell besonders wichtig?"
    )


def fallback_closing_reply(cond: str) -> str:
    if cond == "high":
        return (
            "Du beschreibst, dass dieses Thema im Moment weiterhin viel Raum einnimmt und dich belastet. "
            "Damit endet die kurze Reflexion zu deinem studienbezogenen Thema."
        )
    return (
        "Deutlich wird hier, dass dieses Thema derzeit weiterhin viel Raum einnimmt und mit Belastung verbunden ist. "
        "Damit endet die kurze Reflexion zu diesem studienbezogenen Thema."
    )


def build_safety_message() -> str:
    return (
        "Deine Eingabe enthält Hinweise auf eine mögliche starke psychische Belastung oder Gedanken an "
        "Selbstgefährdung. Dieser Chat kann in solchen Situationen nicht weitergeführt werden.\n\n"
        "Bitte wende dich an eine vertraute Person oder an professionelle Hilfe.\n\n"
        "Telefonseelsorge (kostenlos und anonym):\n"
        "0800 111 0 111\n"
        "0800 111 0 222\n"
        "116 123\n\n"
        "Bei akuter Gefahr: 112"
    )


def check_safety_keywords(text: str) -> bool:
    lower = text.lower().strip()

    if any(kw in lower for kw in SAFETY_KEYWORDS):
        return True
    if "nicht mehr" in lower and any(x in lower for x in ["leben", "da sein", "existieren"]):
        return True
    if any(x in lower for x in ["mir", "mich"]) and "antun" in lower:
        return True
    if "umbringen" in lower:
        return True

    return False


def llm_safety_check(text: str, session_id: str) -> bool:
    system_prompt = """
Du bist ein Sicherheitsklassifikator für ein KI-basiertes Reflexionssystem.

Deine einzige Aufgabe ist es zu prüfen, ob der folgende Text Hinweise auf mögliche
Selbstgefährdung, Suizidgedanken oder den Wunsch enthält, nicht mehr leben oder
nicht mehr existieren zu wollen.

Regeln:
- Antworte ausschließlich mit JA oder NEIN.
- JA = der Text enthält direkte oder indirekte Hinweise auf mögliche Selbstgefährdung,
  Suizidgedanken oder den Wunsch, nicht mehr leben / nicht mehr da sein zu wollen.
- NEIN = solche Hinweise sind nicht erkennbar.
- Antworte niemals mit Erklärungen, nur mit JA oder NEIN.
- Sprachliche Übertreibungen zu Studium, Prüfungen oder Stress ohne echte Selbstgefährdung
  sollen nicht als JA markiert werden.
""".strip()

    try:
        raw = call_llm(
            system_prompt=system_prompt,
            messages=[text],
            cond="safety",
            session_id=session_id,
        )
        if not raw:
            return False
        return raw.strip().upper().startswith("JA")
    except Exception as exc:
        log_error("safety_llm_error", repr(exc), session_id=session_id)
        return False


def check_safety_hybrid(user_text: str, messages: list[dict], session_id: str) -> bool:
    if check_safety_keywords(user_text):
        return True

    combined_recent = " ".join(
        msg["content"] for msg in messages if msg["role"] == "user"
    )[-1200:]

    if check_safety_keywords(combined_recent):
        return True

    if combined_recent:
        return llm_safety_check(combined_recent, session_id=session_id)

    return False


def get_condition_label(cond: str) -> str:
    return "high-anthropomorph" if cond == "high" else "low-anthropomorph"


def init_state() -> None:
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
        pid = f"test_{utc_stamp()}"

    return_url = get_param("return_url", "")
    max_rounds = get_param("rounds", str(DEFAULT_MAX_ROUNDS))
    debug_mode = get_debug_mode()

    try:
        max_rounds_int = max(1, min(int(max_rounds), 10))
    except ValueError:
        max_rounds_int = DEFAULT_MAX_ROUNDS

    defaults = {
        "phase": "intro",
        "pid": pid,
        "cond": cond,
        "return_url": return_url,
        "max_rounds": max_rounds_int,
        "debug_mode": debug_mode,
        "messages": [],
        "turn": 0,
        "topic": "",
        "session_id": f"{pid}_{utc_stamp()}",
        "session_start": now_iso(),
        "session_end": "",
        "chat_completed": False,
        "safety_triggered": False,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def log_message(role: str, text: str) -> None:
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


def write_summary_once() -> None:
    if st.session_state.session_end:
        return

    st.session_state.session_end = now_iso()

    with open(SUMMARY_FILE, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(
            [
                st.session_state.session_id,
                st.session_state.pid,
                st.session_state.cond,
                st.session_state.session_start,
                st.session_state.session_end,
                "yes" if st.session_state.chat_completed else "no",
                st.session_state.turn,
                st.session_state.topic,
                "yes" if st.session_state.safety_triggered else "no",
            ]
        )


def build_system_prompt(cond: str, max_rounds: int) -> str:
    base = f"""
Du bist ein KI-basiertes Reflexionssystem im Rahmen einer psychologischen Studie.

Du bist kein Mensch, empfindest keine Emotionen und bildest keine Beziehung im menschlichen Sinn.
Du bist keine Therapie, kein Coaching, keine Diagnostik und gibst keine Ratschläge, Lösungen oder Ziele vor.
Du erklärst keine psychologischen Modelle, verwendest keine Fachbegriffe und stellst keine Diagnosen.

Deine Aufgabe ist es, Selbstreflexion durch kurze, strukturierende Spiegelung zu unterstützen.

Allgemeine Regeln:
- Du antwortest auf Deutsch.
- Du antwortest als ein einziger Fließtextabschnitt ohne Bulletpoints.
- In den regulären Runden enthält deine Antwort genau eine offene Frage am Ende.
- Diese Frage beginnt nur mit: Was, Wie, Woran, Inwiefern oder Welche.
- Deine Antwort umfasst insgesamt 12 bis 55 Wörter.
- Du gibst keine Ratschläge, Empfehlungen oder Handlungsanweisungen.
- Du vermeidest Imperative.
- Du stellst keine Warum-Fragen.
- Du stellst keine Zukunftsfragen.
- Du stellst keine suggestiven oder diagnostischen Fragen.
- Du verwendest keine Beziehungssprache, keine Empathieformeln und keine therapeutische Sprache.
- Du formulierst knapp, konsistent und kontrolliert.

Struktur jeder regulären Antwort:
1. eine kurze, verdichtete Spiegelung von 1 bis 2 zentralen Aspekten,
2. genau eine offene Frage am Ende.

Spiegelungsregeln:
- Du wiederholst nicht den Wortlaut der Person.
- Du übernimmst keine vollständigen Satzstrukturen aus dem Nutzereingabetext.
- Du formulierst den Inhalt in eigenen Worten neu.
- Du benennst maximal 1 bis 2 zentrale Aspekte.
- Du lässt Beispiele, Nebenaspekte und Wiederholungen weg.
- Du priorisierst, was im Text am stärksten erkennbar ist.
- Du darfst markante Selbstformulierungen der Person punktuell beibehalten, wenn sie inhaltlich zentral sind.
- Du fügst keine neuen Emotionen, Motive, Ursachen oder Deutungen hinzu.
- Du interpretierst nicht und stellst keine Diagnosen.
- Du übersetzt Aussagen nicht in psychologische Kategorien.
- Du leitest keine Konsequenzen, Empfehlungen oder Haltungen aus dem Gesagten ab.

Bedingungsregel:
- Die inhaltliche Qualität, Tiefe, Gesprächsstruktur und Offenheit bleiben in beiden Bedingungen gleich.
- Variiert wird ausschließlich der sprachliche Stil.
- Der Stilunterschied darf keine zusätzlichen Inhalte oder Bewertungen erzeugen.

Gesprächsrahmen:
- Die Sitzung umfasst ungefähr {max_rounds} Nutzereingaben.
- Das Thema ist studienbezogen und sensibel, daher ist ein kontrollierter, passender Stil besonders wichtig.
""".strip()

    low_style = """
Stil der low-Anthropomorphismus-Bedingung:
- formuliere sachlich, nüchtern und eher inhaltsbezogen
- beziehe dich stärker auf den dargestellten Inhalt als auf die Person
- verwende distanzierte, strukturierende Formulierungen
- vermeide soziale Nähe, emotionale Validierung und personenzentrierte Wärme
- klinge klar, präzise und eher technisch-neutral
- bevorzuge eher nominale und beschreibende Formulierungen

Bevorzugte Formulierungen:
- "Deutlich wird hier, dass ..."
- "Zentral ist hier, dass ..."
- "In der Schilderung zeigt sich, dass ..."
- "Erkennbar wird, dass ..."
- "Aus der Beschreibung geht hervor, dass ..."

Kalibrierungsbeispiele:
Nutzertext: "Ich komme mit meiner Masterarbeit nicht voran und denke ständig daran, wie viel noch fehlt."
Antwort: "Deutlich wird hier, dass die Masterarbeit derzeit viel Raum einnimmt und mit anhaltendem Druck verbunden ist. Was ist daran aktuell besonders wichtig?"

Nutzertext: "Ich verliere langsam den Überblick und weiß nicht, wo ich anfangen soll."
Antwort: "In der Schilderung zeigt sich, dass mehrere Aspekte mit fehlender Übersicht und Struktur zusammenhängen. Was ist daran aktuell besonders wichtig?"
""".strip()

    high_style = """
Stil der high-Anthropomorphismus-Bedingung:
- formuliere natürlich, leicht personenbezogen und sprachlich zugänglicher
- bleibe kontrolliert, sachlich und nicht-therapeutisch
- nutze moderat die Perspektive der Person
- vermeide Mitgefühl, Trost, starke Wärme und jede menschenähnliche Rollensimulation
- der Stil soll human-like formal wirken, nicht locker oder flapsig
- bevorzuge eher verbale, direkte und etwas alltagsnähere Formulierungen

Bevorzugte Formulierungen:
- "Du beschreibst, dass ..."
- "So wie du es schilderst, wird deutlich, dass ..."
- "Bei dir zeigt sich im Moment, dass ..."
- "Gerade wird sichtbar, dass ..."
- "Im Moment wirkt besonders präsent, dass ..."

Kalibrierungsbeispiele:
Nutzertext: "Ich komme mit meiner Masterarbeit nicht voran und denke ständig daran, wie viel noch fehlt."
Antwort: "Du beschreibst, dass die Masterarbeit im Moment viel Raum einnimmt und dich belastet. Was daran ist gerade besonders präsent?"

Nutzertext: "Ich verliere langsam den Überblick und weiß nicht, wo ich anfangen soll."
Antwort: "So wie du es schilderst, wird deutlich, dass sich gerade vieles unübersichtlich anfühlt. Woran merkst du im Moment besonders, dass dir Struktur fehlt?"
""".strip()

    return base + "\n\n" + (high_style if cond == "high" else low_style)


def build_closing_prompt(cond: str) -> str:
    base = """
Du bist ein KI-basiertes Reflexionssystem im Rahmen einer psychologischen Studie.

Deine Aufgabe in dieser letzten Nachricht ist es, die bisherige Reflexion knapp und transparent abzuschließen.

Regeln:
- Antworte auf Deutsch.
- Formuliere einen einzigen kurzen Fließtextabschnitt.
- Verwende kein Fragezeichen.
- Beginne mit einer kurzen Spiegelung von 1 bis 2 bereits genannten zentralen Aspekten.
- Verwende ausschließlich Inhalte, die die Person selbst benannt hat.
- Füge keine neuen Emotionen, Motive, Ursachen oder Bewertungen hinzu.
- Interpretiere nicht.
- Gib keine Ratschläge, Empfehlungen oder Zukunftsaussagen.
- Verwende keine therapeutische oder diagnostische Sprache.
- Beende die Antwort mit einem neutralen Abschlussmarker.

Struktur:
1. kurze Spiegelung von 1 bis 2 zentralen Aspekten,
2. neutraler Abschluss der begrenzten Reflexion.

Geeignete Abschlussmarker:
- "Damit endet die kurze Reflexion zu diesem studienbezogenen Thema."
- "Damit endet die kurze Reflexion zu deinem studienbezogenen Thema."
""".strip()

    low_style = """
Stil der low-Anthropomorphismus-Bedingung:
- sachlich, nüchtern und eher inhaltsbezogen
- distanzierte, strukturierende Formulierungen

Bevorzugte Formulierungen:
- "Deutlich wird hier, dass ..."
- "In der Schilderung zeigt sich, dass ..."
- "Erkennbar bleibt, dass ..."
""".strip()

    high_style = """
Stil der high-Anthropomorphismus-Bedingung:
- leicht personenbezogen, aber nicht empathisch
- sachlich, klar und kontrolliert
- human-like formal, nicht locker

Bevorzugte Formulierungen:
- "Du beschreibst, dass ..."
- "So wie du es schilderst, wird deutlich, dass ..."
- "Bei dir zeigt sich im Moment, dass ..."
""".strip()

    return base + "\n\n" + (high_style if cond == "high" else low_style)


def generate_llm_reply(user_text: str, cond: str, topic: str, turn: int, max_rounds: int) -> str:
    system_prompt = build_system_prompt(cond=cond, max_rounds=max_rounds)

    context = [
        f"Thema der Person: {topic}",
        f"Aktuelle Runde: {turn} von ungefähr {max_rounds}",
        f"Letzte Eingabe der Person: {user_text}",
    ]

    raw_reply = call_llm(
        system_prompt=system_prompt,
        messages=context,
        cond=cond,
        session_id=st.session_state.session_id,
    )

    if raw_reply and validate_response(raw_reply) and not too_similar(user_text, raw_reply):
        return raw_reply.strip()

    retry_context = context + [
        "Formuliere jetzt knapper, stärker verdichtend und mit klarerem Stilunterschied der zugewiesenen Bedingung. "
        "Vermeide Wortlautübernahmen aus dem Nutzereingabetext."
    ]

    retry_reply = call_llm(
        system_prompt=system_prompt,
        messages=retry_context,
        cond=cond,
        session_id=st.session_state.session_id,
    )

    if retry_reply and validate_response(retry_reply) and not too_similar(user_text, retry_reply):
        return retry_reply.strip()

    log_error(
        "fallback_used",
        f"raw_reply={repr(raw_reply)} retry_reply={repr(retry_reply if 'retry_reply' in locals() else None)}",
        session_id=st.session_state.session_id,
    )
    return fallback_reply(cond)


def generate_closing_reply(cond: str, topic: str, recent_user_texts: list[str]) -> str:
    system_prompt = build_closing_prompt(cond=cond)

    joined_recent = "\n".join(f"- {txt}" for txt in recent_user_texts if txt.strip())

    context = [
        f"Thema der Person: {topic}",
        "Die folgenden letzten Nutzereingaben sollen für den Abschluss berücksichtigt werden:",
        joined_recent,
        "Dies ist die letzte Nachricht. Formuliere zuerst eine kurze Spiegelung von 1 bis 2 zentralen Aspekten "
        "und beende danach die begrenzte Reflexion transparent. Kein Fragezeichen. Keine neuen Inhalte. "
        "Kein Ratschlag. Keine Interpretation.",
    ]

    raw_reply = call_llm(
        system_prompt=system_prompt,
        messages=context,
        cond=cond,
        session_id=st.session_state.session_id,
    )

    if raw_reply and validate_closing_response(raw_reply):
        return raw_reply.strip()

    retry_context = context + [
        "Formuliere jetzt noch knapper und neutraler. Erste Satzhälfte = Spiegelung. Letzte Satzhälfte = klarer Abschlussmarker."
    ]

    retry_reply = call_llm(
        system_prompt=system_prompt,
        messages=retry_context,
        cond=cond,
        session_id=st.session_state.session_id,
    )

    if retry_reply and validate_closing_response(retry_reply):
        return retry_reply.strip()

    log_error(
        "closing_fallback_used",
        f"raw_reply={repr(raw_reply)} retry_reply={repr(retry_reply if 'retry_reply' in locals() else None)}",
        session_id=st.session_state.session_id,
    )
    return fallback_closing_reply(cond)


def render_return_button(url: str) -> None:
    safe_url = quote(url, safe=":/?&=%#")
    label = html.escape("Zurück zum Fragebogen")
    st.markdown(
        f'<a href="{safe_url}" target="_self" '
        'style="text-decoration:none;color:#111111;display:inline-block;">'
        '<div style="display:inline-block;padding:0.7rem 1rem;'
        'background:#e9dfcf;color:#111111;border-radius:0.6rem;'
        f'font-weight:600;border:1px solid #cbbda8;">{label}</div></a>',
        unsafe_allow_html=True,
    )


def reset_test_session() -> None:
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
    ]:
        if key in st.session_state:
            del st.session_state[key]
    st.rerun()


init_state()

st.title("KI-Reflexionschat")
st.caption("Technischer Prototyp für die Masterarbeit")

if st.session_state.debug_mode:
    with st.sidebar:
        st.markdown("### Studienparameter")
        st.write(
            {
                "pid": st.session_state.pid,
                "cond": st.session_state.cond,
                "cond_label": get_condition_label(st.session_state.cond),
                "rounds": st.session_state.max_rounds,
            }
        )
        st.markdown("### LLM-Konfiguration")
        st.write(
            {
                "model": LLM_MODEL,
                "base_url": LLM_BASE_URL,
                "prompt_version": PROMPT_VERSION,
            }
        )
        st.markdown("### Modus")
        st.info("Debug-/Testmodus aktiv (LLM)")
        st.markdown("### Session")
        st.write({"session_id": st.session_state.session_id})

if st.session_state.phase == "intro":
    st.markdown(INTRO_TEXT)

    topic = st.text_area(
        "Mit welchem studienbezogenen Thema möchtest du dich in dieser kurzen Reflexion beschäftigen?",
        value=st.session_state.topic,
        placeholder=(
            "Zum Beispiel: Unsicherheit im Studienverlauf, Stress im Studium, Schwierigkeiten mit Motivation, "
            "Druck bei der Masterarbeit oder Probleme, den Überblick zu behalten."
        ),
        height=220,
    )
    st.session_state.topic = topic

    if st.button("Reflexion starten", type="primary"):
        valid, topic_or_msg = validate_topic(topic)
        if not valid:
            st.warning(topic_or_msg)
        else:
            st.session_state.topic = topic_or_msg
            intro_msg = (
                "Die Reflexion beginnt jetzt zu diesem studienbezogenen Thema. "
                "Beschreibe dein Thema möglichst so, dass der Chat nachvollziehen kann, worum es geht. "
                "Hilfreich kann sein, kurz zu schildern, was dich aktuell beschäftigt, welche Gedanken dabei auftauchen "
                "und warum das Thema im Moment relevant ist."
            )
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

    user_input = st.chat_input("Schreibe hier deine Antwort in ein bis zwei Sätzen …")

    if user_input:
        valid, input_or_msg = validate_chat_input(user_input)
        if not valid:
            if input_or_msg:
                st.warning(input_or_msg)
            st.rerun()

        user_input = input_or_msg
        st.session_state.messages.append({"role": "user", "content": user_input})
        log_message("user", user_input)

        if check_safety_hybrid(
            user_text=user_input,
            messages=st.session_state.messages,
            session_id=st.session_state.session_id,
        ):
            st.session_state.safety_triggered = True

            safety_msg = build_safety_message()
            st.session_state.messages.append({"role": "assistant", "content": safety_msg})
            log_message("assistant", safety_msg)

            st.session_state.chat_completed = False
            st.session_state.phase = "finished"
            st.rerun()

        if st.session_state.turn >= st.session_state.max_rounds - 1:
            recent_user_texts = [
                msg["content"] for msg in st.session_state.messages if msg["role"] == "user"
            ][-3:]

            closing_reply = generate_closing_reply(
                cond=st.session_state.cond,
                topic=st.session_state.topic,
                recent_user_texts=recent_user_texts,
            )

            st.session_state.messages.append({"role": "assistant", "content": closing_reply})
            log_message("assistant", closing_reply)

            st.session_state.turn += 1
            st.session_state.chat_completed = True
            st.session_state.phase = "finished"
            st.rerun()

        reply = generate_llm_reply(
            user_text=user_input,
            cond=st.session_state.cond,
            topic=st.session_state.topic,
            turn=st.session_state.turn + 1,
            max_rounds=st.session_state.max_rounds,
        )

        st.session_state.messages.append({"role": "assistant", "content": reply})
        log_message("assistant", reply)

        st.session_state.turn += 1
        st.rerun()

elif st.session_state.phase == "finished":
    write_summary_once()
    st.success("Der Chatteil ist beendet. Bitte kehre zum Fragebogen zurück.")

    if st.session_state.safety_triggered:
        st.warning(
            "Die Sitzung wurde aus Sicherheitsgründen beendet. "
            "Dieser Chat ist nicht dafür geeignet, in solchen Situationen Unterstützung zu bieten."
        )

        st.markdown(
            """
Bitte wende dich an eine vertraute Person oder an professionelle Hilfe.

**Telefonseelsorge (kostenlos und anonym):**  
0800 111 0 111  
0800 111 0 222  
116 123  

**Bei akuter Gefahr:**  
112
"""
        )
    else:
        if st.session_state.return_url:
            render_return_button(st.session_state.return_url)
        else:
            st.info("Bitte wechsle zurück zum Fragebogen-Tab in deinem Browser.")

    if st.session_state.debug_mode:
        st.markdown("### Sitzungsdaten (lokale Vorschau)")
        if LOG_FILE.exists():
            df = pd.read_csv(LOG_FILE)
            session_df = df[df["session_id"] == st.session_state.session_id]
            st.dataframe(session_df, use_container_width=True)

        if st.button("Neue Testsitzung starten"):
            reset_test_session()
