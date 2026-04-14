import csv
import re
from datetime import datetime
from pathlib import Path
from urllib.parse import quote

import pandas as pd
import streamlit as st

from llm_client import call_llm, PROMPT_VERSION, LLM_MODEL, LLM_BASE_URL, log_error

st.set_page_config(page_title="KI-Reflexionschat", page_icon="💬", layout="centered")

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
LOG_FILE = DATA_DIR / "chat_logs.csv"
SUMMARY_FILE = DATA_DIR / "chat_sessions.csv"

SAFETY_KEYWORDS = [
    "suizid", "ich will sterben", "nicht mehr leben",
    "mich umbringen", "bring mich um",
    "selbst verletzen", "selbstverletzung",
    "jemanden umbringen", "jemanden verletzen",
]

FORBIDDEN_PHRASES = [
    "ich fühle", "ich fuehle",
    "ich bin für dich da", "ich bin fuer dich da",
    "danke für dein vertrauen", "danke fuer dein vertrauen",
    "es tut mir leid",
    "du solltest", "du musst",
    "nächster schritt", "naechster schritt",
    "warum", "was wirst du tun",
    "bindung", "vermeidung", "dissonanz",
    "ich verstehe dich", "ich fuehle mit dir",
]

QUESTION_START_WORDS = ["Was", "Wie", "Woran", "Inwiefern", "Welche"]


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
    if word_count < 12 or word_count > 55:
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
    if word_count < 10 or word_count > 60:
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
        "ich", "du", "der", "die", "das", "und", "oder", "aber", "den", "dem",
        "des", "ein", "eine", "einer", "einem", "einen", "ist", "sind", "war",
        "bin", "bist", "im", "in", "am", "an", "auf", "mit", "zu", "von", "für",
        "dass", "es", "sich", "nicht", "noch", "wie", "was", "wird", "hier",
        "gerade", "aktuell", "moment", "besonders"
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

    if overlap_ratio > 0.6:
        return True

    if len(shared_bigrams) >= 2:
        return True

    return False


def fallback_reply(cond: str) -> str:
    if cond == "high":
        return (
            "Im Vordergrund steht für dich gerade, dass dieses Thema im Moment viel Raum einnimmt. "
            "Was ist daran aktuell besonders präsent?"
        )
    return (
        "Im Vordergrund steht hier, dass dieses Thema derzeit mit deutlicher Belastung verbunden ist. "
        "Was steht daran aktuell besonders im Vordergrund?"
    )


def check_safety(user_text: str) -> bool:
    text = user_text.lower()
    return any(kw in text for kw in SAFETY_KEYWORDS)


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
    max_rounds = get_param("rounds", "5")
    debug_mode = get_debug_mode()

    try:
        max_rounds_int = max(1, min(int(max_rounds), 10))
    except ValueError:
        max_rounds_int = 5

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
        "session_id": f"{pid}_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}",
        "session_start": now_iso(),
        "session_end": "",
        "chat_completed": False,
        "safety_triggered": False,
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
- Deine Antwort ist ein einziger Fließtextabschnitt ohne Bulletpoints.
- Deine Antwort enthält genau ein Fragezeichen.
- Die Frage steht am Ende.
- Die Frage beginnt nur mit: "Was", "Wie", "Woran", "Inwiefern" oder "Welche".
- Deine Antwort umfasst insgesamt 12 bis 55 Wörter.
- Du gibst keine Ratschläge, Empfehlungen oder Handlungsanweisungen.
- Du vermeidest Imperative.
- Du stellst keine Warum-Fragen.
- Du stellst keine Zukunftsfragen.
- Du stellst keine suggestiven oder diagnostischen Fragen.
- Du verwendest keine Formulierungen wie "ich fühle", "ich bin für dich da", "danke für dein Vertrauen", "es tut mir leid", "ich verstehe dich" oder "ich fühle mit dir".
- Variiere Einstiegsformulierungen leicht, ohne neue Inhalte hinzuzufügen.
- Beginne Antworten nicht wiederholt mit exakt denselben Satzanfängen.

Spiegelungsregeln:
- Du wiederholst nicht den Wortlaut der Person.
- Du übernimmst keine vollständigen Satzstrukturen oder längere Formulierungen aus dem Text.
- Du formulierst den Inhalt in eigenen Worten neu.
- Du benennst maximal 1–2 zentrale Aspekte.
- Du lässt Beispiele, Wiederholungen und Nebenaspekte weg.
- Du priorisierst, was im Text am stärksten im Vordergrund steht.
- Du verdichtest den Inhalt und machst sichtbar, was im Text im Vordergrund steht.
- Wenn mehrere Aspekte genannt werden, kannst du ihre Beziehung knapp sichtbar machen, z. B. als Gleichzeitigkeit, Zusammenhang oder Spannung.
- Verdichtung bedeutet hier: mehrere genannte Aspekte knapp zu ordnen oder auf einen benannten Schwerpunkt zu fokussieren, ohne neue Bedeutungen hinzuzufügen.
- Du verwendest nur Inhalte, die die Person selbst genannt hat.
- Du fügst keine neuen Emotionen, Motive oder Ursachen hinzu.
- Du interpretierst nicht und stellst keine Diagnosen.
- Du übersetzt die Aussage der Person nicht in psychologische Kategorien.
- Wenn die Person markante eigene Begriffe, Metaphern oder Selbstbeschreibungen verwendet, dürfen diese wörtlich übernommen werden.
- Solche Begriffe werden nicht durch fachlichere, neutralere oder emotional gefärbte Alternativen ersetzt.
- Du vermeidest semantische Umdeutungen einzelner Schlüsselbegriffe.
- Du benennst keine verdeckten Muster, Kreisläufe, Dynamiken oder inneren Mechanismen, wenn diese nicht ausdrücklich von der Person selbst genannt wurden.
- Du leitest keine Konsequenzen, Empfehlungen oder Haltungen aus dem Gesagten ab.
- Die Spiegelung ist kurz, präzise und strukturiert.

Wichtige Bedingungsregel:
- Die inhaltliche Qualität, Tiefe und Struktur der Antwort bleibt in beiden Bedingungen gleich.
- Variiert wird ausschließlich die sprachliche Perspektive (inhaltlich orientiert vs. leicht personenbezogen).
- Es dürfen keine zusätzlichen Inhalte, Bewertungen oder impliziten Bedeutungen zwischen den Bedingungen entstehen.
- Die Frage bleibt in beiden Bedingungen gleich offen und gleich wenig lenkend.
- Die Frage soll den bereits benannten Schwerpunkt weiter öffnen, nicht das Thema wechseln.

Antworte immer mit:
1. einer kurzen Spiegelung
2. genau einer offenen Frage am Ende

Die Sitzung umfasst ungefähr {max_rounds} Nutzereingaben.
"""

    low_style = """
Stil der low-Anthropomorphismus-Bedingung:
- formuliere sachlich, nüchtern und eher inhaltsbezogen
- beziehe dich stärker auf den dargestellten Inhalt oder die Beschreibung als auf die Person
- verwende eher distanzierte, strukturierende Formulierungen
- vermeide personenbezogene oder zwischenmenschlich wirkende sprachliche Nähe
- klinge klar und verständlich, aber nicht sozial zugewandt

Bevorzugte Formulierungsarten:
- "Im Vordergrund steht hier ..."
- "Es wird deutlich, dass ..."
- "In der Beschreibung zeigt sich ..."
- "Es tritt hervor, dass ..."
- "Aus der Beschreibung geht hervor, dass ..."
- "Die Schilderung macht deutlich, dass ..."

Kalibrierungsbeispiele:

Nutzertext: "Ich komme mit meiner Masterarbeit nicht voran und denke ständig daran, wie viel noch fehlt."
Antwort: "Im Vordergrund steht hier, dass die Masterarbeit derzeit mit anhaltendem Druck verbunden ist. Was wirkt daran im Moment besonders belastend?"

Nutzertext: "Ich verliere langsam den Überblick und weiß nicht, wo ich anfangen soll."
Antwort: "In der Beschreibung zeigt sich, dass mehrere Aspekte mit fehlender Übersicht und Struktur verbunden sind. Was steht daran aktuell besonders im Vordergrund?"
"""

    high_style = """
Stil der high-Anthropomorphismus-Bedingung:
- formuliere leicht personenbezogener und etwas natürlicher
- beziehe dich stärker auf die Perspektive und Darstellung der Person
- bleibe sachlich und klar nicht-menschlich
- vermeide therapeutische, tröstende oder beziehungsorientierte Sprache
- klinge nicht wärmer oder hilfreicher als in der low-Bedingung, nur etwas personalisierter
- die Personalisierung zeigt sich nur in der sprachlichen Bezugnahme auf "du" bzw. "deine Schilderung", nicht in zusätzlicher Validierung oder emotionaler Anteilnahme

Bevorzugte Formulierungsarten:
- "Im Vordergrund steht für dich gerade ..."
- "Gerade wirkt besonders präsent, dass ..."
- "In deiner Schilderung wird deutlich, dass ..."
- "In dem, was du beschreibst, wird sichtbar, dass ..."
- "Wenn man deine Beschreibung betrachtet, wird deutlich, dass ..."
- "In dem, wie du es darstellst, tritt hervor, dass ..."

Kalibrierungsbeispiele:

Nutzertext: "Ich komme mit meiner Masterarbeit nicht voran und denke ständig daran, wie viel noch fehlt."
Antwort: "Im Vordergrund steht für dich gerade, dass die Masterarbeit im Moment viel Raum einnimmt und mit anhaltendem Druck verbunden ist. Was wirkt daran im Moment besonders belastend?"

Nutzertext: "Ich verliere langsam den Überblick und weiß nicht, wo ich anfangen soll."
Antwort: "In deiner Schilderung wird deutlich, dass sich mehrere Aspekte rund um fehlende Übersicht und Struktur bündeln. Was steht daran aktuell besonders im Vordergrund?"
"""

    if cond == "high":
        return base + "\n" + high_style
    return base + "\n" + low_style


def build_closing_prompt(cond: str) -> str:
    base = """
Du bist ein KI-basiertes Reflexionssystem im Rahmen einer psychologischen Studie.

Du bist kein Mensch, empfindest keine Emotionen und bildest keine Beziehung im menschlichen Sinn.
Du bist keine Therapie, kein Coaching, keine Diagnostik und gibst keine Ratschläge, Lösungen oder Ziele vor.
Du erklärst keine psychologischen Modelle, verwendest keine Fachbegriffe und stellst keine Diagnosen.

Deine Aufgabe in dieser letzten Nachricht ist es, die bisherige Reflexion in einem sehr kurzen, neutralen Abschluss zu bündeln.

Regeln für diese Abschlussnachricht:
- Du antwortest auf Deutsch.
- Du formulierst einen einzigen, kurzen Fließtextabschnitt ohne Bulletpoints.
- Deine Antwort enthält kein Fragezeichen.
- Du fasst nur 1–2 zentrale, bereits genannte Schwerpunkte der Reflexion zusammen.
- Du verwendest ausschließlich Inhalte, die die Person selbst benannt hat.
- Du fügst keine neuen Emotionen, Motive, Ursachen oder Bewertungen hinzu.
- Du interpretierst nicht.
- Du gibst keine Ratschläge, Empfehlungen oder Handlungsanweisungen.
- Du verwendest keine psychologischen Fachbegriffe, keine Diagnosen und keine Zukunftsaussagen.
- Wenn die Person markante eigene Begriffe oder Metaphern verwendet hat, dürfen diese beibehalten werden.
- Der Abschluss soll ruhig, knapp und ordnend wirken und das Ende der Reflexion markieren.
"""

    low_style = """
Stil der low-Anthropomorphismus-Bedingung:
- formuliere sachlich, nüchtern und eher inhaltsbezogen
- beziehe dich stärker auf den dargestellten Inhalt oder die Beschreibung als auf die Person
- verwende eher distanzierte, strukturierende Formulierungen

Bevorzugte Formulierungsarten für den Abschluss:
- "Abschließend wird sichtbar, dass ..."
- "In dieser kurzen Reflexion trat besonders hervor, dass ..."
- "Zusammenfassend zeigt sich in der Beschreibung, dass ..."
"""

    high_style = """
Stil der high-Anthropomorphismus-Bedingung:
- formuliere leicht personenbezogener und etwas natürlicher
- beziehe dich stärker auf die Perspektive und Darstellung der Person
- bleibe sachlich und klar nicht-menschlich

Bevorzugte Formulierungsarten für den Abschluss:
- "Abschließend wird in deiner Schilderung sichtbar, dass ..."
- "In dieser kurzen Reflexion trat für dich besonders hervor, dass ..."
- "Zusammenfassend zeigt sich für dich, dass ..."
"""

    if cond == "high":
        return base + "\n" + high_style
    return base + "\n" + low_style


def generate_llm_reply(user_text: str, cond: str, topic: str, turn: int, max_rounds: int) -> str:
    system_prompt = build_system_prompt(cond=cond, max_rounds=max_rounds)

    context = [
        f"Das Thema der Person lautet: {topic}",
        f"Letzte Eingabe der Person: {user_text}",
    ]

    raw_reply = call_llm(
        system_prompt=system_prompt,
        messages=context,
        cond=cond,
        session_id=st.session_state.session_id,
    )

    if raw_reply and validate_response(raw_reply) and not too_similar(user_text, raw_reply):
        return raw_reply

    retry_context = context + [
        "Formuliere die Spiegelung diesmal deutlich stärker verdichtend und strukturiert. "
        "Vermeide Wiederholungen des Wortlauts der Person, außer bei einzelnen zentralen Schlüsselbegriffen."
    ]

    retry_reply = call_llm(
        system_prompt=system_prompt,
        messages=retry_context,
        cond=cond,
        session_id=st.session_state.session_id,
    )

    if retry_reply and validate_response(retry_reply) and not too_similar(user_text, retry_reply):
        return retry_reply

    log_error(
        "fallback_used",
        f"raw_reply={repr(raw_reply)} retry_reply={repr(retry_reply if 'retry_reply' in locals() else None)}",
        session_id=st.session_state.session_id,
    )
    return fallback_reply(cond)


def generate_closing_reply(cond: str, topic: str, recent_user_texts: list[str]) -> str:
    system_prompt = build_closing_prompt(cond=cond)

    joined_recent = "\n".join(
        [f"- {txt}" for txt in recent_user_texts if txt.strip()]
    )

    context = [
        f"Das Thema der Person lautet: {topic}",
        "Die folgenden letzten Nutzereingaben sollen für den Abschluss berücksichtigt werden:",
        joined_recent,
        "Dies ist die letzte Nachricht der Reflexion. Fasse die zuletzt sichtbaren Schwerpunkte in 1–2 kurzen Sätzen zusammen, ohne neue Inhalte hinzuzufügen.",
    ]

    raw_reply = call_llm(
        system_prompt=system_prompt,
        messages=context,
        cond=cond,
        session_id=st.session_state.session_id,
    )

    if raw_reply and validate_closing_response(raw_reply):
        return raw_reply

    log_error("closing_fallback_used", f"raw_reply={repr(raw_reply)}", session_id=st.session_state.session_id)

    if cond == "high":
        return "Abschließend wird in deiner Schilderung sichtbar, dass dieses studienbezogene Thema derzeit viel Raum einnimmt und mehrere belastende Aspekte zusammenkommen."
    return "Abschließend wird sichtbar, dass dieses studienbezogene Thema derzeit mit mehreren belastenden Aspekten verbunden ist."


def get_condition_label(cond: str) -> str:
    if cond == "high":
        return "high-anthropomorph"
    return "low-anthropomorph"


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
    st.markdown(
        """
Willkommen zur KI-Reflexionssession.

Im Rahmen dieser kurzen Session reflektierst du ein aktuelles studienbezogenes Thema.
"""
    )

    topic = st.text_area(
        "Womit möchtest du dich in dieser kurzen Reflexion zu einem studienbezogenen Thema beschäftigen?",
        value=st.session_state.topic,
        placeholder="Zum Beispiel: Prüfungsdruck, Stress mit der Masterarbeit, Zukunftsunsicherheit im Studium …",
        height=120,
    )
    st.session_state.topic = topic

    if st.button("Reflexion starten", type="primary"):
        if not topic.strip():
            st.warning("Bitte gib zuerst ein Thema ein, bevor du die Reflexion startest.")
        else:
            intro_msg = (
                "Danke. Wir beginnen mit einer kurzen Reflexion zu deinem studienbezogenen Thema. "
                "Beschreibe zunächst, was dich daran aktuell besonders beschäftigt."
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

    user_input = st.chat_input("Schreibe hier deine Antwort …")

    if user_input:
        if check_safety(user_input):
            st.session_state.safety_triggered = True
            st.session_state.messages.append({"role": "user", "content": user_input})
            log_message("user", user_input)

            safety_msg = (
                "Dein Text enthält Hinweise auf starke Belastung oder mögliche Krisensituationen. "
                "Dieses KI-System kann keine Hilfe in Krisen leisten. "
                "Bitte wende dich an vertraute Personen oder professionelle Hilfsangebote "
                "(z. B. Telefonseelsorge, psychologische Beratungsstellen oder den Notruf 112 bei akuter Gefahr). "
                "Du kannst die Teilnahme an der Studie hier beenden."
            )
            st.session_state.messages.append({"role": "assistant", "content": safety_msg})
            log_message("assistant", safety_msg)
            st.session_state.phase = "finished"
            st.rerun()

        st.session_state.messages.append({"role": "user", "content": user_input})
        log_message("user", user_input)

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

        if st.session_state.turn >= st.session_state.max_rounds:
            recent_user_texts = [
                msg["content"]
                for msg in st.session_state.messages
                if msg["role"] == "user"
            ][-3:]

            closing_reply = generate_closing_reply(
                cond=st.session_state.cond,
                topic=st.session_state.topic,
                recent_user_texts=recent_user_texts,
            )

            st.session_state.messages.append({"role": "assistant", "content": closing_reply})
            log_message("assistant", closing_reply)

            st.session_state.chat_completed = True
            st.session_state.phase = "finished"

        st.rerun()

elif st.session_state.phase == "finished":
    write_summary_once()
    st.success("Der Chatteil ist beendet.")
    st.write(
        "Vielen Dank für deine Teilnahme an diesem Chatteil. "
        "Im nächsten Schritt geht es im Fragebogen weiter mit einigen Fragen zu deiner Erfahrung."
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
            ]:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()
