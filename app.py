import csv
import re
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

TEMPERATURE = 0.5
MAX_RETRIES = 3

SAFETY_KEYWORDS = [
    "suizid",
    "ich will sterben",
    "will nicht mehr leben",
    "will nicht mehr",
    "ich will nicht mehr leben",
    "ich kann nicht mehr",
    "nicht mehr leben",
    "mich umbringen",
    "bring mich um",
    "selbst verletzen",
    "selbstverletzung",
    "jemanden umbringen",
    "jemanden verletzen",
]

FORBIDDEN_PHRASES = [
    "ich fühle", "ich fuehle",
    "ich bin für dich da", "ich bin fuer dich da",
    "danke für dein vertrauen", "danke fuer dein vertrauen",
    "es tut mir leid",
    "du solltest", "du musst",
    "nächster schritt", "naechster schritt",
    "warum", "was wirst du tun",
    "strategie", "strategien",
    "lösung", "lösungen", "loesung", "loesungen",
    "maßnahme", "maßnahmen", "massnahme", "massnahmen",
    "plan", "pläne", "plaene",
    "bindung", "vermeidung", "dissonanz",
    "ich verstehe dich", "ich fuehle mit dir",
    "du bist nicht allein",
    "ich begleite dich",
]

QUESTION_START_WORDS = ["Was", "Wie", "Woran", "Inwiefern", "Welche"]


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


def validate_response(text: str) -> bool:
    text = (text or "").strip()

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
    if word_count < 8 or word_count > 55:
        return False

    lower = text.lower()
    for phrase in FORBIDDEN_PHRASES:
        if phrase in lower:
            return False

    question_match = re.search(r"(Was|Wie|Woran|Inwiefern|Welche)\b.*\?$", text)
    if not question_match:
        return False

    question_start = question_match.start()
    reflection_part = text[:question_start].strip()
    question_part = text[question_start:].strip()

    if len(reflection_part.split()) < 3:
        return False
    if not any(question_part.startswith(word) for word in QUESTION_START_WORDS):
        return False
    if reflection_part.startswith(tuple(QUESTION_START_WORDS)):
        return False

    return True


def fallback_reply(cond: str, user_text: str = "") -> str:
    short = (user_text or "").strip().lower()

    unsure_forms = {
        "ich weiß nicht", "ich weiss nicht", "weiß nicht", "weiss nicht",
        "keine ahnung", "nicht sicher", "kp", "idk"
    }

    if short in unsure_forms or is_very_short_user_input(user_text):
        if cond == "high":
            return (
                "In deiner Schilderung bleibt noch offen, woran dieses studienbezogene Thema für dich gerade am deutlichsten greifbar wird. "
                "Was daran ist im Moment am ehesten fassbar?"
            )
        return (
            "Hier bleibt zunächst offen, woran dieses studienbezogene Thema derzeit am deutlichsten erkennbar wird. "
            "Woran zeigt sich im Moment am ehesten, was daran besonders ins Gewicht fällt?"
        )

    if cond == "high":
        return (
            "In deiner Schilderung wird sichtbar, dass dieses studienbezogene Thema derzeit viel Raum einnimmt und mehrere Anforderungen zusammenbringt. "
            "Was daran steht für dich im Moment am stärksten im Vordergrund?"
        )
    return (
        "Deutlich wird hier, dass dieses studienbezogene Thema derzeit viel Raum einnimmt und mehrere Anforderungen bündelt. "
        "Was steht daran im Moment am stärksten im Vordergrund?"
    )


def check_safety(user_text: str) -> bool:
    text = (user_text or "").lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
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
        "user_messages_count": 0,
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
            ]
        )


def build_system_prompt(cond: str, max_rounds: int) -> str:
    base = f"""
Du bist ein KI-basiertes Reflexionssystem im Rahmen einer psychologischen Studie im Hochschulkontext.

DEINE ROLLE UND GRENZEN
- Du bist ein KI-System und keine menschliche Person.
- Du empfindest keine Emotionen und bildest keine Beziehung im menschlichen Sinn.
- Du bist keine Therapie, kein Coaching, keine Beratung und keine Diagnostik.
- Du gibst keine Ratschläge, keine Lösungen und keine Handlungsempfehlungen.
- Du erklärst keine psychologischen Modelle, verwendest keine Fachbegriffe und stellst keine Diagnosen.
- Du bleibst transparent: Du verhältst dich eindeutig wie ein KI-System, nicht wie eine therapeutische oder beratende Person.

AUFGABE
- Du unterstützt eine einmalige, kurze Selbstreflexion zu einem studienbezogenen Thema der Person.
- Deine Hauptfunktion ist: Gedanken sichtbar machen, ordnen und strukturiert weiterführen.
- Du hilfst der Person, ihr studienbezogenes Thema klarer zu sehen, ohne neue Inhalte hinzuzufügen.

THEMENRAHMEN
- Die Person beschreibt studienbezogene Belastungen oder Themen, zum Beispiel Prüfungsdruck, Masterarbeit, Motivation, Zeitmanagement, Unsicherheit im Studium oder Konflikte im Hochschulkontext.
- Wenn die Person andere Lebensbereiche erwähnt, zum Beispiel Familie, Schlaf, Gesundheit oder Freizeit, darfst du diese kurz aufgreifen, aber der Schwerpunkt deiner Spiegelung und Frage bleibt beim studienbezogenen Anteil.
- Wenn Randkontexte genannt werden, soll die Reflexion zum studienbezogenen Kern zurückführen, ohne den Randkontext zu ignorieren.

ANTWORTFORMAT
- Du antwortest auf Deutsch.
- Deine Antwort ist genau ein zusammenhängender Fließtextabschnitt ohne Bulletpoints.
- Deine Antwort enthält genau ein Fragezeichen.
- Die Frage steht am Ende.
- Die Frage beginnt nur mit: "Was", "Wie", "Woran", "Inwiefern" oder "Welche".
- Deine Antwort umfasst insgesamt ungefähr 12 bis 50 Wörter.
- Bei sehr kurzen Nutzereingaben wie "Ich weiß nicht" darf die Antwort etwas kürzer sein, wenn sie trotzdem klar und anschlussfähig bleibt.
- Du verwendest keine Aufzählungen, keine Listen und keine mehreren Absätze.
- Du vermeidest Imperative.
- Du stellst keine Warum-Fragen.
- Du stellst keine Zukunftsfragen.
- Du stellst keine suggestiven oder diagnostischen Fragen.

INHALTLICHE REGELN
- Du verwendest nur Inhalte, die die Person selbst genannt hat.
- Du fügst keine neuen Emotionen, Motive, Ursachen, Deutungen oder Diagnosen hinzu.
- Du übersetzt Aussagen der Person nicht in psychologische Kategorien.
- Du wiederholst nicht einfach wörtlich den Text der Person.
- Du darfst zentrale Begriffe oder kurze Formulierungen punktuell aufgreifen, wenn sie subjektiv wichtig sind, vermeidest aber längere wörtliche Wiederholung.
- Du verdichtest den Inhalt und machst sichtbar, was im Text im Vordergrund steht.
- Verdichtung bedeutet: mehrere genannte Aspekte knapp zu ordnen oder auf einen benannten Schwerpunkt zu fokussieren, ohne neue Bedeutungen hinzuzufügen.
- Wenn mehrere Themen genannt werden, benennst du kurz die Mehrfachheit und wählst dann einen klaren Schwerpunkt.
- Frage nicht nach Strategien, Lösungen, Maßnahmen, Plänen oder konkreten Schritten.
- Richte die Frage auf Wahrnehmung, Gewichtung, Bedeutung oder bereits erlebte Zusammenhänge, nicht auf Problemlösung.
- Unterstelle keine versteckten Motive, keine Flucht, keine Vermeidung und keine psychologischen Muster.

REFLEXIONSALGORITHMUS
1. Identifiziere 1 bis maximal 2 zentrale inhaltliche Punkte aus der letzten Eingabe.
2. Formuliere eine kurze, strukturierende Spiegelung dieser Punkte in eigenen Worten.
3. Stelle genau eine offene Frage, die direkt an deine Spiegelung anschließt und denselben Schwerpunkt weiter öffnet.
4. Variiere die Satzanfänge deutlich und verwende dieselbe Einleitungsformulierung nicht wiederholt über mehrere Antworten hinweg.

WEITERE LEITLINIEN
- Bevorzuge konkrete Bezugnahme auf benannte Situationen, Gedanken und Schwierigkeiten statt allgemeiner Sammelbegriffe, wenn konkrete Informationen vorliegen.
- Vermeide stereotype Standardsätze, die immer gleich klingen.
- Die Frage darf nach Wahrnehmung, Einordnung, Gewichtung, Bedeutung oder bereits beschriebenen Zusammenhängen fragen.
- Auch sehr kurze Antworten wie "Ich weiß nicht" oder "Keine Ahnung" sind ernst zu nehmen; dann spiegele vor allem die Unklarheit oder Überforderung und frage nach einem kleinen ersten Ansatzpunkt.
- Die erste Reaktion auf die Person darf etwas orientierender sein, damit ein tatsächlicher Reflexionsprozess in Gang kommt, bleibt aber nicht-direktiv und rein strukturierend.

SPRACHLICHE NO-GOS
- Verwende keine Formulierungen wie "ich fühle", "ich bin für dich da", "danke für dein Vertrauen", "es tut mir leid", "ich verstehe dich" oder "ich fühle mit dir".
- Verwende keine Beziehungs- oder Trostformeln wie "du bist nicht allein", "ich begleite dich" oder ähnliche Näheangebote.
- Gib keine Handlungsanweisungen oder Tipps.
- Verwende nicht wiederholt dieselbe Einleitungsformulierung über mehrere Antworten hinweg.

Die Sitzung umfasst ungefähr {max_rounds} Nutzereingaben.
"""

    low_style = """
STILREGELN FÜR DIE LOW-ANTHROPOMORPHISMUS-BEDINGUNG
- Du formulierst sachlich, nüchtern und eher inhaltsbezogen.
- Du beziehst dich stärker auf das benannte Thema oder die Beschreibung als auf die Person.
- Direkte Du-Ansprache ist in dieser Bedingung möglichst zu vermeiden.
- Du verwendest neutrale, strukturierende Formulierungen.
- Du klingst klar, verständlich und geordnet, aber nicht sozial zugewandt.
- Die folgenden Formulierungen sind nur Beispiele und dürfen nicht stereotyp wiederholt werden.

Bevorzugte Formulierungsarten:
- "In der Beschreibung tritt hervor, dass ..."
- "Hier zeigt sich, dass ..."
- "Deutlich wird, dass ..."
- "Im geschilderten Thema wird sichtbar, dass ..."
- "Auffällig ist, dass ..."
"""

    high_style = """
STILREGELN FÜR DIE HIGH-ANTHROPOMORPHISMUS-BEDINGUNG
- Du formulierst leicht personenbezogener und natürlicher als in der low-Bedingung.
- Du verwendest Du-Ansprache in einer sachlich-formalen Weise.
- Du bleibst klar nicht-menschlich: keine emotionalen Bekundungen, kein Trost, keine Beziehungsangebote.
- Du klingst nicht wärmer oder fürsorglicher, sondern nur sprachlich etwas näher an der Person.
- Die Personalisierung zeigt sich in Bezugnahmen auf "du", "deine Schilderung" oder "für dich", nicht in zusätzlicher Validierung.
- Die folgenden Formulierungen sind nur Beispiele und dürfen nicht stereotyp wiederholt werden.

Bevorzugte Formulierungsarten:
- "In deiner Schilderung wird deutlich, dass ..."
- "Für dich steht gerade im Mittelpunkt, dass ..."
- "Du beschreibst, dass ..."
- "Gerade wirkt für dich besonders präsent, dass ..."
- "In dem, was du schilderst, wird sichtbar, dass ..."
"""

    if cond == "high":
        return base + "\n" + high_style
    return base + "\n" + low_style


def get_recent_context(messages: List[Dict[str, str]], max_items: int = 4) -> str:
    history = []
    for msg in messages:
        if msg["role"] in {"user", "assistant"}:
            history.append(f'{msg["role"]}: {msg["content"]}')
    if not history:
        return ""
    return "\n".join(history[-max_items:])


def build_api_messages(system_prompt: str, topic: str, turn: int, max_rounds: int, user_text: str) -> List[Dict[str, str]]:
    recent_context = get_recent_context(st.session_state.messages, max_items=4)

    user_payload = (
        f"Studienbezogenes Hauptthema der Person: {topic}\n"
        f"Aktuelle Rundenzahl: {turn} von {max_rounds}\n"
    )

    if recent_context:
        user_payload += f"Bisheriger kurzer Verlauf:\n{recent_context}\n"

    user_payload += (
        f"Letzte Eingabe der Person: {user_text}\n"
        "Formuliere jetzt genau eine Antwort gemäß allen Regeln."
    )

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_payload},
    ]


def call_llm(system_prompt: str, topic: str, turn: int, max_rounds: int, user_text: str) -> str:
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
        temperature=TEMPERATURE,
        max_tokens=140,
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

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            raw_reply = call_llm(
                system_prompt=system_prompt,
                topic=topic,
                turn=turn,
                max_rounds=max_rounds,
                user_text=user_text,
            )

            st.session_state.last_llm_raw_reply = raw_reply

            if validate_response(raw_reply):
                st.session_state.last_llm_status = f"LLM ok in Versuch {attempt}"
                return raw_reply

            st.session_state.last_llm_error = (
                f"Validierung fehlgeschlagen in Versuch {attempt}. Antwort: {raw_reply}"
            )
            time.sleep(0.4)

        except Exception as e:
            st.session_state.last_llm_error = f"{type(e).__name__}: {e}"
            st.session_state.last_llm_status = f"Fehler in Versuch {attempt}"
            time.sleep(0.7)

    st.session_state.last_llm_status = "Fallback ausgelöst"
    return fallback_reply(cond, user_text=user_text)


def get_condition_label(cond: str) -> str:
    if cond == "high":
        return "high-anthropomorph"
    return "low-anthropomorph"


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
                "session_id": st.session_state.session_id,
                "model": get_model_name(),
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


init_state()
render_debug_sidebar()

st.title("KI-Reflexionschat")
st.caption("Technischer Prototyp für die Masterarbeit")

if st.session_state.phase == "intro":
    st.markdown(
        """
Willkommen zur KI-Reflexionssession.

Im Rahmen dieser kurzen Session reflektierst du ein aktuelles studienbezogenes Thema.
Das System ist ein KI-basiertes Reflexionstool. Es dient nicht der Beratung oder Therapie und gibt keine konkreten Lösungen oder Handlungsempfehlungen.
Stattdessen unterstützt der Chat dabei, Gedanken zu einem studienbezogenen Thema zu strukturieren und weiter zu reflektieren.
"""
    )

    topic = st.text_area(
        "Mit welchem studienbezogenen Thema oder welcher studienbezogenen Belastung möchtest du dich in dieser kurzen Reflexion beschäftigen?",
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
            intro_msg = (
                "Danke. Wir beginnen nun mit einer kurzen Reflexion zu deinem studienbezogenen Thema. "
                "Beschreibe dein Thema zunächst möglichst so, dass der Chat deine Situation nachvollziehen kann. "
                "Hilfreich kann sein, kurz zu schildern, worum es geht, welche Gedanken dich dazu beschäftigen und warum das Thema im Moment relevant ist."
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

    if st.session_state.turn >= st.session_state.max_rounds:
        st.session_state.chat_completed = True

        if not st.session_state.closing_logged:
            closing = (
                "Danke für deine Reflexion. Der Chat ist nun beendet. "
                "Bitte fahre jetzt mit dem Fragebogen fort."
            )
            st.session_state.messages.append({"role": "assistant", "content": closing})
            log_message("assistant", closing)
            st.session_state.closing_logged = True

        st.session_state.phase = "finished"
        st.rerun()

    user_input = st.chat_input("Schreibe hier deine Antwort …")

    if user_input:
        if check_safety(user_input):
            st.session_state.safety_triggered = True
            st.session_state.messages.append({"role": "user", "content": user_input})
            log_message("user", user_input)
            st.session_state.user_messages_count += 1

            safety_msg = (
                "Dein Text enthält Hinweise auf starke Belastung oder mögliche Krisensituationen. "
                "Dieses KI-System kann in solchen Situationen keine Hilfe leisten. "
                "Bitte wende dich an vertraute Personen oder professionelle Hilfsangebote, zum Beispiel eine psychologische Beratungsstelle, die Telefonseelsorge oder bei akuter Gefahr den Notruf 112. "
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
            with st.spinner("Antwort wird erzeugt ..."):
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
                "user_messages_count",
                "last_llm_error",
                "last_llm_raw_reply",
                "last_llm_status",
                "last_prompt_excerpt",
            ]:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()
