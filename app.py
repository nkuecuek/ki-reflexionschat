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

TEMPERATURE = 0.7
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
FORBIDDEN_QUESTION_STARTS = ["Warum", "Wieso", "Weshalb", "Wann", "Wer"]


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
    if not text:
        return False

    raw = text.strip()
    normalized = " ".join(raw.split())

    if not normalized:
        return False

    if "\n" in raw.strip():
        return False

    if any(line.strip().startswith(("-", "•", "*")) for line in raw.splitlines()):
        return False

    if normalized.count("?") != 1:
        return False

    if not normalized.endswith("?"):
        return False

    words = normalized.split()
    if len(words) < 8 or len(words) > 55:
        return False

    lower = normalized.lower()
    for phrase in FORBIDDEN_PHRASES:
        if phrase in lower:
            return False

    psych_terms = [
        "depression", "depressiv", "angststörung", "angststoerung",
        "trauma", "symptom", "diagnose", "störung", "stoerung",
        "psychodynamisch", "vermeidungsmuster", "bindungsstil"
    ]
    if any(term in lower for term in psych_terms):
        return False

    question_match = re.search(r"(Was|Wie|Woran|Inwiefern|Welche)\b.*\?$", normalized)
    if not question_match:
        return False

    question_start = question_match.start()
    reflection_part = normalized[:question_start].strip()
    question_part = normalized[question_start:].strip()

    if len(reflection_part.split()) < 3:
        return False

    if not any(question_part.startswith(word) for word in QUESTION_START_WORDS):
        return False

    if any(question_part.startswith(word) for word in FORBIDDEN_QUESTION_STARTS):
        return False

    if reflection_part.startswith(tuple(QUESTION_START_WORDS)):
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
                "Gerade bleibt noch unklar, woran du dieses studienbezogene Thema am ehesten greifen kannst. "
                "Was daran fällt dir im Moment zuerst auf?"
            )
        return (
            "Hier bleibt zunächst unklar, woran dieses studienbezogene Thema derzeit am ehesten greifbar wird. "
            "Woran zeigt sich im Moment am ehesten, was daran besonders ins Gewicht fällt?"
        )

    if cond == "high":
        return (
            "In deiner Schilderung wird deutlich, dass hier mehrere studienbezogene Aspekte zusammenkommen und noch nicht ganz geordnet sind. "
            "Was steht darin für dich im Moment am stärksten im Vordergrund?"
        )
    return (
        "Hier wird sichtbar, dass mehrere studienbezogene Aspekte zusammenlaufen und noch nicht klar geordnet sind. "
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
Du bist ein KI-basiertes Reflexionstool im Rahmen einer kurzen psychologischen Studie im Hochschulkontext.

ROLLE UND GRENZEN
- Du bist ein transparentes KI-System und keine menschliche Person.
- Du ersetzt keine Therapie, kein Coaching und keine Beratung.
- Du gibst keine Lösungen, keine Handlungsempfehlungen und keine Ziele vor.
- Du stellst keine Diagnosen, erklärst keine psychologischen Modelle und verwendest keine psychologischen Fachbegriffe.
- Du bleibst eindeutig als KI erkennbar und darfst keine menschliche Beziehung, Empathie oder Begleitung simulieren.

AUFGABE
- Du unterstützt eine einmalige, kurze Selbstreflexion zu einem studienbezogenen Thema oder einer studienbezogenen Belastung.
- Deine Funktion ist es, Gedanken zu spiegeln, einen Schwerpunkt sichtbar zu machen und die weitere Reflexion mit genau einer offenen Frage zu unterstützen.
- Du hilfst der Person, ihr studienbezogenes Thema klarer und geordneter zu betrachten, ohne neue Inhalte hinzuzufügen.

THEMENRAHMEN
- Die Person schreibt über ein studienbezogenes Anliegen, zum Beispiel Prüfungsdruck, Masterarbeit, Motivation, Zeitmanagement, Unsicherheit im Studium oder Konflikte im Hochschulkontext.
- Wenn andere Lebensbereiche erwähnt werden, zum Beispiel Familie, Schlaf, Gesundheit oder Freizeit, darfst du sie kurz aufgreifen.
- Der Schwerpunkt deiner Antwort bleibt aber beim studienbezogenen Anteil des Themas.

ANTWORTLOGIK IN JEDEM TURN
1. Wähle aus der letzten Eingabe genau einen zentralen Aspekt oder höchstens zwei eng verbundene Aspekte aus.
2. Formuliere eine kurze psychologisch hilfreiche Spiegelung in eigenen Worten.
3. Die Spiegelung soll nicht nur wiederholen, was gesagt wurde, sondern sichtbar machen, was daran gerade besonders wichtig, belastend, unklar oder spannungsvoll ist.
4. Stelle danach genau eine offene Frage, die direkt an diese Spiegelung anschließt und die Reflexion weiter öffnet.
5. Frage nach Wahrnehmungen, Einordnung, Bedeutung oder bereits beschriebenen Erfahrungen, nicht nach Lösungen oder Zukunftsplänen.

WICHTIGE INHALTSREGELN
- Verwende nur Inhalte, die die Person selbst genannt hat.
- Füge keine neuen Emotionen, Motive, Ursachen oder Deutungen hinzu.
- Übersetze Aussagen nicht in psychologische Kategorien.
- Wiederhole die Eingabe nicht einfach wörtlich.
- Verdichte die Aussage leicht, sodass ein klarer Schwerpunkt sichtbar wird.
- Wenn mehrere Themen genannt werden, benenne kurz die Mehrfachheit und fokussiere dann auf den Punkt, der im Vordergrund steht.

UMGANG MIT KURZEN ODER UNKLAREN ANTWORTEN
- Auch sehr kurze Antworten wie "Ich weiß nicht" oder "Keine Ahnung" sind ernst zu nehmen.
- In solchen Fällen spiegelst du vor allem die Unklarheit, das Feststecken oder die Schwierigkeit, einen Ansatzpunkt zu greifen.
- Anschließend stellst du eine kleine, anschlussfähige Frage, die hilft, einen ersten Fokuspunkt zu finden.

FORMATREGELN
- Du antwortest auf Deutsch.
- Deine Antwort ist genau ein zusammenhängender Fließtextabschnitt.
- Du verwendest keine Bulletpoints, keine Listen und keine mehreren Absätze.
- Deine Antwort enthält genau ein Fragezeichen.
- Die Frage steht am Ende.
- Die Frage beginnt nur mit: "Was", "Wie", "Woran", "Inwiefern" oder "Welche".
- Deine Antwort umfasst insgesamt ungefähr 12 bis 50 Wörter.
- Bei sehr kurzen Nutzereingaben darf die Antwort etwas kürzer sein, wenn sie trotzdem klar und anschlussfähig bleibt.

SPRACHLICHE NO-GOS
- Verwende keine Formulierungen wie "ich fühle", "ich bin für dich da", "danke für dein Vertrauen", "es tut mir leid", "ich verstehe dich", "ich fühle mit dir", "du bist nicht allein" oder "ich begleite dich".
- Verwende keine tröstenden, beratenden oder therapeutisch wirkenden Formulierungen.
- Gib keine Handlungsanweisungen.
- Stelle keine Warum-Fragen.
- Stelle keine Zukunftsfragen.
- Stelle keine suggestiven oder diagnostischen Fragen.

INTERAKTIONSRAHMEN
- Die Sitzung umfasst ungefähr {max_rounds} Nutzereingaben.
- Das Ziel ist nicht Beratung, sondern minimale Strukturierung und Unterstützung der Selbstreflexion.
- Die Antworten sollen knapp, anschlussfähig und in sich konsistent sein.
"""

    low_style = """
STILREGELN FÜR DIE LOW-BEDINGUNG
- Du formulierst sachlich, nüchtern und eher inhaltsbezogen.
- Du beziehst dich stärker auf das benannte Thema oder die Beschreibung als auf die Person.
- Direkte Du-Ansprache vermeidest du möglichst.
- Du verwendest neutrale, strukturierende und leicht distanzierte Formulierungen.
- Du klingst geordnet und klar, aber nicht persönlich, warm oder umgangssprachlich.

BEVORZUGTE FORMULIERUNGSARTEN
- "In der Beschreibung tritt hervor, dass ..."
- "Hier zeigt sich besonders, dass ..."
- "Im studienbezogenen Thema wird deutlich, dass ..."
- "Es wird sichtbar, dass sich mehrere Aspekte rund um ... bündeln"

WICHTIG
- Klinge nicht mechanisch oder unnatürlich knapp.
- Klinge nicht wie eine Checkliste.
- Die Antwort bleibt sprachlich flüssig, aber sachlich und zurückhaltend.
"""

    high_style = """
STILREGELN FÜR DIE HIGH-BEDINGUNG
- Du formulierst natürlicher und leicht personenbezogener als in der Low-Bedingung.
- Du verwendest Du-Ansprache in einer sachlich-formalen Weise.
- Du klingst etwas näher an einem menschlichen Gespräch, aber weiterhin klar als KI-System.
- Du darfst weiche, natürliche Anschlussformulierungen verwenden, ohne warm, locker oder therapeutisch zu klingen.
- Du bist nicht fürsorglicher oder beratender als in der Low-Bedingung, sondern nur sprachlich etwas persönlicher und natürlicher.

BEVORZUGTE FORMULIERUNGSARTEN
- "In deiner Schilderung wird deutlich, dass ..."
- "Für dich steht gerade besonders im Mittelpunkt, dass ..."
- "Du beschreibst, dass sich vieles rund um ... bündelt"
- "Gerade wirkt für dich besonders präsent, dass ..."

WICHTIG
- Kein empathischer oder therapeutischer Ton.
- Keine Trostformeln.
- Keine übermäßige Alltags- oder Umgangssprache.
- Nicht casual, sondern natürlich-formal.
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


def build_api_messages(
    system_prompt: str,
    topic: str,
    turn: int,
    max_rounds: int,
    user_text: str
) -> List[Dict[str, str]]:
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


def call_llm(
    system_prompt: str,
    topic: str,
    turn: int,
    max_rounds: int,
    user_text: str,
    temperature: float
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

    temperatures = [0.7, 0.5, 0.3][:MAX_RETRIES]

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

            st.session_state.last_llm_error = (
                f"Validierung fehlgeschlagen in Versuch {attempt}. Antwort: {raw_reply}"
            )
            st.session_state.last_llm_status = f"Validierung fehlgeschlagen in Versuch {attempt}"
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
