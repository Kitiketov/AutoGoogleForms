# gform_groq_iter.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import os, json, re, time
from typing import Any, Dict, List, Optional
import requests

from parser import GFormParser                 # твой класс-парсер формы
from form_answer_builder import FormAnswerBuilder  # билдер ответов

# ⬇️ новое: контекст + память Q→A из отдельного модуля
from qa_context import QACache, make_section_context_map, RE_SECTION

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
CHAT_PATH = "/chat/completions"
MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
TEMPERATURE = 0.1

# ===== системный промпт читаем из файла =====
_SYSTEM_PROMPT_CACHE: Optional[str] = None
def load_system_prompt() -> str:
    """Читает системный промпт из файла (по умолчанию system_prompt.txt).
       Путь можно задать через SYSTEM_PROMPT_PATH. Результат кэшируется."""
    global _SYSTEM_PROMPT_CACHE
    if _SYSTEM_PROMPT_CACHE is not None:
        return _SYSTEM_PROMPT_CACHE
    path = os.getenv("SYSTEM_PROMPT_PATH", "system_prompt.txt")
    try:
        with open(path, "r", encoding="utf-8") as f:
            txt = f.read().strip()
            if not txt:
                raise ValueError("system_prompt.txt пустой")
            _SYSTEM_PROMPT_CACHE = txt
    except Exception as e:
        _SYSTEM_PROMPT_CACHE = (
            "Ты математик. Отвечай кратко и точно. Верни ТОЛЬКО JSON по инструкции."
        )
        print(f"[warn] Не удалось загрузить системный промпт из {path}: {e}. "
              f"Использую дефолтный короткий промпт.")
    return _SYSTEM_PROMPT_CACHE


# ----------- Клиент Groq -----------

class GroqClient:
    def __init__(self, api_key: str, base_url: str = GROQ_BASE_URL, timeout: int = 60):
        self.api_key = (api_key or "").strip().strip('"').strip("'")
        if not self.api_key:
            raise SystemExit("Не установлен GROQ_API_KEY.")
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        })

    def _raise_for_error(self, r: requests.Response):
        if r.status_code >= 400:
            try:
                j = r.json()
                msg = j.get("error") or j.get("message") or j
            except Exception:
                msg = r.text
            raise RuntimeError(f"Groq API error {r.status_code}: {msg}")

    def list_models(self) -> list:
        url = f"{self.base_url}/models"
        r = self.session.get(url, timeout=self.timeout)
        self._raise_for_error(r)
        return r.json().get("data", [])

    def chat(self, messages: List[Dict[str, str]], model: str = MODEL,
             temperature: float = TEMPERATURE, **extra) -> str:
        url = f"{self.base_url}{CHAT_PATH}"
        payload = {"model": model, "messages": messages, "temperature": temperature}
        if extra:
            payload.update(extra)
        r = self.session.post(url, json=payload, timeout=self.timeout)
        self._raise_for_error(r)
        data = r.json()
        return data["choices"][0]["message"]["content"]


# ----------- Утилиты сопоставления вариантов -----------

_ws = re.compile(r"\s+", re.S)
def _norm(s: str) -> str:
    s = (s or "").strip().lower().replace("\xa0", " ")
    return _ws.sub(" ", s)

def pick_single_option(answer_text: str, options: List[str]) -> Optional[str]:
    if not options:
        return None
    norm_opts = [_norm(o) for o in options]
    ans = _norm(answer_text)
    if ans.isdigit():
        k = int(ans)
        if 1 <= k <= len(options):
            return options[k-1]
    for i, no in enumerate(norm_opts):
        if ans == no:
            return options[i]
    m = re.search(r"[:\-–]\s*(.+)$", ans)
    if m:
        ans2 = m.group(1).strip().strip("\"'«»")
        for i, no in enumerate(norm_opts):
            if ans2 == no:
                return options[i]
        ans = ans2
    hits = [i for i, no in enumerate(norm_opts) if ans in no or no in ans]
    if len(hits) == 1:
        return options[hits[0]]
    return None

def pick_multi_options(answer_text: str, options: List[str]) -> Optional[List[str]]:
    if not options:
        return None
    ans = answer_text.strip()
    try:
        val = json.loads(ans)
        if isinstance(val, list):
            picked = []
            for item in val:
                match = pick_single_option(str(item), options)
                if match and match not in picked:
                    picked.append(match)
            return picked or None
    except Exception:
        pass
    parts = [p.strip() for p in re.split(r"[,;/\n]+", ans) if p.strip()]
    if not parts:
        return None
    picked = []
    for p in parts:
        match = pick_single_option(p, options)
        if match and match not in picked:
            picked.append(match)
    return picked or None


# ----------- Построение промпта -----------

def build_messages_for_question(
    q: Dict[str, Any],
    *,
    section_ctx: str = "",
    history_text: str = ""
) -> List[Dict[str, str]]:
    """Формируем сообщения: system — из файла, user — контекст + вопрос."""
    system_text = load_system_prompt()

    qtext = q.get("text") or ""
    qtype = (q.get("type") or "").lower()
    opts: List[str] = q.get("choices_or_rows") or []

    blocks = []
    if section_ctx:
        blocks.append("Общий контекст:\n" + section_ctx.strip())
    if history_text:
        blocks.append(history_text.strip())
    blocks.append("Вопрос:\n" + qtext.strip())
    body = "\n\n".join(blocks) + "\n\n"

    if qtype in ("multiple_choice", "dropdown", "checkboxes") and opts:
        instruct = (
            body +
            "Варианты:\n" + "\n".join(f"- {o}" for o in opts) + "\n\n"
            "Верни только JSON.\n"
            "Одиночный выбор: {\"answer\": \"ОДИН_ИЗ_ВАРИАНТОВ_ТОЧНО_КАК_В_СПИСКЕ\"}\n"
            "Множественный: {\"answer\": [\"ВАР_1\", \"ВАР_2\"]}"
        )
    else:
        instruct = body + "Верни только JSON: {\"answer\": \"КОРОТКИЙ_ТЕКСТ\"}"

    return [
        {"role": "system", "content": system_text},
        {"role": "user", "content": instruct},
    ]


def extract_answer_from_llm(raw_content: str) -> Optional[Any]:
    try:
        data = json.loads(raw_content)
        return data.get("answer")
    except Exception:
        m = re.search(r"\{.*\}", raw_content or "", flags=re.S)
        if m:
            try:
                return json.loads(m.group(0)).get("answer")
            except Exception:
                return None
    return None


# ----------- Основной запуск -----------

def answer_form_with_groq(url: str, delay_sec: float = 0.3, do_submit: bool = False) -> None:
    parsed = GFormParser(url).parse()
    builder = FormAnswerBuilder(parsed, strict=True)
    client = GroqClient(api_key=os.getenv("GROQ_API_KEY", ""))

    # контекст секций и «память» предыдущих Q→A
    section_ctx_map = make_section_context_map(parsed.get("questions", []))
    qa_cache = QACache()
    RESET_HISTORY_ON_NEW_SECTION = True

    # sanity-check моделей и chat
    try:
        models = client.list_models()
        if models:
            print("Модели (срез):", [m.get("id", m) for m in models[:8]])
    except Exception as e:
        print("list_models:", e)

    try:
        test = client.chat(
            messages=[
                {"role": "system", "content": "Отвечай одним словом."},
                {"role": "user", "content": "ping"},
            ],
            model=MODEL, temperature=0.0,
        )
        print("Проверка chat OK:", test)
    except Exception as e:
        raise SystemExit(f"Chat не работает: {e}")

    # Итерация по вопросам
    for q in parsed.get("questions", []):
        eid = q.get("entry_id")
        if not eid:
            continue

        q_text = (q.get("text") or "")
        # очистка истории при новой «большой» секции (1., 2., 3., ...)
        if RESET_HISTORY_ON_NEW_SECTION and RE_SECTION.match(q_text or ""):
            qa_cache.clear()

        qtype = (q.get("type") or "").lower()
        opts: List[str] = q.get("choices_or_rows") or []

        section_ctx = section_ctx_map.get(str(eid), "")
        history_text = qa_cache.as_text()

        try:
            content = client.chat(
                messages=build_messages_for_question(
                    q, section_ctx=section_ctx, history_text=history_text
                ),
                model=MODEL,
                temperature=TEMPERATURE
            )
        except Exception as e:
            print(f"[{eid}] Ошибка запроса к Groq: {e}")
            continue

        ans = extract_answer_from_llm(content)
        if ans is None:
            print(f"[{eid}] Не удалось извлечь JSON-ответ: {content!r}")
            continue

        value: Optional[Any] = None
        if qtype == "checkboxes" and opts:
            value = (pick_multi_options(ans, opts) if isinstance(ans, str)
                     else [o for o in (pick_single_option(str(x), opts) for x in (ans if isinstance(ans, list) else [])) if o])
        elif qtype in ("multiple_choice", "dropdown") and opts:
            value = (pick_single_option(str(ans[0]), opts) if isinstance(ans, list) and ans
                     else pick_single_option(str(ans), opts))
        else:
            value = str(ans)

        # >>> NEW: поддержка "Другое" (свободный ответ), если не заматчилось
        if (not value or (isinstance(value, list) and not value)) and q.get("other_allowed"):
            # берём текст из ans
            other_text = None
            if isinstance(ans, str):
                other_text = ans.strip()
            elif isinstance(ans, list) and ans:
                other_text = str(ans[0]).strip()
            if other_text:
                # передадим в билдер спец-структуру с ключом "__other__"
                value = {"__other__": other_text}

        if not value or (isinstance(value, list) and not value):
            print(f"[{eid}] Ответ получен, но не маппится на варианты. Пропуск.")
            continue

        try:
            builder.set_answer(eid, value)
            qa_cache.add(q_text, value)  # пополняем «память»

            preview = value if isinstance(value, str) else ", ".join(map(str, value))
            short_q = q_text.splitlines()[0][:80]
            print(f"[{eid}] OK -> {preview}    ({short_q}...)")
        except ValueError as e:
            print(f"[{eid}] Валидатор отклонил ответ: {e}")
            continue

        if delay_sec > 0:
            time.sleep(delay_sec)

    action, pairs = builder.build_pairs()
    print("\nСформирован payload для POST", action)
    for k, v in pairs:
        if k.startswith("entry."):
            print(k, "->", v)

    if do_submit:
        print("\nОтправляю форму…")
        r = builder.submit(referer=url)
        try:
            r.raise_for_status()
            print("OK,", r.status_code)
        except Exception as e:
            print("Ошибка отправки:", e)
            print("Body snippet:", r.text[:500])


if __name__ == "__main__":
    URL = "https://docs.google.com/forms/d/1z8uChzEqNhtdtYp4WNO8BoRuWHXVjeNyijQMiaG3QHc/viewform?edit_requested=true"
    # перед запуском: set GROQ_API_KEY (и опц. GROQ_MODEL, SYSTEM_PROMPT_PATH, QA_CACHE_PATH/QA_CACHE_MAX_*)
    answer_form_with_groq(URL, delay_sec=0.3, do_submit=False)
