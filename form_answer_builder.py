# form_answer_builder.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import re
from typing import Any, Dict, Iterable, Iterator, List, Tuple, Union, Optional
import requests

Answer = Union[str, int, float, List[Union[str, int, float]], Tuple[Union[str, int, float], ...]]

DEFAULT_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
              "(KHTML, like Gecko) Chrome/123.0 Safari/537.36")
# form_answer_builder.py
# -*- coding: utf-8 -*-

import requests
from typing import Any, Dict, List, Tuple, Optional

class FormAnswerBuilder:
    def __init__(self, parsed: Dict[str, Any], strict: bool = True, ua: str = None, timeout: int = 20):
        self.parsed = parsed
        self.strict = strict
        self.ua = ua or "Mozilla/5.0"
        self.timeout = timeout

        self.action: str = parsed.get("meta", {}).get("action", "")
        self.fbzx: str = parsed.get("meta", {}).get("fbzx", "")
        self.q_by_id: Dict[str, Dict[str, Any]] = {}

        for q in parsed.get("questions", []):
            eid = q.get("entry_id")
            if eid:
                self.q_by_id[str(eid)] = q

        # ответы пользователя: entry_id -> value
        # value может быть:
        #  - str (обычный выбор/текст),
        #  - list[str] (чекбоксы),
        #  - dict {"__other__": "text"} (Другое),
        #  - dict {"__other__": "text", "__selected__": [варианты]} (чекбоксы с «Другое»)
        self.answers: Dict[str, Any] = {}

    def set_answer(self, entry_id: str, value: Any) -> None:
        entry_id = str(entry_id)
        q = self.q_by_id.get(entry_id)
        if not q:
            if self.strict:
                raise ValueError(f"Неизвестный entry_id: {entry_id}")
            return

        qtype = (q.get("type") or "").lower()
        choices = q.get("choices_or_rows") or []
        other_allowed = bool(q.get("other_allowed"))

        # если сразу пришёл dict с "__other__" — принимаем без проверок
        if isinstance(value, dict) and "__other__" in value:
            self.answers[entry_id] = {"__other__": str(value["__other__"])}
            # если вместе передали предварительно выбранные чекбоксы
            if "__selected__" in value and isinstance(value["__selected__"], list):
                self.answers[entry_id]["__selected__"] = [str(x) for x in value["__selected__"]]
            return

        if qtype in ("multiple_choice", "dropdown", "choice"):
            if isinstance(value, str) and value in choices:
                self.answers[entry_id] = value
                return
            if other_allowed and isinstance(value, str):
                # свободный ответ
                self.answers[entry_id] = {"__other__": value}
                return
            if self.strict:
                raise ValueError(f"Ответ '{value}' не входит в варианты и 'Другое' не разрешено для entry.{entry_id}")
            return

        if qtype == "checkboxes":
            # нормализуем к списку
            vals = value if isinstance(value, list) else [value]
            vals = [str(v) for v in vals if v is not None]

            selected: List[str] = []
            others: List[str] = []
            for v in vals:
                if v in choices:
                    selected.append(v)
                else:
                    others.append(v)

            if others:
                if other_allowed:
                    ans: Dict[str, Any] = {"__other__": "; ".join(others)}
                    if selected:
                        ans["__selected__"] = selected
                    self.answers[entry_id] = ans
                elif self.strict:
                    raise ValueError(f"Часть ответов {others} не входят в варианты и 'Другое' не разрешено для entry.{entry_id}")
                else:
                    self.answers[entry_id] = selected  # тихо отбрасываем лишнее
            else:
                self.answers[entry_id] = selected
            return

        # все прочие типы — просто строка
        self.answers[entry_id] = str(value)

    def build_pairs(self) -> Tuple[str, List[Tuple[str, str]]]:
        pairs: List[Tuple[str, str]] = []
        if self.fbzx:
            pairs.append(("fbzx", self.fbzx))

        for eid, val in self.answers.items():
            q = self.q_by_id.get(eid) or {}
            key = f"entry.{eid}"
            other_key = q.get("other_response_key") or f"{key}.other_option_response"
            other_value_flag = q.get("other_value", "__other_option__")
            qtype = (q.get("type") or "").lower()

            if isinstance(val, dict) and "__other__" in val:
                # выбрана опция "Другое"
                pairs.append((key, str(other_value_flag)))
                pairs.append((other_key, str(val["__other__"])))
                # чекбоксы могли иметь и обычные выбранные варианты
                for sel in val.get("__selected__", []):
                    pairs.append((key, str(sel)))
                continue

            if qtype == "checkboxes" and isinstance(val, list):
                for item in val:
                    pairs.append((key, str(item)))
                continue

            # обычный случай
            pairs.append((key, str(val)))

        return self.action, pairs

    def submit(self, referer: Optional[str] = None) -> requests.Response:
        action, pairs = self.build_pairs()
        data = {}
        # Google Forms допускает повторяющиеся ключи для чекбоксов;
        # requests это умеет, если передать список tuples, поэтому
        # здесь НЕ сводим в dict, а шлём как есть:
        headers = {"User-Agent": self.ua}
        if referer:
            headers["Referer"] = referer
        return requests.post(action, data=pairs, headers=headers, timeout=self.timeout)


    # --------------------------- Вспомогательное ---------------------------

    def _iter_fields(self) -> Iterator[Tuple[str, str]]:
        """
        Выдаёт пары ('entry.<id>', 'value') для всех установленных ответов.
        Валидирует multiple_choice/dropdown/checkboxes при strict=True.
        """
        for eid, raw in self._answers.items():
            q = self._eid_to_q.get(str(eid)) or {}
            qtype = (q.get("type") or "").lower()
            options = set(q.get("choices_or_rows") or [])

            def _emit(val):
                yield (f"entry.{eid}", str(val))

            # чекбоксы
            if qtype == "checkboxes":
                vals = raw if isinstance(raw, (list, tuple, set)) else [raw]
                for v in vals:
                    if self.strict and options and str(v) not in options:
                        raise ValueError(self._bad_opt_msg(q, v, options))
                    yield from _emit(v)
                continue

            # одиночный выбор
            if qtype in ("multiple_choice", "dropdown"):
                if self.strict and options and str(raw) not in options:
                    raise ValueError(self._bad_opt_msg(q, raw, options))
                yield from _emit(raw)
                continue

            # прочие типы — как строку
            yield from _emit(raw)

    def available_options(self, key: Union[str, int]) -> List[str]:
        """
        Вернёт список допустимых вариантов для вопроса (по entry_id, индексу или тексту).
        """
        q = self._resolve_question(key)
        return list(q.get("choices_or_rows") or [])

    # --------------------------- приватные утилиты ---------------------------

    def _resolve_question(self, key: Union[str, int]) -> Dict[str, Any]:
        if isinstance(key, (int,)):
            eid = self._index_to_eid.get(int(key))
            if not eid:
                raise KeyError(f"Нет entry_id у вопроса с индексом {key}")
            return self._eid_to_q.get(str(eid), {})
        # текст или entry_id
        if isinstance(key, str) and not key.isdigit():
            eid = self._text_to_eid.get(self._norm(key))
            if not eid:
                raise KeyError(f"Не найден вопрос с текстом: {key!r}")
            return self._eid_to_q.get(str(eid), {})
        # entry_id
        return self._eid_to_q.get(str(key), {})

    def _bad_opt_msg(self, q: Dict[str, Any], got: Any, options: Iterable[str]) -> str:
        return (f"Неверное значение '{got}' для вопроса: {q.get('text')!r}. "
                f"Допустимые: {sorted(map(str, options))}")

    def _action_fbzx(self) -> Tuple[str, str]:
        meta = self.parsed.get("meta", {}) or {}
        return (meta.get("action") or "", meta.get("fbzx") or "")

    @staticmethod
    def _norm(s: str) -> str:
        return re.sub(r"\s+", " ", (s or "")).strip().lower()
    def __iter__(self):
        """Итерируемся по всем вопросам как по словарям из parsed['questions']."""
        yield from (self.parsed.get("questions", []) or [])

    def __len__(self):
        return len(self.parsed.get("questions", []) or [])

    def iter_unanswered(self):
        """
        Итерируемся по вопросам, у которых есть entry_id и пока нет ответа.
        Удобно для поэтапного заполнения.
        """
        for q in (self.parsed.get("questions", []) or []):
            eid = q.get("entry_id")
            if not eid:
                continue
            if str(eid) not in self._answers:
                yield q