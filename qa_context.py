# qa_context.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import os, json, re, threading, pathlib
from typing import Any, Dict, List, Optional

__all__ = [
    "QACache",
    "RE_SECTION",
    "RE_SUBPART",
    "make_section_context_map",
    "extract_section_intro",
]

# ============ КЭШ Q→A ДЛЯ КОНТЕКСТА ============
class QACache:
    """
    Хранит последние Q→A для подмешивания в промпт.
    Настройки через env:
      QA_CACHE_PATH=/tmp/qa.json   (опц. путь для сохранения)
      QA_CACHE_MAX_PAIRS=5         (по умолч. 5)
      QA_CACHE_MAX_CHARS=900       (лимит символов)
    """
    def __init__(self):
        self.path = pathlib.Path(os.getenv("QA_CACHE_PATH", "")) if os.getenv("QA_CACHE_PATH") else None
        self.max_pairs = int(os.getenv("QA_CACHE_MAX_PAIRS", "5"))
        self.max_chars = int(os.getenv("QA_CACHE_MAX_CHARS", "900"))
        self._lock = threading.Lock()
        self.pairs: List[Dict[str, str]] = []
        self._load()

    def _load(self):
        if not self.path or not self.path.exists():
            return
        try:
            self.pairs = json.load(self.path.open("r", encoding="utf-8"))
        except Exception:
            self.pairs = []

    def _save(self):
        if not self.path:
            return
        try:
            with self._lock, self.path.open("w", encoding="utf-8") as f:
                json.dump(self.pairs, f, ensure_ascii=False)
        except Exception:
            pass

    def clear(self):
        self.pairs = []
        self._save()

    def _len_chars(self) -> int:
        return sum(len(p.get("q","")) + len(p.get("a","")) + 6 for p in self.pairs)

    def add(self, q_text: str, answer: Any):
        q = (q_text or "").strip().replace("\n", " ")
        a = (", ".join(map(str, answer)) if isinstance(answer, list) else str(answer)).strip().replace("\n", " ")
        self.pairs.append({"q": q[:200], "a": a[:200]})
        # урежем по числу пар
        while len(self.pairs) > self.max_pairs:
            self.pairs.pop(0)
        # урежем по символам
        while self._len_chars() > self.max_chars and self.pairs:
            self.pairs.pop(0)
        self._save()

    def as_text(self) -> str:
        if not self.pairs:
            return ""
        lines = [f"- Q: {p['q']} | A: {p['a']}" for p in self.pairs]
        return "Предыдущие вопросы и ответы (для контекста):\n" + "\n".join(lines)

# ============ СЕКЦИОННЫЙ КОНТЕКСТ (общий стем) ============
RE_SECTION = re.compile(r"^\s*(\d+)[\.\)]\s*", re.I)          # "1." / "2)"
RE_SUBPART = re.compile(r"^\s*([a-zа-я])[\)\.]\s*", re.I)     # "a)" / "б)"
RE_SPLIT_FIRST_SUB = re.compile(r"\n\s*[a-zа-я][\)\.]\s+", re.I)

def extract_section_intro(text: str) -> str:
    """Возвращает общий стем секции: часть до первого подпункта 'a)/б)/в)'."""
    m = RE_SECTION.match(text or "")
    if not m:
        return ""
    parts = RE_SPLIT_FIRST_SUB.split(text, maxsplit=1)
    return parts[0].strip() if len(parts) >= 2 else (text or "").strip()

def make_section_context_map(questions: List[Dict[str, Any]]) -> Dict[str, str]:
    """
    Возвращает словарь {entry_id: контекст}, который нужно приклеивать к подпунктам.
    Логика:
      - встречаем секцию (начинается с '1.'/'2.'/'3.' ...) — запоминаем как current.
      - все последующие вопросы, начинающиеся на 'a)/б)/в)' получают current как контекст,
        пока не встретим новую секцию.
    """
    ctx_map: Dict[str, str] = {}
    current = ""
    for q in questions:
        t = (q.get("text") or "").strip()
        eid = q.get("entry_id")
        if not t:
            continue
        if RE_SECTION.match(t):
            current = extract_section_intro(t) or t
            if eid and RE_SUBPART.search(t):
                ctx_map[str(eid)] = current
            continue
        if RE_SUBPART.match(t) and current and eid:
            ctx_map[str(eid)] = current
    return ctx_map
