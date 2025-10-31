# gform_parser_class.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import json, re, sys, html
import os
from typing import Any, Dict, List, Optional, Tuple, Iterable
import requests

URL = "https://docs.google.com/forms/d/1z8uChzEqNhtdtYp4WNO8BoRuWHXVjeNyijQMiaG3QHc/viewform?edit_requested=true"
OUTPUT_PATH = None
PRETTY = True
# --- ДОБАВЬ к импортам ---

class GFormParser:
    DEFAULT_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/123.0 Safari/537.36")

    TYPE_MAP = {
        0: "short_answer",
        1: "paragraph",
        2: "multiple_choice",
        3: "dropdown",
        4: "checkboxes",
        5: "linear_scale",
        7: "date",
        8: "time",
    }


    def __init__(self, url: str, user_agent: Optional[str] = None, timeout: int = 20):
        self.url = url
        self.user_agent = user_agent or self.DEFAULT_UA
        self.timeout = timeout

    # -------------------- Публичные методы --------------------

    def parse(self) -> Dict[str, Any]:
        html_text = self._fetch_html(self.url)
        meta = self._extract_form_meta(html_text, self.url)
        data = self._extract_fb_payload(html_text)

        title = self._dig(data, [1, 8]) if isinstance(self._dig(data, [1, 8]), str) else None
        if not title:
            for node in self._walk_lists(data):
                if isinstance(node, list):
                    for v in node:
                        if isinstance(v, str) and v.strip():
                            title = v
                            break
                if title:
                    break

        description = self._dig(data, [1, 0]) if isinstance(self._dig(data, [1, 0]), str) else None

        items_raw = self._guess_items_root(data)
        label_map = meta.get("label_to_entry", {}) or {}
        entry_ids = meta.get("entry_ids", []) or []
        eid_idx = 0

        def match_entry_by_label(qtext: str) -> Optional[str]:
            key = self._normalize(qtext)
            if key in label_map:
                return label_map[key]
            for k, eid in label_map.items():
                if key.startswith(k) or k.startswith(key) or key in k or k in key:
                    return eid
            return None

        questions: List[Dict[str, Any]] = []
        for it in items_raw:
            if not isinstance(it, list):
                continue

            # --- текст вопроса ---
            text = None
            if isinstance(self._dig(it, [1]), str) and self._dig(it, [1]).strip():
                text = self._dig(it, [1]).strip()
            elif isinstance(self._dig(it, [0, 1]), str) and self._dig(it, [0, 1]).strip():
                text = self._dig(it, [0, 1]).strip()
            else:
                for node in self._walk_lists(it):
                    if isinstance(node, list):
                        for v in node:
                            if isinstance(v, str) and v.strip():
                                text = v.strip()
                                break
                    if text:
                        break
            if not text:
                continue

            qtype = self._question_type(it)
            required = self._is_required(it)
            choices, cols = self._extract_choices(it)

            q: Dict[str, Any] = {"text": html.unescape(text), "type": qtype}
            if required is not None:
                q["required"] = required
            if choices:
                q["choices_or_rows"] = choices
            if cols:
                q["columns"] = cols

            # --- назначаем entry_id: HTML-порядок -> по label -> из FB JSON ---
            eid: Optional[str] = None
            if eid_idx < len(entry_ids):
                eid = entry_ids[eid_idx]
                eid_idx += 1

            if not eid:
                eid = match_entry_by_label(q["text"])

            if not eid:
                eid = self._extract_entry_id_from_item_fb(it)

            q["entry_id"] = eid  # может остаться None

            # --- поддержка "свободного ответа" ("Другое") для choice/dropdown/checkboxes ---
            # если среди вариантов встретилась пустая строка, считаем что разрешён свободный ответ
            if q.get("type") in ("multiple_choice", "dropdown", "checkboxes", "choice"):
                choices = q.get("choices_or_rows") or []
                if isinstance(choices, list) and any((isinstance(c, str) and c.strip() == "") for c in choices):
                    # вычистим пустые варианты из списка
                    cleaned = [c for c in choices if not (isinstance(c, str) and c.strip() == "")]
                    q["choices_or_rows"] = cleaned
                    # отметим, что доступен "другой" ответ
                    q["other_allowed"] = True
                    # стандартные для Google Forms значения для отправки "Другое"
                    # - нужно выбрать специальное значение "__other_option__" в entry.<id>
                    # - и передать текст по ключу entry.<id>.other_option_response
                    q["other_value"] = "__other_option__"
                    if eid:
                        q["other_response_key"] = f"entry.{eid}.other_option_response"

            questions.append(q)

        return {
            "title": title,
            "description": description,
            "questions_count": len(questions),
            "questions": questions,
            "meta": {
                "action": meta.get("action", ""),
                "fbzx": meta.get("fbzx", ""),
                "entry_ids": entry_ids,
                "label_map_size": len(label_map),
                "note": "Если entry_id пустой, открой публичную ссылку вида .../viewform без edit_requested=true.",
            },
        }

    def parse_to_file(self, out_path: str, pretty: bool = True) -> None:
        data = self.parse()
        text = json.dumps(data, ensure_ascii=False, indent=2 if pretty else None)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(text)

    # -------------------- Внутренние утилиты --------------------

    def _fetch_html(self, url: str) -> str:
        r = requests.get(url, headers={"User-Agent": self.user_agent}, timeout=self.timeout)
        r.raise_for_status()
        return r.text

    def _extract_fb_payload(self, page_html: str) -> Any:
        m = re.search(r"var\s+FB_PUBLIC_LOAD_DATA_\s*=\s*(\[.+?\]);\s*</script>", page_html, flags=re.S)
        if not m:
            m = re.search(r"FB_PUBLIC_LOAD_DATA_\s*=\s*(\[.+?\]);", page_html, flags=re.S)
        if not m:
            raise RuntimeError("Не нашёл служебные данные формы (FB_PUBLIC_LOAD_DATA_). "
                               "Форма может быть закрыта/требовать вход или изменилась разметка.")
        return json.loads(m.group(1))

    def _extract_form_meta(self, html_text: str, viewform_url: str) -> Dict[str, Any]:
        m_action = re.search(r'<form[^>]+action="([^"]+/formResponse)"', html_text)
        action = m_action.group(1) if m_action else viewform_url.replace("/viewform", "/formResponse").split("?", 1)[0]

        m_fbzx = re.search(r'name="fbzx"\s+value="([^"]+)"', html_text)
        fbzx = m_fbzx.group(1) if m_fbzx else ""

        # entry.<id> по порядку
        entries = re.findall(r'name="entry\.(\d+)"', html_text)
        seen = set(); entry_ids: List[str] = []
        for e in entries:
            if e not in seen:
                entry_ids.append(e); seen.add(e)

        # карта label/placeholder -> entry
        label_to_entry: Dict[str, str] = {}
        for m in re.finditer(r'name="entry\.(\d+)"[^>]*?\saria-label="([^"]+)"', html_text):
            entry, label = m.group(1), m.group(2)
            label_to_entry.setdefault(self._normalize(label), entry)
        for m in re.finditer(r'name="entry\.(\d+)"[^>]*?\splaceholder="([^"]+)"', html_text):
            entry, label = m.group(1), m.group(2)
            label_to_entry.setdefault(self._normalize(label), entry)

        return {"action": action, "fbzx": fbzx, "entry_ids": entry_ids, "label_to_entry": label_to_entry}

    def _guess_items_root(self, data: Any) -> List:
        candidate = self._dig(data, [1, 1])
        if isinstance(candidate, list) and candidate:
            return candidate
        best: List = []
        for node in self._walk_lists(data):
            if not node or not isinstance(node, list):
                continue
            str_count = sum(1 for v in node if isinstance(v, list) and any(isinstance(x, str) and x.strip() for x in v))
            if str_count >= max(2, len(node)//3) and len(node) > len(best):
                best = node
        if best:
            return best
        raise RuntimeError("Не удалось найти список вопросов в данных формы.")

    def _extract_choices(self, item: list) -> Tuple[Optional[List[str]], Optional[List[str]]]:
        node = self._dig(item, [4, 0, 1])
        if isinstance(node, list) and node and all(isinstance(x, list) for x in node):
            choices = []
            for ch in node:
                if ch and isinstance(ch[0], str):
                    choices.append(ch[0])
                elif len(ch) > 1 and isinstance(ch[1], str):
                    choices.append(ch[1])
            cols_node = self._dig(item, [4, 0, 2])
            cols = None
            if isinstance(cols_node, list) and cols_node and all(isinstance(x, list) for x in cols_node):
                cols = []
                for c in cols_node:
                    if c and isinstance(c[0], str):
                        cols.append(c[0])
                    elif len(c) > 1 and isinstance(c[1], str):
                        cols.append(c[1])
            return (choices or None), cols

        # fallback: «список списков со строками»
        best_list = None
        for node in self._walk_lists(item):
            if isinstance(node, list) and node and all(isinstance(x, list) for x in node):
                sample = node[0]
                if sample and any(isinstance(y, str) and y.strip() for y in sample):
                    if best_list is None or len(node) > len(best_list):
                        best_list = node
        if best_list:
            choices = []
            for ch in best_list:
                if ch and isinstance(ch[0], str):
                    choices.append(ch[0])
                elif len(ch) > 1 and isinstance(ch[1], str):
                    choices.append(ch[1])
            if choices:
                return choices, None
        return None, None

    def _is_required(self, item: list) -> Optional[bool]:
        node = self._dig(item, [4, 0])
        if isinstance(node, list):
            for v in node[::-1]:
                if isinstance(v, bool):
                    return v
                if isinstance(v, list):
                    for vv in v[::-1]:
                        if isinstance(vv, bool):
                            return vv
        for node in self._walk_lists(item):
            for v in node:
                if isinstance(v, bool):
                    return v
        return None

    def _question_type(self, item: list) -> str:
        t = self._dig(item, [3])
        if isinstance(t, int) and t in self.TYPE_MAP:
            return self.TYPE_MAP[t]
        choices, cols = self._extract_choices(item)
        if choices and cols:
            return "grid"
        if choices:
            return "choice"
        return "text"

    def _extract_entry_id_from_item_fb(self, item: list) -> Optional[str]:
        for path in ([4,0,0], [4,0,3,0], [4,0,0,0], [0]):
            v = self._dig(item, path)
            if isinstance(v, int) and v >= 10_000:
                return str(v)
            if isinstance(v, list):
                for x in v:
                    if isinstance(x, int) and x >= 10_000:
                        return str(x)
        best = None
        for node in self._walk_lists(item):
            for val in node:
                if isinstance(val, int) and val >= 10_000:
                    s = str(val)
                    if best is None or len(s) > len(best):
                        best = s
        return best

    # -------------------- статические утилиты --------------------

    @staticmethod
    def _dig(x: Any, path: List[int]) -> Any:
        cur = x
        for p in path:
            if not isinstance(cur, list) or p < 0 or p >= len(cur):
                return None
            cur = cur[p]
        return cur

    @staticmethod
    def _walk_lists(x: Any):
        if isinstance(x, list):
            yield x
            for v in x:
                yield from GFormParser._walk_lists(v)

    @staticmethod
    def _normalize(s: str) -> str:
        s = s or ""
        s = s.replace("\xa0", " ")
        s = re.sub(r"\s+", " ", s, flags=re.S)
        s = s.strip().lower()
        return s


# -------------------- запуск из консоли --------------------

if __name__ == "__main__":
    try:
        parser = GFormParser(URL)
        result = parser.parse()
    except Exception as e:
        print(f"[!] Ошибка: {e}", file=sys.stderr)
        sys.exit(2)

    data = json.dumps(result, ensure_ascii=False, indent=2 if PRETTY else None)
    if OUTPUT_PATH:
        with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
            f.write(data)
        print(f"Готово: {OUTPUT_PATH}")
    else:
        print(data)
