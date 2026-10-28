import json
import re
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List


REPO_ROOT = Path(__file__).resolve().parents[2]
POEM_DATA_FILE = REPO_ROOT / "hexi_poems_clean_simplified.json"
TEXT_NORMALIZE_RE = re.compile(r"[，。！？；：、“”‘’（）()《》〈〉【】〔〕—\-—\s]+")

THEME_RULES: List[tuple[str, tuple[str, ...]]] = [
    ("边塞诗", ("边塞", "玉门关", "阳关", "嘉峪关", "凉州", "羌笛", "塞下", "关山", "烽火", "戍", "军", "胡", "大漠", "沙场")),
    ("送别诗", ("送别", "送君", "别君", "离别", "折柳", "劝君", "故人", "西出阳关", "送友", "赠别")),
    ("山水诗", ("山", "水", "江", "河", "湖", "溪", "泉", "瀑", "川", "岭", "峰", "岳")),
    ("田园诗", ("田园", "农", "桑", "麻", "稻", "村", "野", "牛", "耕", "渔", "庐", "柴门")),
    ("咏史诗", ("咏史", "古人", "汉", "秦", "魏", "晋", "唐", "将军", "故国", "兴亡", "古", "旧事")),
    ("咏物诗", ("梅", "兰", "竹", "菊", "松", "荷", "莲", "雪", "马", "鹤", "雁", "柳", "花")),
]

SPACE_TYPE_RULES: List[tuple[str, tuple[str, ...]]] = [
    ("关隘", ("玉门关", "阳关", "嘉峪关", "关城", "关隘", "关山", "关河")),
    ("驿站", ("驿", "站", "邮亭", "亭", "驿路", "驿站")),
    ("边塞", ("边塞", "塞下", "凉州", "河西", "西域", "大漠", "沙场", "烽火", "戍楼")),
    ("行旅路线", ("行旅", "征途", "道路", "古道", "西行", "远行", "归程", "出塞", "入关")),
    ("都城", ("长安", "洛阳", "汴京", "京城", "帝京", "宫", "阙")),
    ("江南", ("江南", "吴", "越", "姑苏", "金陵", "扬州", "杭州")),
    ("山川", ("山", "水", "江", "河", "湖", "溪", "泉", "岭", "峰", "川")),
]

PLACE_GAZETTEER: List[Dict[str, Any]] = [
    {"name": "玉门关", "aliases": ("玉门关",), "lng": 93.8631, "lat": 40.3603},
    {"name": "阳关", "aliases": ("阳关",), "lng": 94.6767, "lat": 39.9398},
    {"name": "敦煌", "aliases": ("敦煌", "沙州"), "lng": 94.6619, "lat": 40.1421},
    {"name": "瓜州", "aliases": ("瓜州",), "lng": 95.7826, "lat": 40.5203},
    {"name": "玉门", "aliases": ("玉门",), "lng": 97.0450, "lat": 40.2917},
    {"name": "酒泉", "aliases": ("酒泉", "肃州"), "lng": 98.5174, "lat": 39.7432},
    {"name": "张掖", "aliases": ("张掖", "甘州"), "lng": 100.4498, "lat": 38.9259},
    {"name": "武威", "aliases": ("武威", "凉州"), "lng": 102.6347, "lat": 37.9282},
    {"name": "嘉峪关", "aliases": ("嘉峪关", "嘉峪关关城"), "lng": 98.2773, "lat": 39.7865},
    {"name": "祁连山", "aliases": ("祁连山",), "lng": 99.9000, "lat": 38.2000},
    {"name": "河西走廊", "aliases": ("河西走廊", "河西"), "lng": 98.6000, "lat": 39.6000},
    {"name": "大漠", "aliases": ("大漠", "沙漠"), "lng": 95.5000, "lat": 40.2000},
    {"name": "瀚海", "aliases": ("瀚海",), "lng": 94.5000, "lat": 41.3000},
    {"name": "流沙", "aliases": ("流沙",), "lng": 93.4000, "lat": 40.0000},
    {"name": "长安/西安", "aliases": ("长安", "西安"), "lng": 108.9398, "lat": 34.3416},
    {"name": "洛阳", "aliases": ("洛阳",), "lng": 112.4540, "lat": 34.6197},
    {"name": "兰州", "aliases": ("兰州",), "lng": 103.8343, "lat": 36.0611},
]

PLACE_LIKE_KEYWORDS = ("关", "州", "泉", "阳", "安", "山", "河", "城", "路", "道", "塞", "海", "漠", "原", "川", "湖", "江")


def get_hexi_poetry_filters() -> Dict[str, Any]:
    poems = _classified_poems()
    dynasty_counts = Counter(poem["dynasty"] for poem in poems if poem["dynasty"])
    author_counts = Counter(poem["author"] for poem in poems if poem["author"])
    theme_counts = Counter(poem["matched_theme"] for poem in poems if poem["matched_theme"])
    space_counts = Counter(poem["matched_space_type"] for poem in poems if poem["matched_space_type"])
    return {
        "total_poems": len(poems),
        "dynasties": _counter_to_options(dynasty_counts),
        "authors": _counter_to_options(author_counts, limit=30),
        "themes": _all_rule_options(THEME_RULES, theme_counts),
        "space_types": _all_rule_options(SPACE_TYPE_RULES, space_counts),
    }


def list_hexi_poems(
    *,
    dynasty: str = "",
    author: str = "",
    theme: str = "",
    space_type: str = "",
    q: str = "",
    limit: int = 20,
    offset: int = 0,
) -> Dict[str, Any]:
    filtered = _filter_poems(
        dynasty=dynasty,
        author=author,
        theme=theme,
        space_type=space_type,
        q=q,
    )
    start = max(0, int(offset))
    page_size = max(1, min(int(limit), 100))
    end = start + page_size
    page = filtered[start:end]
    return {
        "total": len(filtered),
        "limit": page_size,
        "offset": start,
        "items": [
            {
                "title": poem["title"],
                "author": poem["author"],
                "dynasty": poem["dynasty"],
                "content": poem["content"],
                "matched_theme": poem["matched_theme"],
                "matched_space_type": poem["matched_space_type"],
            }
            for poem in page
        ],
    }


def get_hexi_poetry_place_heatmap(
    *,
    dynasty: str = "",
    author: str = "",
    theme: str = "",
    space_type: str = "",
    q: str = "",
) -> Dict[str, Any]:
    poems = _filter_poems(
        dynasty=dynasty,
        author=author,
        theme=theme,
        space_type=space_type,
        q=q,
    )
    place_entries = _place_entries()
    matched_counts: Dict[str, int] = defaultdict(int)
    matched_poems: Dict[str, set[str]] = defaultdict(set)
    skipped_candidates: set[str] = set()

    for poem in poems:
        searchable_text = "\n".join(
            [
                poem["title"],
                poem["author"],
                poem["dynasty"],
                poem["content"],
                " ".join(poem["keywords"]),
            ]
        )
        poem_key = f"{poem['title']}|{poem['author']}|{poem['dynasty']}"
        for entry in place_entries:
            aliases = entry["aliases"]
            hit_count = sum(searchable_text.count(alias) for alias in aliases if alias)
            if hit_count <= 0:
                continue
            matched_counts[entry["name"]] += hit_count
            matched_poems[entry["name"]].add(poem_key)

        matched_aliases = _matched_place_aliases(searchable_text)
        for token in _extract_place_like_candidates(poem["keywords"], poem["content"], poem["title"]):
            if token not in matched_aliases:
                skipped_candidates.add(token)

    items = []
    for entry in sorted(place_entries, key=lambda row: matched_counts.get(row["name"], 0), reverse=True):
        count = int(matched_counts.get(entry["name"], 0))
        if count <= 0:
            continue
        items.append(
            {
                "name": entry["name"],
                "lng": float(entry["lng"]),
                "lat": float(entry["lat"]),
                "count": count,
                "poems_count": len(matched_poems.get(entry["name"], set())),
            }
        )

    return {
        "total_poems": len(poems),
        "skipped_count": len(skipped_candidates),
        "items": items,
    }


@lru_cache(maxsize=1)
def _classified_poems() -> List[Dict[str, Any]]:
    rows = _load_poems()
    result: List[Dict[str, Any]] = []
    for row in rows:
        title = _clean_text(row.get("title"))
        author = _clean_text(row.get("author"))
        dynasty = _clean_text(row.get("dynasty"))
        lines = _extract_lines(row.get("text"))
        content = "\n".join(lines)
        keywords = _extract_keywords(row.get("keywords"))
        haystack = " ".join([title, author, dynasty, content, " ".join(keywords)])
        matched_theme = _match_rule_label(haystack, THEME_RULES)
        matched_space_type = _match_rule_label(haystack, SPACE_TYPE_RULES)
        result.append(
            {
                "title": title,
                "author": author,
                "dynasty": dynasty,
                "content": content,
                "keywords": keywords,
                "matched_theme": matched_theme,
                "matched_space_type": matched_space_type,
                "search_text": _normalize_text(
                    " ".join([title, author, dynasty, content, " ".join(keywords), matched_theme, matched_space_type])
                ),
            }
        )
    return result


def _filter_poems(
    *,
    dynasty: str = "",
    author: str = "",
    theme: str = "",
    space_type: str = "",
    q: str = "",
) -> List[Dict[str, Any]]:
    poems = _classified_poems()
    normalized_dynasty = _normalize_dynasty(dynasty)
    normalized_author = _normalize_text(author)
    normalized_theme = _clean_text(theme)
    normalized_space_type = _clean_text(space_type)
    normalized_query = _normalize_text(q)

    filtered: List[Dict[str, Any]] = []
    for poem in poems:
        if normalized_dynasty and _normalize_dynasty(poem["dynasty"]) != normalized_dynasty:
            continue
        if normalized_author and _normalize_text(poem["author"]) != normalized_author:
            continue
        if normalized_theme and poem["matched_theme"] != normalized_theme:
            continue
        if normalized_space_type and poem["matched_space_type"] != normalized_space_type:
            continue
        if normalized_query and normalized_query not in poem["search_text"]:
            continue
        filtered.append(poem)
    return filtered


@lru_cache(maxsize=1)
def _place_entries() -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    for item in PLACE_GAZETTEER:
        aliases = tuple(dict.fromkeys(_clean_text(alias) for alias in item.get("aliases", ()) if _clean_text(alias)))
        if not aliases:
            continue
        entries.append(
            {
                "name": _clean_text(item.get("name")),
                "aliases": aliases,
                "lng": float(item["lng"]),
                "lat": float(item["lat"]),
            }
        )
    entries.sort(key=lambda row: max(len(alias) for alias in row["aliases"]), reverse=True)
    return entries


@lru_cache(maxsize=1)
def _matched_place_aliases(text: str) -> set[str]:
    matched = set()
    for entry in _place_entries():
        for alias in entry["aliases"]:
            if alias and alias in text:
                matched.add(alias)
    return matched


def _extract_place_like_candidates(keywords: Iterable[str], content: str, title: str) -> set[str]:
    candidates = set()
    for token in list(keywords) + _split_place_tokens(title):
        text = _clean_text(token)
        if not text or len(text) < 2 or len(text) > 6:
            continue
        if any(alias == text for entry in _place_entries() for alias in entry["aliases"]):
            continue
        if any(marker in text for marker in PLACE_LIKE_KEYWORDS):
            candidates.add(text)
    return candidates


def _split_place_tokens(text: str) -> List[str]:
    return [token for token in re.split(r"[\s，。！？；：、“”‘’（）()《》〈〉【】〔〕—\-]+", _clean_text(text)) if token]


@lru_cache(maxsize=1)
def _load_poems() -> List[Dict[str, Any]]:
    if not POEM_DATA_FILE.exists():
        return []
    payload = json.loads(POEM_DATA_FILE.read_text(encoding="utf-8"))
    return payload if isinstance(payload, list) else []


def _extract_lines(value: Any) -> List[str]:
    if not isinstance(value, list):
        return []
    return [_clean_text(item) for item in value if _clean_text(item)]


def _extract_keywords(value: Any) -> List[str]:
    if not isinstance(value, list):
        return []
    seen = set()
    result: List[str] = []
    for item in value:
        text = _clean_text(item)
        if text and text not in seen:
            seen.add(text)
            result.append(text)
    return result


def _match_rule_label(haystack: str, rules: List[tuple[str, tuple[str, ...]]]) -> str:
    normalized = _normalize_text(haystack)
    for label, keywords in rules:
        for keyword in keywords:
            normalized_keyword = _normalize_text(keyword)
            if normalized_keyword and normalized_keyword in normalized:
                return label
    return ""


def _counter_to_options(counter: Counter[str], *, limit: int | None = None) -> List[Dict[str, Any]]:
    items = counter.most_common(limit)
    return [{"value": value, "count": count} for value, count in items if value]


def _all_rule_options(rules: List[tuple[str, tuple[str, ...]]], counter: Counter[str]) -> List[Dict[str, Any]]:
    return [{"value": label, "count": int(counter.get(label, 0))} for label, _ in rules]


def _normalize_dynasty(value: str) -> str:
    text = _clean_text(value)
    if text.endswith("代"):
        text = text[:-1]
    return text


def _normalize_text(value: Any) -> str:
    text = _clean_text(value)
    return TEXT_NORMALIZE_RE.sub("", text).lower()


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()
