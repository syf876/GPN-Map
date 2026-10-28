import json
import hashlib
from pathlib import Path
from typing import List, Dict, Optional

# ----------------- 1. 基本配置：使用简体版诗词库 -----------------
# 确保同一目录里已经有 hexi_poems_clean_simplified.json
DATA_FILE = Path(__file__).resolve().parent / "hexi_poems_clean_simplified.json"


# ----------------- 2. 加载 & 加 ID -----------------
def load_poems() -> List[Dict]:
    """从 JSON 文件中加载所有诗词"""
    with DATA_FILE.open("r", encoding="utf-8") as f:
        poems = json.load(f)
    print(f"Loaded {len(poems)} poems from {DATA_FILE}")
    return poems


def make_poem_id(poem: Dict) -> str:
    """根据朝代+作者+标题生成一个稳定的 hash id"""
    base = f"{poem.get('dynasty', '')}-{poem.get('author', '')}-{poem.get('title', '')}"
    h = hashlib.md5(base.encode("utf-8")).hexdigest()[:8]
    return f"poem_{h}"


def attach_ids(poems: List[Dict]) -> List[Dict]:
    """给每首诗添加 id 字段（如果没有的话）"""
    for p in poems:
        if "id" not in p:
            p["id"] = make_poem_id(p)
    return poems


# 模块级别加载一次，后面所有函数都用这个 POEMS
POEMS: List[Dict] = attach_ids(load_poems())


# ----------------- 3. 检索函数 -----------------
def search_by_keyword_in_field(word: str, max_results: int = 10) -> List[Dict]:
    """
    在 keywords 字段里搜索（如：酒泉、甘州、凉州、敦煌、河西等）
    """
    word = word.strip()
    results = []
    for p in POEMS:
        if word in p.get("keywords", []):
            results.append(p)
        if len(results) >= max_results:
            break
    return results


def search_by_text_contains(word: str, max_results: int = 10) -> List[Dict]:
    """
    在标题 + 正文里做简单的包含搜索
    """
    word = word.strip()
    results = []
    for p in POEMS:
        full = "".join(p.get("text", []))
        if word in full or word in p.get("title", ""):
            results.append(p)
        if len(results) >= max_results:
            break
    return results


# ----------------- 4. 格式化 & 构造上下文 -----------------
def format_poem_for_context(poem: Dict) -> str:
    """
    把一首诗格式化成“资料块”，后面会放进 prompt 里给大模型看
    """
    lines = "\n".join(poem.get("text", []))
    keywords = ", ".join(poem.get("keywords", []))
    return (
        f"【编号】{poem['id']}\n"
        f"【标题】{poem['title']}（{poem['dynasty']}·{poem['author']}）\n"
        f"【关键词】{keywords}\n"
        f"【正文】\n{lines}\n"
    )


def build_context_by_place(place: str, limit: int = 5) -> str:
    """
    根据地点/关键词（如“酒泉”“甘州”“凉州”“敦煌”“河西”）
    找出若干首相关诗，拼成一整段“资料”
    """
    # 先在 keywords 里找（更准确）
    candidates = search_by_keyword_in_field(place, max_results=limit)
    # 如果 keywords 里没找到，就退而求其次，在正文里找
    if not candidates:
        candidates = search_by_text_contains(place, max_results=limit)

    blocks = [format_poem_for_context(p) for p in candidates]
    return "\n\n".join(blocks)


# ----------------- 4.1 RAG 问答 prompt -----------------
def build_prompt_with_local_poems(
    user_question: str,
    place: Optional[str] = None,
    limit: int = 5,
) -> str:
    """
    拼出一个完整的 RAG 问答 prompt：
    - 前面是系统说明 + 本地诗词资料
    - 后面是用户问题
    """
    target = place if place is not None else user_question
    context = build_context_by_place(target, limit=limit)

    system_instruction = (
        "你是一个精通河西走廊相关古诗词与历史地理的助手。\n"
        "下面提供的是『本地诗词库』中与主题相关的原文资料，"
        "请回答问题时，优先基于这些诗句进行分析和引用，不要胡编新的诗句。"
    )

    prompt = (
        f"{system_instruction}\n\n"
        f"【本地诗词资料】\n{context}\n\n"
        f"【用户问题】\n{user_question}\n\n"
        "请用通俗的现代汉语回答，可以适当引用上面的诗句，"
        "并解释其中涉及地点/意象的含义和情感。"
    )
    return prompt


# ----------------- 4.2 地理信息抽取 prompt -----------------
def build_geo_extraction_prompt(poem: Dict) -> str:
    """
    针对单首诗构造“古诗词地理信息抽取专家”的提示词，
    使用你给出的完整模板，让模型最终只输出一个合法 JSON。
    """
    # 把正文合并成一行，和你示例里的格式一致
    full_text = " ".join(poem.get("text", []))

    # 这里生成这一段：
    # 【题目】{...} 【作者】{...} 【朝代】{...} 【全文】 {"..."}
    poem_block = (
        f"【题目】{{{poem.get('title', '')}}} "
        f"【作者】{{{poem.get('author', '')}}} "
        f"【朝代】{{{poem.get('dynasty', '')}}} "
        f"【全文】 {{\"{full_text}\"}}\n\n"
    )

    # 规则部分 + “意象→语义→主题/情感→区域→地点”的说明
    rules_block = (
        "请按如下规则抽取结构化信息：\n\n"
        "imagery（意象）：从诗中提取体现地域特征或边塞特征的意象，"
        "例如“大漠、黄沙、孤城、羌笛、关山、瀚海、戍楼、烽火、胡笳、边塞”等。请输出一个列表。\n"
        "place_hist（历史地名）：判断诗句中暗示或直接出现的历史地名，如“凉州、玉门关、阳关、敦煌、甘州、肃州、张掖、武威”等。"
        "可以是 1~3 个，输出列表。\n"
        "place_modern（现代地名）：将 place_hist 映射为现代地名，写成“省/市/区/遗址”的形式，例如：\n"
        "凉州 → 甘肃省武威市\n"
        "玉门关 → 甘肃省敦煌市玉门关遗址\n"
        "输出一个同样长度的列表。\n"
        "reason（依据）：用不超过 60 字说明判断依据。\n"
        "confidence（置信度）：根据判断可靠程度给出 0~1 的数值。请遵循以下规则：\n"
        "如果诗句出现明确古地名（如“玉门关、阳关、凉州”），给 0.8～1.0；\n"
        "如果意象明显指向河西边塞，但无明确地名，给 0.6～0.8；\n"
        "如果定位模糊，只能大致判断是西北地区，给 0.4～0.6；\n"
        "如果几乎没有线索，只是推测，给 0.0～0.4。\n\n"
        "在判断时，请先识别意象→解读语义→推断主题/情感→推断区域→最后确定可能的具体地点。\n\n"
        "⚠️ 最终只能输出一个合法 JSON 对象，不允许输出任何解释说明、标题或文字。\n\n"
        "JSON 输出格式如下（字段名必须一致）：\n\n"
        '{ "imagery": [], "place_hist": [], "place_modern": [], "reason": "", "confidence": 0.0 }'
    )

    # 总提示词 = 身份设定 + “请分析以下诗词” + 具体诗 + 规则
    prompt = (
        "你现在是“古诗词地理信息抽取专家”。请严格按要求分析，并且最后 "
        "只输出一段合法 JSON，不允许有额外解释、没有前后缀、没有自然语言。\n\n"
        "请分析以下诗词：\n\n"
        + poem_block
        + rules_block
    )
    return prompt


# ----------------- 5. 测试入口 main（综合测试） -----------------
def main():
    # 1. 看第一首，确认结构
    print("=== 看第一首诗 ===")
    first = POEMS[0]
    print(first["id"], first["title"], first["author"], first["dynasty"])
    print("".join(first["text"]), "\n")

    # 2. 测试 keyword 搜索
    print("=== keyword 搜索 '酒泉' ===")
    for p in search_by_keyword_in_field("酒泉"):
        print(p["id"], p["title"], p["author"])
    print()

    # 3. 测试 正文搜索
    print("=== 正文搜索 '凉州' ===")
    for p in search_by_text_contains("凉州"):
        print(p["id"], p["title"], p["author"])
    print()

    # 4. 测试：构造一个 RAG 问答 prompt 示例
    print("=== 构造 RAG 问答 prompt 示例（酒泉） ===")
    question = "请结合这些诗词说明“酒泉”在词中的情感意象。"
    prompt_qa = build_prompt_with_local_poems(question, place="酒泉", limit=3)
    print(prompt_qa[:800], "\n...[后面略]...\n")

    # 5. 测试：构造单首诗的地理信息抽取 prompt 示例
    print("=== 构造地理信息抽取 prompt 示例（第一首诗） ===")
    geo_prompt = build_geo_extraction_prompt(first)
    print(geo_prompt[:800], "\n...[后面略]...")


if __name__ == "__main__":
    main()





