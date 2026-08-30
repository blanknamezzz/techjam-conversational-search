from __future__ import annotations

import re


TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
SPACE_RE = re.compile(r"\s+")

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
    "i", "in", "is", "it", "me", "my", "of", "on", "or", "please", "some",
    "that", "the", "this", "to", "want", "with", "would", "you", "looking",
    "those", "these", "options", "not", "quite", "right", "yet", "ask", "about",
    "one", "specific", "attribute", "what", "matters", "have", "additional",
    "preference", "your", "judgment", "actually", "ignore", "earlier", "need",
}

COLORS = {
    "black", "white", "blue", "red", "pink", "green", "brown", "gray", "grey",
    "purple", "yellow", "orange", "beige", "navy", "gold", "silver", "multicolor",
}
MATERIALS = {
    "cotton", "polyester", "nylon", "leather", "wool", "spandex", "silk", "rayon",
    "fabric", "denim", "linen", "rubber", "suede", "synthetic", "acrylic", "fleece",
}

SYNONYMS = {
    "grey": ("gray",),
    "gray": ("grey",),
    "rainy": ("waterproof", "water", "resistant"),
    "rain": ("waterproof", "water", "resistant"),
    "hiking": ("trekking", "outdoor"),
    "trekking": ("hiking", "outdoor"),
    "sneakers": ("shoes", "trainer"),
    "sneaker": ("shoe", "trainer"),
    "footwear": ("shoe", "shoes"),
    "tee": ("shirt", "tshirt"),
    "wedding": ("formal", "dressy"),
    "gym": ("workout", "fitness", "athletic"),
}


def flatten_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        return " ".join(f"{key} {flatten_text(item)}" for key, item in value.items())
    if isinstance(value, list):
        return " ".join(flatten_text(item) for item in value)
    return str(value)


def normalize_text(text: str) -> str:
    return SPACE_RE.sub(" ", text).strip().casefold()


def terms(text: str, *, expand: bool = False, limit: int = 80) -> list[str]:
    result: list[str] = []
    for token in TOKEN_RE.findall(text.casefold()):
        if len(token) <= 1 or token in STOPWORDS:
            continue
        result.append(token)
        if expand:
            result.extend(SYNONYMS.get(token, ()))
    return list(dict.fromkeys(result))[:limit]


def product_search_text(product: dict) -> str:
    fields = ("title", "categories", "features", "details", "store", "description")
    return normalize_text(" ".join(flatten_text(product.get(field)) for field in fields))


def product_embedding_text(product: dict, limit: int = 2200) -> str:
    parts = [
        f"Title: {flatten_text(product.get('title'))}",
        f"Category: {flatten_text(product.get('categories'))}",
        f"Features: {flatten_text(product.get('features'))}",
        f"Details: {flatten_text(product.get('details'))}",
        f"Description: {flatten_text(product.get('description'))}",
        f"Brand or store: {flatten_text(product.get('store'))}",
    ]
    return SPACE_RE.sub(" ", ". ".join(parts)).strip()[:limit]


def classify_constraint(value: str) -> str:
    lowered = value.casefold()
    tokens = set(TOKEN_RE.findall(lowered))
    if "budget" in lowered or re.search(r"(?:\$|<=|under|below|less than)\s*\d", lowered):
        return "budget"
    if tokens & MATERIALS:
        return "material"
    if tokens & COLORS or "color" in tokens or "colour" in tokens:
        return "color"
    if tokens & {"size", "sizing", "width", "wide", "narrow", "small", "medium", "large"}:
        return "size"
    if tokens & {"department", "style", "fit", "sleeve", "neck", "casual", "formal"}:
        return "style"
    if tokens & {"hiking", "running", "gym", "winter", "outdoor", "work", "wedding"}:
        return "use_case"
    return "feature"
