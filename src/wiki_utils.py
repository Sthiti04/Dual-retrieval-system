import wikipediaapi
from config import USER_AGENT, MAX_DEPTH

wiki = wikipediaapi.Wikipedia(user_agent=USER_AGENT, language="en")


def collect_titles(cat_names, limit):
    """Breadth-first walk of each category (down to MAX_DEPTH subcategory levels).
    Returns up to `limit` candidate article titles per category, de-duplicated."""
    all_titles = []
    for cat_name in cat_names:
        root = wiki.page(cat_name)
        if not root.exists():
            print(f"   [MISSING] {cat_name}")
            continue
        found, seen, queue = [], set(), [(root, 0)]
        while queue and len(found) < limit:
            cat, depth = queue.pop(0)
            if cat.title in seen:
                continue
            seen.add(cat.title)
            for m in cat.categorymembers.values():
                if m.ns == wikipediaapi.Namespace.MAIN and not m.title.startswith("List of"):
                    found.append(m.title)
                elif m.ns == wikipediaapi.Namespace.CATEGORY and depth < MAX_DEPTH:
                    queue.append((m, depth + 1))
        all_titles += found
    return list(dict.fromkeys(all_titles))