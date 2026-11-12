import json, re
from config import DATA, INDEX_WORDS

TYPED = r"(created|developed|designed|released|first appeared).{0,40}197[0-9]"
CHECKS = {
    "q09": (r"dynamic(ally)?[- ]typ", None),
    "q10": (r"(no|not|without|lacks?|neither).{0,40}garbage|manual memory management", None),
    "q11": (r"computer virus|computer worm|antivirus|anti-virus", None),
    "q12": (r"mosquito", None),
    "q13": (r"Apollo.{0,60}(god|son of Zeus|Leto|Delphi|Hyacinth|Daphne|lyre|oracle)|(god|Leto|Delphi|oracle).{0,60}Apollo", None),
    "q14": (r"Apollo.{0,80}(experiment|instrument|deployed)|ALSEP", None),
    "q15": (r"\btrojan\b.{0,40}(horse|malware|downloader|botnet)|trojan horse \(computing\)", ["malware_security"]),
    "q16": (TYPED, ["programming_languages"]),
    "q17": (TYPED, ["computing_systems", "history_of_computing"]),
    "q18": (r"(discovered|isolated|first identified).{0,40}18[0-9][0-9]", ["chemical_elements", "element_groups"]),
    "q19": (r"Project Mercury|Mercury program|Mercury-Redstone|Mercury-Atlas|Little Joe", ["space_missions"]),
    "q20": (r"(no|not) (an? )?(effective |licensed )?vaccine", None),
}

docs = [json.loads(l) for l in (DATA / "corpus.jsonl").open(encoding="utf-8")]
for qid, (pat, cats) in CHECKS.items():
    rx = re.compile(pat, re.I)
    hits = [d for d in docs
            if (not cats or d["category"] in cats)
            and rx.search(" ".join(d["text"].split()[:INDEX_WORDS]))]
    print(f"\n{qid}: {len(hits)} regex matches")
    print("   " + " ; ".join(d["title"][:30] for d in hits[:25]))