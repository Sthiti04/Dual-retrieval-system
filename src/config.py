from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"

USER_AGENT = "IRTermProject/1.0 (h20250055@pilani.bits-pilani.ac.in)"

# label -> ([Wikipedia categories], max articles to keep for this label)
CATEGORIES = {
    "space_missions":        (["Category:Space probes", "Category:Apollo program",
                               "Category:Project Mercury"], 150),
    "solar_system":          (["Category:Moons of Saturn",
                               "Category:Planets of the Solar System"], 100),
    "greek_mythology":       (["Category:Titans (mythology)",
                               "Category:Olympian deities"], 120),
    "chemical_elements":     (["Category:Chemical elements"], 100),
    "programming_languages": (["Category:Programming languages"], 150),
    "malware_security":      (["Category:Malware", "Category:Computer security"], 100),
    "infectious_diseases":   (["Category:Infectious diseases",
                               "Category:Epidemics"], 150),
    "industrial_history":    (["Category:Industrial Revolution",
                               "Category:Cold War"], 150),
    "noise_instruments":     (["Category:Musical instruments"], 80),
}

MIN_WORDS = 80    # skip stubs
MAX_WORDS = 300   # truncate so BM25 and the embedding model see the same amount of text
MAX_DEPTH = 1     # how many subcategory levels to descend
SEED = 42         # makes the random sample reproducible
INDEX_WORDS = 250   # words per document seen by both engines (chosen via token_check.py)