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

# added new categories to expand the corpus 
CATEGORIES.update({
    "bacterial_viral_diseases": (["Category:Bacterial diseases", "Category:Viral diseases"], 200),
    "epidemic_history":         (["Category:19th-century epidemics", "Category:20th-century     epidemics", "Category:Pandemics", "Category:Cholera outbreaks"], 100),
    "astronauts":               (["Category:NASA astronauts", "Category:Soviet cosmonauts"], 150),
    "astronomy_bodies":         (["Category:Asteroids", "Category:Comets", "Category:Stars"], 150),
    "metals_corrosion":         (["Category:Corrosion", "Category:Metallurgy"], 120),
    "element_groups":           (["Category:Nonmetals", "Category:Noble gases",
                                  "Category:Metalloids", "Category:Halogens",
                                  "Category:Alkali metals"], 120),
    "computing_systems":        (["Category:Operating systems", "Category:Computer networking"], 150),
    "history_of_computing":     (["Category:History of computing"], 100),
    "roman_mythology":          (["Category:Roman gods"], 80),
    "trojan_war":               (["Category:Trojan War"], 80),
    "maps_cartography":         (["Category:Atlases", "Category:Cartography"], 100),
    "gemstones_crystals":       (["Category:Gemstones", "Category:Crystals"], 100),
    "indonesia_geography":      (["Category:Islands of Indonesia", "Category:Coffee"], 80),
    "banking_finance":          (["Category:Banking"], 100),
    "river_landforms":          (["Category:Fluvial landforms"], 100),
    "noise_geography":          (["Category:Mountains"], 100),
    "noise_cuisine":            (["Category:Cuisine"], 100),
})