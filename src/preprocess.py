import re
import unicodedata
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# Legal suffix normalization
# These are STRIPPED from business names so "Acme Corp" and "Acme Corporation"
# both become "Acme" and match on TF-IDF cosine similarity.
# Open-set: covers US, India, France without hardcoding country logic.
# ─────────────────────────────────────────────────────────────────────────────
LEGAL_PATTERNS = [
    r"\b(corporation|incorporated|corp|inc|limited|ltd|co|company|llc|llp|plc)\b",
    r"\b(private limited|pvt ltd|pvt|p ltd)\b",
    r"\b(sarl|sas|sasu|sa|eurl|snc|gie|sci|ei|scop)\b",
]
LEGAL_REGEX = re.compile("|".join(LEGAL_PATTERNS), flags=re.IGNORECASE)

# ─────────────────────────────────────────────────────────────────────────────
# Street abbreviation expansion (addresses only)
# NOTE: "st" deliberately EXCLUDED — fires on "east", "west", "first" etc.
# Single-pass compiled regex is ~14x faster than individual re.sub calls.
# ─────────────────────────────────────────────────────────────────────────────
STREET_ABBREVIATIONS = {
    r"\brd\b": "road",
    r"\bave\b|\bav\b": "avenue",
    r"\bdr\b": "drive",
    r"\bln\b": "lane",
    r"\bblvd\b|\bbvd\b": "boulevard",
    r"\bhwy\b": "highway",
    r"\bpkwy\b": "parkway",
    r"\bsq\b": "square",
    r"\bft\b": "fort",
    r"\bmt\b": "mount",
    # France
    r"\bbd\b": "boulevard",
    r"\ball\b|\ballee\b": "allee",
    r"\bimp\b|\bimpasse\b": "impasse",
    r"\bche\b|\bchemin\b": "chemin",
}

_STREET_KEYS = list(STREET_ABBREVIATIONS.keys())
_STREET_REPLACEMENTS = list(STREET_ABBREVIATIONS.values())

# Build one single compiled regex with named groups for each pattern
_STREET_PATTERN = re.compile(
    "|".join(f"(?P<pat{i}>{p})" for i, p in enumerate(_STREET_KEYS)),
    flags=re.IGNORECASE,
)


def _expand_street(match: re.Match) -> str:
    """Callback for single-pass street abbreviation expansion."""
    for i in range(len(_STREET_KEYS)):
        if match.group(f"pat{i}") is not None:
            return _STREET_REPLACEMENTS[i]
    return match.group(0)


def strip_accents(text: str) -> str:
    """Strips accents from French/European characters via NFKD decomposition.
    'Société' → 'Societe', 'café' → 'cafe', 'Île' → 'Ile'
    """
    if not isinstance(text, str):
        return ""
    normalized = unicodedata.normalize("NFKD", text)
    return "".join(c for c in normalized if not unicodedata.combining(c))


def clean_string(text: str, is_address: bool = False) -> str:
    """Normalizes a raw business name or address string.
    1. Strip accents (handles French)
    2. Lowercase
    3. & → and
    4a. (name) Strip legal suffixes
    4b. (address) Expand street abbreviations
    5. Remove non-alphanumeric characters
    6. Collapse whitespace
    """
    if not isinstance(text, str):
        return ""

    text = strip_accents(text).lower()
    text = text.replace("&", " and ")

    if is_address:
        text = _STREET_PATTERN.sub(_expand_street, text)
    else:
        text = LEGAL_REGEX.sub(" ", text)

    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_address_digits(text: str) -> set:
    """Extracts all standalone numbers from an address string.
    Used for numeric Jaccard overlap feature (door numbers, PINs, postcodes).
    Returns a set so intersection/union is O(1).
    """
    if not isinstance(text, str):
        return set()
    return set(re.findall(r"\b\d+\b", text))


def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Applies full preprocessing pipeline to a raw source DataFrame.
    Adds columns: clean_name, clean_address, addr_numbers, composite_text.
    Shows tqdm progress bars for each column (visible on terminal).
    """
    df = df.copy()
    df["business_name"] = df["business_name"].fillna("").astype(str)
    df["business_address"] = df["business_address"].fillna("").astype(str)
    # CONSTRAINT: country is open-set — never hardcode. Store as-is uppercased string.
    df["country"] = df["country"].fillna("UNKNOWN").astype(str).str.strip().str.upper()

    from tqdm import tqdm
    tqdm.pandas(desc="  Cleaning strings")

    df["clean_name"] = df["business_name"].progress_apply(
        lambda x: clean_string(x, is_address=False)
    )
    df["clean_address"] = df["business_address"].progress_apply(
        lambda x: clean_string(x, is_address=True)
    )
    df["addr_numbers"] = df["business_address"].progress_apply(extract_address_digits)

    # Composite text used as TF-IDF input for blocking
    df["composite_text"] = df["clean_name"] + " " + df["clean_address"]
    return df
