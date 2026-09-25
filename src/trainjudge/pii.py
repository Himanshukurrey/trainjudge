"""Detect sensitive identifiers in training data, across regions.

- Global: payment card numbers (Luhn-checked), IBANs (mod-97-checked),
  international phone numbers, email addresses
- India: Aadhaar (Verhoeff-checked), PAN, UPI IDs, mobile numbers
- US: Social Security numbers, formatted phone numbers
- UK: National Insurance numbers
- Bank account numbers, medical record numbers and dates of birth when
  labelled as such

Detectors use checksums, official format rules or context keywords to keep
false positives low. Findings carry line numbers only; matched values are
never stored or printed.

`mask_text` / `mask_value` replace each detected identifier with a placeholder
such as [EMAIL] or [MRN], keeping labels like "MRN:" readable, so a dataset can
be trained on without the identifiers themselves.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Callable

CARD = "card number"
AADHAAR = "Aadhaar number"
PAN = "PAN"
ACCOUNT = "account number"
UPI_ID = "UPI ID"
SSN = "US SSN"
UK_NINO = "UK National Insurance number"
IBAN = "IBAN"
MRN = "medical record number"
DOB = "date of birth"
PHONE = "phone number"
EMAIL = "email address"
KINDS = (CARD, AADHAAR, PAN, ACCOUNT, UPI_ID, SSN, UK_NINO, IBAN, MRN, DOB, PHONE, EMAIL)

# Where to look for obligations when a kind is found. Pointers, not legal advice.
KIND_REFERENCES = {
    CARD: "PCI DSS",
    AADHAAR: "India's DPDP Act 2023",
    PAN: "India's DPDP Act 2023",
    UPI_ID: "India's DPDP Act 2023",
    SSN: "US state privacy and breach-notification laws",
    UK_NINO: "UK GDPR",
    IBAN: "GDPR",
    MRN: "HIPAA",
}

PLACEHOLDERS = {
    CARD: "[CARD]",
    AADHAAR: "[AADHAAR]",
    PAN: "[PAN]",
    ACCOUNT: "[ACCOUNT]",
    UPI_ID: "[UPI_ID]",
    SSN: "[SSN]",
    UK_NINO: "[NINO]",
    IBAN: "[IBAN]",
    MRN: "[MRN]",
    DOB: "[DOB]",
    PHONE: "[PHONE]",
    EMAIL: "[EMAIL]",
}

_CARD_RE = re.compile(r"(?<![\d-])(?:\d[ -]?){14,18}\d(?![\d-])")
_CARD_LENGTHS = {15, 16, 19}
_CARD_PREFIX_RE = re.compile(r"^(?:[2-6]|8[12])")
_AADHAAR_SPACED_RE = re.compile(r"(?<![\d-])[2-9]\d{3}[ -]\d{4}[ -]\d{4}(?![\d-])")
_AADHAAR_KEYWORD_RE = re.compile(
    r"\b(?:aadhaar|aadhar|uidai|uid)\b\D{0,20}?([2-9]\d{11})(?!\d)", re.IGNORECASE
)
_PAN_RE = re.compile(r"\b[A-Z]{3}[ABCFGHLJPTK][A-Z]\d{4}[A-Z]\b")
_ACCOUNT_RE = re.compile(
    r"\b(?:a/c|acct|account)\s*(?:no\.?|number|num|#)?\s*(?:is\s*)?[:.\-]?\s*(\d{9,18})(?!\d)",
    re.IGNORECASE,
)
_UPI_RE = re.compile(
    r"\b[\w.\-]{3,}@(?:ok(?:axis|sbi|hdfcbank|icici)|ybl|ibl|axl|paytm|upi|apl|waicici|"
    r"icici|sbi|hdfcbank|kotak|barodampay)\b(?!\.)",
    re.IGNORECASE,
)
_PHONE_RE = re.compile(r"(?<![\d+])(?:\+91[\s-]?|0)?[6-9]\d{9}(?!\d)")
# +<country code> then 7-14 more digits, optionally grouped with spaces or dashes.
_INTL_PHONE_RE = re.compile(r"(?<![\w+])\+[1-9]\d{0,2}(?:[\s-]?\(?\d{1,4}\)?){2,5}(?![\d-])")
# US numbers only count when formatted, e.g. (415) 555-0123 or 415-555-0123.
_US_PHONE_RE = re.compile(r"(?<![\d-])(?:\(\d{3}\)\s?|\d{3}[-.])\d{3}[-.]\d{4}(?![\d-])")
# Area 000, 666 and 900-999, group 00 and serial 0000 are never issued.
_SSN_RE = re.compile(r"(?<![\d-])(?!000|666|9\d\d)\d{3}-(?!00)\d{2}-(?!0000)\d{4}(?![\d-])")
_SSN_KEYWORD_RE = re.compile(
    r"\b(?:ssn|social security(?: number| no\.?)?)\b\D{0,15}?(\d{9})(?!\d)", re.IGNORECASE
)
# First letter not D/F/I/Q/U/V, second not D/F/I/O/Q/U/V; some prefixes are never issued.
_NINO_RE = re.compile(
    r"\b(?!BG|GB|NK|KN|TN|NT|ZZ)[A-CEGHJ-PR-TW-Z][A-CEGHJ-NPR-TW-Z]\s?\d{2}\s?\d{2}\s?\d{2}\s?[A-D]\b"
)
# Medical record numbers and dates of birth only count when labelled as such.
_MRN_RE = re.compile(
    r"\b(?:mrn|medical record(?: number| no\.?| #)?)\s*[:#-]?\s*([A-Z]{0,3}\d[\d-]{4,14})\b",
    re.IGNORECASE,
)
_DOB_RE = re.compile(
    r"\b(?:dob|d\.o\.b\.?|date of birth|born on)\s*[:-]?\s*"
    r"(\d{1,4}[/.-]\d{1,2}[/.-]\d{1,4}"  # 03/14/1985, 1985-03-14
    r"|\d{1,2}\s+[a-z]{3,9}\s+\d{4}"  # 14 March 1985
    r"|[a-z]{3,9}\s+\d{1,2},?\s+\d{4})",  # March 14, 1985
    re.IGNORECASE,
)
_IBAN_RE = re.compile(r"\b[A-Z]{2}\d{2}(?:\s?[A-Z0-9]){11,30}\b")
_EMAIL_RE = re.compile(r"\b[\w.+-]+@((?:[\w-]+\.)+[a-z]{2,})\b", re.IGNORECASE)
_RESERVED_EMAIL_DOMAINS = re.compile(r"(?:^|\.)(?:example\.(?:com|org|net)|test|invalid|example)$")

# Verhoeff tables (dihedral group D5), used by UIDAI for the Aadhaar check digit.
_D = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    [1, 2, 3, 4, 0, 6, 7, 8, 9, 5],
    [2, 3, 4, 0, 1, 7, 8, 9, 5, 6],
    [3, 4, 0, 1, 2, 8, 9, 5, 6, 7],
    [4, 0, 1, 2, 3, 9, 5, 6, 7, 8],
    [5, 9, 8, 7, 6, 0, 4, 3, 2, 1],
    [6, 5, 9, 8, 7, 1, 0, 4, 3, 2],
    [7, 6, 5, 9, 8, 2, 1, 0, 4, 3],
    [8, 7, 6, 5, 9, 3, 2, 1, 0, 4],
    [9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
]
_P = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    [1, 5, 7, 6, 2, 8, 3, 0, 9, 4],
    [5, 8, 0, 3, 7, 9, 6, 1, 4, 2],
    [8, 9, 1, 6, 0, 4, 3, 5, 2, 7],
    [9, 4, 5, 3, 1, 2, 6, 8, 7, 0],
    [4, 2, 8, 6, 5, 7, 3, 9, 0, 1],
    [2, 7, 9, 3, 8, 0, 6, 4, 1, 5],
    [7, 0, 4, 6, 9, 1, 3, 2, 5, 8],
]
_INV = [0, 4, 3, 2, 1, 5, 6, 7, 8, 9]


def luhn_valid(digits: str) -> bool:
    return _luhn_sum(digits) % 10 == 0


def luhn_check_digit(digits: str) -> str:
    return str((10 - _luhn_sum(digits + "0") % 10) % 10)


def _luhn_sum(digits: str) -> int:
    total = 0
    for i, ch in enumerate(reversed(digits)):
        n = int(ch)
        if i % 2:
            n = n * 2 - 9 if n > 4 else n * 2
        total += n
    return total


def verhoeff_valid(digits: str) -> bool:
    c = 0
    for i, ch in enumerate(reversed(digits)):
        c = _D[c][_P[i % 8][int(ch)]]
    return c == 0


def verhoeff_check_digit(digits: str) -> str:
    c = 0
    for i, ch in enumerate(reversed(digits)):
        c = _D[c][_P[(i + 1) % 8][int(ch)]]
    return str(_INV[c])


def iban_valid(iban: str) -> bool:
    """ISO 13616 mod-97 check."""
    compact = re.sub(r"\s", "", iban).upper()
    if not 15 <= len(compact) <= 34:
        return False
    rearranged = compact[4:] + compact[:4]
    return int("".join(str(int(ch, 36)) for ch in rearranged)) % 97 == 1


def references(kinds) -> list[str]:
    """Distinct compliance pointers for the kinds found, in a stable order."""
    seen: list[str] = []
    for kind in KINDS:
        ref = KIND_REFERENCES.get(kind)
        if kind in kinds and ref and ref not in seen:
            seen.append(ref)
    return seen


def scan_value(value: object) -> set[str]:
    """Scan every string inside a parsed JSON value."""
    return scan_text("\n".join(_strings(value)))


def _strings(value: object):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for v in value.values():
            yield from _strings(v)
    elif isinstance(value, list):
        for v in value:
            yield from _strings(v)


def _digits(text: str) -> str:
    return re.sub(r"\D", "", text)


def _valid_card(m: re.Match) -> bool:
    digits = _digits(m.group())
    return len(digits) in _CARD_LENGTHS and bool(_CARD_PREFIX_RE.match(digits)) and luhn_valid(digits)


def _valid_email(m: re.Match) -> bool:
    return not _RESERVED_EMAIL_DOMAINS.search(m.group(1).lower())


def _always(m: re.Match) -> bool:
    return True


# (kind, pattern, capture group holding the identifier, validator)
_DETECTORS: list[tuple[str, re.Pattern, int, Callable[[re.Match], bool]]] = [
    (CARD, _CARD_RE, 0, _valid_card),
    (AADHAAR, _AADHAAR_SPACED_RE, 0, lambda m: verhoeff_valid(_digits(m.group()))),
    (AADHAAR, _AADHAAR_KEYWORD_RE, 1, lambda m: verhoeff_valid(m.group(1))),
    (PAN, _PAN_RE, 0, _always),
    (ACCOUNT, _ACCOUNT_RE, 1, _always),
    (UPI_ID, _UPI_RE, 0, _always),
    (SSN, _SSN_RE, 0, _always),
    (SSN, _SSN_KEYWORD_RE, 1, _always),
    (UK_NINO, _NINO_RE, 0, _always),
    (IBAN, _IBAN_RE, 0, lambda m: iban_valid(m.group())),
    (MRN, _MRN_RE, 1, _always),
    (DOB, _DOB_RE, 1, _always),
    (PHONE, _PHONE_RE, 0, _always),
    (PHONE, _US_PHONE_RE, 0, _always),
    (PHONE, _INTL_PHONE_RE, 0, lambda m: 8 <= len(_digits(m.group())) <= 15),
    (EMAIL, _EMAIL_RE, 0, _valid_email),
]


def find_spans(text: str) -> list[tuple[int, int, str]]:
    """Every detected identifier as (start, end, kind); spans may overlap."""
    spans = []
    for kind, pattern, group, valid in _DETECTORS:
        for m in pattern.finditer(text):
            if valid(m):
                spans.append((m.start(group), m.end(group), kind))
    return spans


def scan_text(text: str) -> set[str]:
    """Return the kinds of sensitive identifier found in text."""
    return {kind for _, _, kind in find_spans(text)}


def mask_text(text: str) -> tuple[str, Counter]:
    """Replace each identifier with its placeholder; return the text and counts by kind.

    Where matches overlap (a phone-number UPI ID, say), the longest one wins.
    """
    chosen: list[tuple[int, int, str]] = []
    for start, end, kind in sorted(find_spans(text), key=lambda s: (s[0], s[0] - s[1])):
        if not chosen or start >= chosen[-1][1]:
            chosen.append((start, end, kind))
    counts: Counter = Counter()
    for start, end, kind in reversed(chosen):
        text = text[:start] + PLACEHOLDERS[kind] + text[end:]
        counts[kind] += 1
    return text, counts


def mask_value(value: object) -> tuple[object, Counter]:
    """Mask every string inside a parsed JSON value; return it and counts by kind."""
    counts: Counter = Counter()
    if isinstance(value, str):
        return mask_text(value)
    if isinstance(value, dict):
        out = {}
        for k, v in value.items():
            out[k], c = mask_value(v)
            counts += c
        return out, counts
    if isinstance(value, list):
        items = []
        for v in value:
            masked, c = mask_value(v)
            items.append(masked)
            counts += c
        return items, counts
    return value, counts
