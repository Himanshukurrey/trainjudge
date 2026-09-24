"""Generate the BFSI transaction-categorization demo dataset (deterministic).

Bank statement narrations (UPI, NEFT, IMPS, NACH, card, ATM/AePS, BBPS) paired
with a JSON label: category, merchant, channel, direction. Mapping cryptic
narrations to a bank's own category taxonomy is a format/behavior gap, and
field-level exact match measures it objectively.

All banks, merchants, people and identifiers are fictional. A few rows carry
unmasked card numbers, Aadhaar numbers, PANs or personal UPI IDs on purpose,
so the sensitive-data scan has something to find. The identifiers are
checksum-valid test values, not real ones. Duplicate, low-quality and malformed
rows are mixed in as well.

    python demo/bfsi_transactions/generate.py
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from trainjudge.pii import luhn_check_digit, verhoeff_check_digit

HERE = Path(__file__).parent
SEED = 20260924

N_CLEAN = 600
N_PII = 14  # counted within N_CLEAN
N_DUPLICATE = 36
N_LOW_QUALITY = 12
N_MALFORMED = 8

INSTRUCTION = (
    "Categorize this bank transaction. Reply with JSON: category, merchant, channel, direction."
)

MERCHANTS = {
    "groceries": ["FreshKart", "DailyBasket", "GreenLeaf Mart"],
    "food_delivery": ["QuickBite", "TiffinBox"],
    "fuel": ["IndoFuel Station", "HighwayPetro"],
    "shopping": ["StyleLoop", "GadgetHub", "HomeNest"],
    "travel": ["RailGo", "SkyHop Airlines", "CabNow"],
}
BILLERS = ["Metro Power Distribution", "CityGas", "AirNet Broadband", "Jal Nigam Water"]
LENDERS = ["Sunrise Finance", "Pinnacle Home Finance", "Crescent Auto Loans"]
AMCS = ["Horizon MF", "Evergreen MF"]
INSURERS = ["Shield Life Insurance", "SecureHealth Insurance"]
EMPLOYERS = ["Orbit Technologies", "Lotus Pharma", "Delta Logistics"]
PEOPLE = ["Ravi Menon", "Anita Rao", "Farhan Ali", "Meera Iyer", "Suresh Pillai", "Kavya Nair"]
BANK_CODES = ["NSTB", "KSVB", "ORCB", "SRBK"]
CITIES = ["BLR", "MUM", "DEL", "HYD", "CHN", "PUN"]
MONTHS = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]


def _digits(rng: random.Random, n: int, first: str = "123456789") -> str:
    return rng.choice(first) + "".join(rng.choice("0123456789") for _ in range(n - 1))


def _code(name: str) -> str:
    return "".join(ch for ch in name.upper() if ch.isalnum())[:12]


def _label(category, merchant, channel, direction) -> str:
    return json.dumps(
        {"category": category, "merchant": merchant, "channel": channel, "direction": direction}
    )


def transaction(rng: random.Random) -> tuple[str, str]:
    kind = rng.choices(
        ["upi", "pos", "ecom", "salary", "emi", "sip", "insurance", "atm", "imps", "rent",
         "bill", "charges", "interest"],
        weights=[14, 8, 6, 4, 4, 3, 2, 4, 4, 3, 4, 2, 2],
    )[0]
    rrn = _digits(rng, 12, "3456")
    bank = rng.choice(BANK_CODES)
    month = rng.choice(MONTHS)
    if kind in ("upi", "pos", "ecom"):
        category = rng.choice(list(MERCHANTS))
        merchant = rng.choice(MERCHANTS[category])
        if kind == "upi":
            note = rng.choice(["Payment", "UPI", "Order", "Pay to merchant"])
            return (f"UPI/DR/{rrn}/{merchant.upper()}/{bank}/{note}",
                    _label(category, merchant, "UPI", "debit"))
        if kind == "pos":
            return ((f"POS XXXXXXXXXXXX{_digits(rng, 4, '0123456789')} "
                    f"{_code(merchant)}*{rng.choice(CITIES)} {rng.randint(1, 28):02d}{month}"),
                    _label(category, merchant, "CARD_POS", "debit"))
        return (f"ECOM PUR/{_code(merchant)}.IN/{_digits(rng, 10, '12345')}",
                _label(category, merchant, "CARD_ECOM", "debit"))
    if kind == "salary":
        employer = rng.choice(EMPLOYERS)
        return (f"NEFT CR-{bank}N5{_digits(rng, 13, '12345')}-{employer.upper()}-SALARY {month}",
                _label("salary", employer, "NEFT", "credit"))
    if kind == "emi":
        lender = rng.choice(LENDERS)
        return (f"NACH/DR/{lender.upper()}/LN{_digits(rng, 11, '12345')}",
                _label("emi", lender, "NACH", "debit"))
    if kind == "sip":
        amc = rng.choice(AMCS)
        return (f"ACH D- {_code(amc)}-SIP-{_digits(rng, 8, '12345')}",
                _label("mutual_fund_sip", amc, "NACH", "debit"))
    if kind == "insurance":
        insurer = rng.choice(INSURERS)
        return (f"NACH/DR/{insurer.upper()}/PREMIUM/{_digits(rng, 8, '12345')}",
                _label("insurance_premium", insurer, "NACH", "debit"))
    if kind == "atm":
        return (f"ATM WDL/{bank}{_digits(rng, 5, '12345')}/{rng.choice(CITIES)}",
                _label("cash_withdrawal", None, "ATM", "debit"))
    if kind == "imps":
        person = rng.choice(PEOPLE)
        direction = rng.choice(["debit", "credit"])
        tag = "P2A" if direction == "debit" else "CR"
        return (f"IMPS/{tag}/{rrn}/{person.upper()}/{bank}",
                _label("transfer", None, "IMPS", direction))
    if kind == "rent":
        person = rng.choice(PEOPLE)
        return (f"UPI/DR/{rrn}/{person.upper()}/{bank}/RENT {month}",
                _label("rent", None, "UPI", "debit"))
    if kind == "bill":
        biller = rng.choice(BILLERS)
        return (f"BBPS/{_code(biller)}/{_digits(rng, 11, '12345')}",
                _label("utilities", biller, "BBPS", "debit"))
    if kind == "charges":
        narration = rng.choice([f"SMS ALERT CHGS QTR {month}", f"DEBIT CARD AMC {month}",
                                "CHQ BOOK ISSUE CHGS", f"MIN BAL CHGS {month}"])
        return narration, _label("bank_charges", None, "BANK", "debit")
    return (f"INT.PD:{_digits(rng, 4, '0123456789')}:01{month}-30{month}",
            _label("interest", None, "BANK", "credit"))


def pii_transaction(rng: random.Random, i: int) -> tuple[str, str]:
    """A narration that leaks an unmasked identifier (test values, checksum-valid)."""
    kind = i % 4
    if kind == 0:
        body = "4" + _digits(rng, 14, "0123456789")
        card = body + luhn_check_digit(body)
        category = rng.choice(list(MERCHANTS))
        merchant = rng.choice(MERCHANTS[category])
        return (f"POS {card} {_code(merchant)}*{rng.choice(CITIES)}",
                _label(category, merchant, "CARD_POS", "debit"))
    if kind == 1:
        body = _digits(rng, 11, "23456789")
        aadhaar = body + verhoeff_check_digit(body)
        spaced = f"{aadhaar[:4]} {aadhaar[4:8]} {aadhaar[8:]}"
        return (f"AEPS/CW/{spaced}/{rng.choice(BANK_CODES)}",
                _label("cash_withdrawal", None, "AEPS", "debit"))
    if kind == 2:
        pan = "".join(rng.choice("ABCDEFGHJKLMNPRSTUVWXYZ") for _ in range(3)) + "P"
        pan += rng.choice("ABCDEFGHJKLMNPRSTUVWXYZ") + _digits(rng, 4, "123456789") + "K"
        employer = rng.choice(EMPLOYERS)
        return ((f"NEFT CR-{rng.choice(BANK_CODES)}N5{_digits(rng, 13, '12345')}-"
                f"{employer.upper()}-BONUS PAN {pan}"),
                _label("salary", employer, "NEFT", "credit"))
    vpa = _digits(rng, 10, "6789") + "@ybl"
    return (f"UPI/CR/{_digits(rng, 12, '3456')}/{rng.choice(PEOPLE).upper()}/{vpa}",
            _label("transfer", None, "UPI", "credit"))


def _prompt(narration: str) -> str:
    return f"{INSTRUCTION}\nNarration: {narration}"


def main() -> None:
    rng = random.Random(SEED)
    clean, seen = [], set()
    for i in range(N_PII):
        narration, label = pii_transaction(rng, i)
        seen.add(narration)
        clean.append({"prompt": _prompt(narration), "completion": label})
    while len(clean) < N_CLEAN:
        narration, label = transaction(rng)
        if narration not in seen:
            seen.add(narration)
            clean.append({"prompt": _prompt(narration), "completion": label})

    lines = [json.dumps(r) for r in clean]
    for i in range(N_DUPLICATE):
        row = dict(rng.choice(clean[N_PII:]))
        if i % 2:
            row["prompt"] = row["prompt"].replace("Narration: ", "Narration:  ") + " "
        lines.append(json.dumps(row))
    for i in range(N_LOW_QUALITY):
        narration, _ = transaction(rng)
        completion = ["N/A", "TODO", "I'm sorry, I can't categorize this transaction."][i % 3]
        lines.append(json.dumps({"prompt": _prompt(narration), "completion": completion}))
    for i in range(N_MALFORMED):
        narration, label = transaction(rng)
        lines.append([
            json.dumps({"prompt": _prompt(narration)}),
            json.dumps({"prompt": _prompt(narration), "completion": ""}),
            json.dumps({"prompt": _prompt(narration), "completion": json.loads(label)}),
            json.dumps({"prompt": _prompt(narration), "completion": label})[:60],
        ][i % 4])
    rng.shuffle(lines)

    out = HERE / "data.jsonl"
    out.write_text("\n".join(lines) + "\n")
    print(f"wrote {len(lines)} rows to {out} ({N_CLEAN} clean incl. {N_PII} with PII, "
          f"{N_DUPLICATE} duplicate, {N_LOW_QUALITY} low-quality, {N_MALFORMED} malformed)")


if __name__ == "__main__":
    main()
