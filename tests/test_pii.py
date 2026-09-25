import pytest

from trainjudge import pii
from trainjudge.pii import (
    luhn_check_digit,
    luhn_valid,
    scan_text,
    scan_value,
    verhoeff_check_digit,
    verhoeff_valid,
)


def test_luhn():
    assert luhn_valid("4111111111111111")
    assert not luhn_valid("4111111111111112")
    assert luhn_check_digit("411111111111111") == "1"


def test_verhoeff():
    # Reference example: 236 -> check digit 3.
    assert verhoeff_check_digit("236") == "3"
    assert verhoeff_valid("2363")
    assert not verhoeff_valid("2364")


def _aadhaar(body="23456789012"):
    return body + verhoeff_check_digit(body)


@pytest.mark.parametrize(
    "text, kind",
    [
        ("POS 4111111111111111 FRESHKART", pii.CARD),
        ("card 4111 1111 1111 1111 used", pii.CARD),
        ("card 4111-1111-1111-1111 used", pii.CARD),
        ("AEPS/CW/{a4} {a8} {a12}/NSTB", pii.AADHAAR),
        ("Aadhaar: {a}", pii.AADHAAR),
        ("NEFT CR BONUS PAN ABCPK1234L", pii.PAN),
        ("please credit account no. 123456789012", pii.ACCOUNT),
        ("A/C 00123456789", pii.ACCOUNT),
        ("UPI/CR/412345678901/RAVI/ravi.m@okaxis", pii.UPI_ID),
        ("call me on +91 9876543210", pii.PHONE),
        ("call me on 9876543210", pii.PHONE),
        ("mail anita.rao@gmail.com", pii.EMAIL),
    ],
)
def test_detects(text, kind):
    a = _aadhaar()
    text = text.format(a=a, a4=a[:4], a8=a[4:8], a12=a[8:])
    assert kind in scan_text(text)


@pytest.mark.parametrize(
    "text",
    [
        "POS XXXXXXXXXXXX1234 FRESHKART*BLR",  # masked card
        "POS 4111111111111112 FRESHKART",  # fails Luhn
        "UPI/DR/412345678901/FRESHKART/NSTB/Payment",  # 12-digit RRN
        "NACH/DR/SUNRISE FINANCE/LN12345678901",
        "NEFT CR-NSTBN51234567890123-ORBIT TECHNOLOGIES-SALARY SEP",
        "SELECT email FROM customers WHERE signup_date > '2025-01-01';",
        "support@example.com",  # reserved test domain
        "ABCDE1234F",  # 4th char E isn't a valid PAN holder type
        "reference {bad_aadhaar}",  # not Verhoeff-valid
    ],
)
def test_ignores(text):
    a = _aadhaar()
    bad = a[:-1] + str((int(a[-1]) + 1) % 10)
    text = text.replace("{bad_aadhaar}", f"{bad[:4]} {bad[4:8]} {bad[8:]}")
    assert scan_text(text) == set()


def test_scan_value_walks_nested_json():
    row = {"prompt": "x", "meta": {"notes": ["call 9876543210"]}}
    assert scan_value(row) == {pii.PHONE}


@pytest.mark.parametrize(
    "text, kind",
    [
        ("SSN 123-45-6789 on file", pii.SSN),
        ("ssn: 123456789", pii.SSN),
        ("Social Security Number 123456789", pii.SSN),
        ("NI number AB 12 34 56 C", pii.UK_NINO),
        ("nino AB123456C", pii.UK_NINO),
        ("IBAN GB82 WEST 1234 5698 7654 32", pii.IBAN),
        ("pay DE89370400440532013000 now", pii.IBAN),
        ("call +44 20 7946 0958", pii.PHONE),
        ("call (415) 555-0123", pii.PHONE),
        ("call 415-555-0123", pii.PHONE),
    ],
)
def test_detects_other_regions(text, kind):
    assert kind in scan_text(text)


@pytest.mark.parametrize(
    "text",
    [
        "SSN 000-12-3456",  # area 000 is never issued
        "id 666-12-3456",  # area 666 is never issued
        "ref 123-00-4567",  # group 00 is never issued
        "date 2025-01-01",
        "order 123-45-678",
        "DE89370400440532013001",  # fails mod-97
        "BG123456A",  # BG prefix is never issued
        "v1.2.3-4",
        "+1 2",
    ],
)
def test_ignores_other_region_lookalikes(text):
    assert scan_text(text) == set()


def test_iban_checksum():
    assert pii.iban_valid("GB82 WEST 1234 5698 7654 32")
    assert not pii.iban_valid("GB82 WEST 1234 5698 7654 33")
    assert not pii.iban_valid("GB82")


def test_references_follow_what_was_found():
    assert pii.references({pii.CARD, pii.AADHAAR, pii.PAN}) == ["PCI DSS", "India's DPDP Act 2023"]
    assert pii.references({pii.IBAN, pii.UK_NINO}) == ["UK GDPR", "GDPR"]
    assert pii.references({pii.EMAIL}) == []
