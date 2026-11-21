"""
Safe Copy-Paste Parser for Facebook Group Rental Posts (0% Account Ban Risk).

Instead of scraping closed Facebook Groups, hosts paste a raw listing message here.
The parser extracts bedroom count, neighborhood, monthly asking rent (MXN), and
whether CFE electricity is billed separately from English or Spanish listing text.
"""
from __future__ import annotations

import re
from typing import Dict

# Known neighborhoods across Playa del Carmen, Tulum, and CDMX benchmarks.
_KNOWN_NEIGHBORHOODS = (
    "gonzalo guerrero",
    "centro",
    "la veleta",
    "aldea zama",
    "roma norte",
    "condesa",
)

# Common English/Spanish phrases indicating the tenant pays CFE electricity on top of rent.
_TENANT_PAYS_CFE_KEYWORDS = (
    "+ cfe",
    "plus cfe",
    "mas cfe",
    "más cfe",
    "plus electric",
    "no incluye luz",
)

# Matches e.g. "2BR", "1 bed", "2 recámaras", "1 habitacion".
_BEDROOM_RE = re.compile(r"(\d)\s*(?:br|bed|rec[aá]mara|habitaci[oó]n)")

# Matches 5-digit Mexican monthly rents such as "$24,500 MXN/mes", "25.000 pesos", or "26000".
_MONTHLY_PRICE_RE = re.compile(r"\$?\s*(\d{2}[,\.]?\d{3})\s*(?:mxn|pesos|/mes|monthly)?")


def parse_facebook_rental_post(raw_text: str) -> Dict[str, object]:
    """Parse unstructured Facebook rental listing text into structured comparison fields.

    Args:
        raw_text: Unmodified text copied from a Facebook Group listing.

    Returns:
        Dict with keys `bedrooms` (int), `neighborhood` (str),
        `asking_monthly_mxn` (float or None), and `tenant_pays_cfe` (bool).
    """
    text_lower = raw_text.lower()

    # Default to 1 bedroom if the listing omits an explicit bedroom count.
    bedrooms = 1
    br_match = _BEDROOM_RE.search(text_lower)
    if br_match:
        bedrooms = int(br_match.group(1))

    neighborhood = "Unknown"
    for zone in _KNOWN_NEIGHBORHOODS:
        if zone in text_lower:
            neighborhood = zone.title()
            break

    price_mxn = None
    price_match = _MONTHLY_PRICE_RE.search(text_lower)
    if price_match:
        # Strip thousands separators (both "," and "." are used in MX listings).
        price_mxn = float(price_match.group(1).replace(",", "").replace(".", ""))

    tenant_pays_cfe = any(kw in text_lower for kw in _TENANT_PAYS_CFE_KEYWORDS)

    return {
        "bedrooms": bedrooms,
        "neighborhood": neighborhood,
        "asking_monthly_mxn": price_mxn,
        "tenant_pays_cfe": tenant_pays_cfe,
    }
