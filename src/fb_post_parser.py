"""
Safe Copy-Paste Parser for Facebook Group Rental Posts (0% Account Ban Risk).
Extracts bedroom count, neighborhood, monthly asking rent in MXN, and whether CFE
electricity is excluded from raw text copied out of Playa del Carmen or Tulum groups.
"""
import re
from typing import Dict


def parse_facebook_rental_post(raw_text: str) -> Dict[str, object]:
    """Parses unstructured Facebook rental listing text into structured fields."""
    text_lower = raw_text.lower()

    bedrooms = 1
    br_match = re.search(r"(\d)\s*(?:br|bed|rec[aĂˇ]mara|habitaci[oĂł]n)", text_lower)
    if br_match:
        bedrooms = int(br_match.group(1))

    neighborhood = "Unknown"
    for zone in ("gonzalo guerrero", "centro", "la veleta", "aldea zama", "roma norte", "condesa"):
        if zone in text_lower:
            neighborhood = zone.title()
            break

    price_mxn = None
    price_match = re.search(r"\$?\s*(\d{2}[,\.]?\d{3})\s*(?:mxn|pesos|/mes|monthly)?", text_lower)
    if price_match:
        price_mxn = float(price_match.group(1).replace(",", "").replace(".", ""))

    tenant_pays_cfe = any(
        kw in text_lower for kw in ("+ cfe", "plus cfe", "mas cfe", "mĂˇs cfe", "plus electric", "no incluye luz")
    )

    return {
        "bedrooms": bedrooms,
        "neighborhood": neighborhood,
        "asking_monthly_mxn": price_mxn,
        "tenant_pays_cfe": tenant_pays_cfe,
    }