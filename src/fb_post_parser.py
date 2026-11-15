"""
Re-export Wrapper: Safe Copy-Paste Parser for Facebook Group Rental Posts.
Delegates all logic to `src.rental_engine.fb_parser`.
"""
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.rental_engine.fb_parser import parse_facebook_rental_post  # noqa: E402

__all__ = ["parse_facebook_rental_post"]
