"""
Streamlit UI package (`src/ui`) for the Rental Strategy Optimizer.

Splits the bilingual (EN / ES) web app into focused, testable modules:
  - `i18n`: Language selection helper and centralized bilingual copy.
  - `presets`: Typed scenario presets (no fragile label string-matching).
  - `sidebar`: Property, lease offer, operating cost, and CFE/uncertainty inputs.
  - `charts`: Plotly histogram and monthly net-income bar chart builders.
  - `tabs`: Presentation logic for each of the four dashboard tabs.
"""
