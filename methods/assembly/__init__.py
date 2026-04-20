"""Assembly subpackage: collect per-region metric CSVs into cross-region summaries."""

from .cross_region import assemble
from .tier_concordance import assemble_tier_concordance, build_tier_concordance
