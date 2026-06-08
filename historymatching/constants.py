"""Shared constants used across the history matching package."""

# DataFrame column schemas for the core domain objects.
PARAMETER_SPACE_COLUMNS = ["parameter", "minimum", "maximum"]
OBSERVATIONS_COLUMNS = ["feature", "mean", "std"]

# Shared kwargs for sc.savefig().
SAVE_KW = dict(dpi=150, bbox_inches="tight")

# Plot palette and colormap, shared across plotting.py, engine.py, and iteration_result.py.
NROY_COLOR = "#3575b5"     # steel blue — the surviving / non-implausible cloud
PRIOR_COLOR = "#bcbcbc"    # grey — the prior / earlier-wave cloud
TRUTH_COLOR = "#d44d4d"    # red — known true values (synthetic-recovery demos)
MEDIAN_COLOR = "#2a7f3f"   # green — estimated median / central value
TARGET_COLOR = "#d44d4d"   # red — observation targets
WAVE_CMAP = "plasma"       # colormap for per-wave / per-sample gradients
