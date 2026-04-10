"""Plot the geographic locations of all NEON sites in the dataset."""

import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

# ---------------------------------------------------------------------------
# Site data
# ---------------------------------------------------------------------------

SITES = [
    # code, full name, lat, lon, split
    # "both"     = train + test
    # "train"    = train only
    # "test"     = test only
    # "excluded" = missing CHM in train partition
    ("BART", "Bartlett Exp. Forest", 44.06, -71.29, "both"),
    ("DELA", "Dead Lake", 32.54, -87.80, "both"),
    ("HARV", "Harvard Forest", 42.54, -72.17, "both"),
    ("JERC", "Jones Ecol. Research Ctr.", 31.19, -84.47, "both"),
    ("LENO", "Lenoir Landing", 31.85, -88.16, "both"),
    ("MLBS", "Mountain Lake Biol. Station", 37.38, -80.52, "both"),
    ("NIWO", "Niwot Ridge", 40.05, -105.58, "both"),
    ("OSBS", "Ordway-Swisher Biol. Station", 29.69, -81.99, "both"),
    ("SJER", "San Joaquin Exp. Range", 37.11, -119.73, "both"),
    ("TEAK", "Teakettle", 37.01, -119.01, "both"),
    ("TOOL", "Toolik", 68.66, -149.37, "train"),
    ("ABBY", "Abby Road", 45.76, -122.33, "test"),
    ("BLAN", "Blandy Exp. Farm", 39.06, -78.07, "test"),
    ("BONA", "Caribou Creek / Poker Flat", 65.15, -147.50, "test"),
    ("CLBJ", "LBJ National Grassland", 33.40, -97.57, "test"),
    ("SCBI", "Smithsonian Cons. Bio. Inst.", 38.89, -78.14, "test"),
    ("SERC", "Smithsonian Env. Research Ctr.", 38.89, -76.56, "test"),
    ("SOAP", "Soaproot Saddle", 37.03, -119.26, "test"),
    ("TALL", "Talladega National Forest", 32.95, -87.39, "test"),
    ("WREF", "Wind River Exp. Forest", 45.82, -121.95, "test"),
    ("DSNY", "Disney Wilderness Preserve", 28.13, -81.44, "excluded"),
    ("ONAQ", "Onaqui-Ault", 40.18, -112.45, "excluded"),
    ("YELL", "Yellowstone", 44.95, -110.54, "excluded"),
]

STYLE = {
    #           color       marker  zorder
    "both": ("#5D3F8E", "o", 4),
    "train": ("#1A5276", "^", 4),
    "test": ("#1A7A4A", "s", 4),
    "excluded": ("#C0392B", "X", 3),
}

LEGEND_LABELS = {
    "both": "Train + Test (10 sites)",
    "train": "Train only (1 site)",
    "test": "Test only (10 sites)",
    "excluded": "Excluded, no CHM (3 sites)",
}

# ---------------------------------------------------------------------------
# Load base map
# ---------------------------------------------------------------------------

ne_url_countries = (
    "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/"
    "master/geojson/ne_110m_admin_0_countries.geojson"
)
ne_url_states = (
    "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/"
    "master/geojson/ne_110m_admin_1_states_provinces.geojson"
)

world = gpd.read_file(ne_url_countries)
states = gpd.read_file(ne_url_states)

us = world[world["ADMIN"] == "United States of America"]
us_states = states[states["admin"] == "United States of America"]

# ---------------------------------------------------------------------------
# Build figure with two axes: contiguous US + Alaska inset
# ---------------------------------------------------------------------------

fig = plt.figure(figsize=(13, 8))

# Main axis: contiguous US
ax_main = fig.add_axes([0.01, 0.12, 0.88, 0.82])
# Inset axis: Alaska
ax_ak = fig.add_axes([0.01, 0.01, 0.28, 0.30])


def draw_basemap(ax, xlim, ylim):
    us_states.plot(ax=ax, color="#F0EDE8", edgecolor="#BDBDBD", linewidth=0.4)
    us.plot(ax=ax, color="none", edgecolor="#888888", linewidth=0.8)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_facecolor("#D6EAF8")
    fig.patch.set_facecolor("white")


draw_basemap(ax_main, xlim=(-128, -65), ylim=(24, 52))
draw_basemap(ax_ak, xlim=(-170, -130), ylim=(54, 72))

# ---------------------------------------------------------------------------
# Plot sites
# ---------------------------------------------------------------------------

# Sites that go on Alaska inset (roughly west of -130 or north of 60)
ak_codes = {"TOOL", "BONA"}


def plot_sites_on(ax, codes_to_plot):
    for code, name, lat, lon, split in SITES:
        if code not in codes_to_plot:
            continue
        color, marker, zorder = STYLE[split]
        ax.scatter(
            lon,
            lat,
            c=color,
            marker=marker,
            s=70,
            zorder=zorder,
            edgecolors="white",
            linewidths=0.5,
        )
        # Label offset: nudge to avoid overlap
        dx, dy = 0.4, 0.4
        ha = "left"
        # Manual nudges for crowded areas — (dx, dy, ha)
        nudge = {
            # Southeast cluster
            "LENO": (-0.5, 0.6, "right"),
            "DELA": (0.5, 0.6, "left"),
            "TALL": (0.5, -0.7, "left"),
            "JERC": (0.5, -0.7, "left"),
            "OSBS": (0.5, 0.5, "left"),
            "DSNY": (0.5, -0.7, "left"),
            # Virginia / Maryland cluster
            "BLAN": (-0.5, 0.5, "right"),
            "SCBI": (-0.5, -0.6, "right"),
            "SERC": (0.5, 0.5, "left"),
            "MLBS": (0.5, -0.6, "left"),
            # Northeast
            "BART": (0.4, 0.5, "left"),
            "HARV": (0.4, -0.6, "left"),
            # California cluster
            "SJER": (-0.5, 0.6, "right"),
            "SOAP": (0.5, 0.6, "left"),
            "TEAK": (0.5, -0.7, "left"),
            # Pacific Northwest
            "ABBY": (-0.5, 0.5, "right"),
            "WREF": (0.5, -0.6, "left"),
            # Others
            "NIWO": (0.5, 0.4, "left"),
            "CLBJ": (0.5, 0.4, "left"),
            "YELL": (0.5, 0.4, "left"),
            "ONAQ": (0.5, 0.4, "left"),
        }
        if code in nudge:
            dx, dy, ha = nudge[code]
        ax.annotate(
            code,
            xy=(lon, lat),
            xytext=(lon + dx, lat + dy),
            fontsize=7,
            ha=ha,
            va="center",
            color="#1a1a1a",
            fontweight="semibold",
        )


main_codes = {c for c, *_ in SITES} - ak_codes
plot_sites_on(ax_main, main_codes)
plot_sites_on(ax_ak, ak_codes)

# Alaska inset border and label
for spine in ax_ak.spines.values():
    spine.set_visible(True)
    spine.set_edgecolor("#888888")
    spine.set_linewidth(0.8)
ax_ak.axis("on")
ax_ak.set_xticks([])
ax_ak.set_yticks([])
ax_ak.set_title("Alaska", fontsize=8, pad=3)

# ---------------------------------------------------------------------------
# Legend
# ---------------------------------------------------------------------------

handles = [
    mpatches.Patch(
        facecolor=STYLE[split][0],
        label=LEGEND_LABELS[split],
        linewidth=0,
    )
    for split in ["both", "train", "test", "excluded"]
]
ax_main.legend(
    handles=handles,
    loc="lower right",
    frameon=True,
    framealpha=0.9,
    edgecolor="#CCCCCC",
    fontsize=9,
    title="Split membership",
    title_fontsize=9,
)

ax_main.set_title(
    "NEON sites in the NeonTreeEvaluation dataset",
    fontsize=13,
    fontweight="bold",
    pad=10,
)

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

out_path = Path(__file__).parent.parent.parent / "data" / "neon_sites_map.pdf"
fig.savefig(out_path, dpi=200, bbox_inches="tight")
print(f"Saved to {out_path}")
plt.show()
