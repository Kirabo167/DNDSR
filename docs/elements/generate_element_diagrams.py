# -*- coding: utf-8 -*-
"""Generate 12 standard finite-element node-ordering diagrams (docs/elements/*_nodes.png).

These images are referenced by docs/presentations/res_manifest.txt but were never
committed to the repository. Numbering follows the VTK convention used by DNDSR.
"""
import os
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = "/mnt/ssd-SATARAID5/home/mrz/projects/DNDSR/docs/elements"
os.makedirs(OUT_DIR, exist_ok=True)

# ── projection helpers ────────────────────────────────────────────────────
def proj(p, azim_deg, tilt_deg):
    x, y, z = p
    a = math.radians(azim_deg)
    t = math.radians(tilt_deg)
    x1 = x * math.cos(a) - y * math.sin(a)
    y1 = x * math.sin(a) + y * math.cos(a)
    y2 = y1 * math.cos(t) - z * math.sin(t)
    z2 = y1 * math.sin(t) + z * math.cos(t)
    return x1, y2, z2

def mid(p, q):
    return tuple((a + b) / 2.0 for a, b in zip(p, q))

# ── element definitions: nodes + edges + view ─────────────────────────────
def cube():
    n = [(x, y, z) for z in (0, 1) for y in (0, 1) for x in (0, 1)]
    # reorder to VTK Hex8: 0..3 bottom CCW, 4..7 top
    n = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
         (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)]
    return n

ELEMS = {}

# ---- 2D ----
ELEMS["Tri3"] = dict(
    nodes=[(0, 0), (1, 0), (0, 1)],
    edges=[(0, 1), (1, 2), (2, 0)],
    corners={0, 1, 2},
)
ELEMS["Tri6"] = dict(
    nodes=[(0, 0), (1, 0), (0, 1), (0.5, 0), (0.5, 0.5), (0, 0.5)],
    edges=[(0, 1), (1, 2), (2, 0)],
    corners={0, 1, 2},
)
ELEMS["Quad4"] = dict(
    nodes=[(0, 0), (1, 0), (1, 1), (0, 1)],
    edges=[(0, 1), (1, 2), (2, 3), (3, 0)],
    corners={0, 1, 2, 3},
)
ELEMS["Quad9"] = dict(
    nodes=[(0, 0), (1, 0), (1, 1), (0, 1),
           (0.5, 0), (1, 0.5), (0.5, 1), (0, 0.5), (0.5, 0.5)],
    edges=[(0, 1), (1, 2), (2, 3), (3, 0)],
    corners={0, 1, 2, 3},
)

# ---- 3D ----
tet = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)]
ELEMS["Tet4"] = dict(
    nodes=tet,
    edges=[(0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3)],
    corners={0, 1, 2, 3},
    view=(-58, 24),
)
ELEMS["Tet10"] = dict(
    nodes=tet + [mid(tet[0], tet[1]), mid(tet[1], tet[2]), mid(tet[2], tet[0]),
                 mid(tet[0], tet[3]), mid(tet[1], tet[3]), mid(tet[2], tet[3])],
    edges=[(0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3)],
    corners={0, 1, 2, 3},
    view=(-58, 24),
)

hex8 = cube()
ELEMS["Hex8"] = dict(
    nodes=hex8,
    edges=[(0, 1), (1, 2), (2, 3), (3, 0),
           (4, 5), (5, 6), (6, 7), (7, 4),
           (0, 4), (1, 5), (2, 6), (3, 7)],
    corners=set(range(8)),
    view=(-32, 22),
)
ELEMS["Hex27"] = dict(
    nodes=hex8
    + [mid(hex8[a], hex8[b]) for a, b in
       [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7)]]
    + [mid(hex8[a], hex8[b]) for a, b in [(0, 2), (4, 6), (0, 5), (1, 6), (2, 7), (3, 4)]]
    + [(0.5, 0.5, 0.5)],
    edges=[(0, 1), (1, 2), (2, 3), (3, 0),
           (4, 5), (5, 6), (6, 7), (7, 4),
           (0, 4), (1, 5), (2, 6), (3, 7)],
    corners=set(range(8)),
    view=(-32, 22),
)

prism = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 0, 1), (0, 1, 1)]
ELEMS["Prism6"] = dict(
    nodes=prism,
    edges=[(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3), (0, 3), (1, 4), (2, 5)],
    corners={0, 1, 2, 3, 4, 5},
    view=(38, 20),
)
ELEMS["Prism18"] = dict(
    nodes=prism
    + [mid(prism[a], prism[b]) for a, b in
       [(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3), (0, 3), (1, 4), (2, 5)]]
    + [mid(prism[a], prism[b]) for a, b in [(0, 2), (3, 5), (0, 4)]],
    edges=[(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3), (0, 3), (1, 4), (2, 5)],
    corners={0, 1, 2, 3, 4, 5},
    view=(38, 20),
)

pyr = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0), (0.5, 0.5, 1)]
ELEMS["Pyramid5"] = dict(
    nodes=pyr,
    edges=[(0, 1), (1, 2), (2, 3), (3, 0), (0, 4), (1, 4), (2, 4), (3, 4)],
    corners={0, 1, 2, 3, 4},
    view=(-45, 28),
)
ELEMS["Pyramid14"] = dict(
    nodes=pyr
    + [mid(pyr[a], pyr[b]) for a, b in
       [(0, 1), (1, 2), (2, 3), (3, 0), (0, 4), (1, 4), (2, 4), (3, 4)]]
    + [(0.5, 0.5, 0)],
    edges=[(0, 1), (1, 2), (2, 3), (3, 0), (0, 4), (1, 4), (2, 4), (3, 4)],
    corners={0, 1, 2, 3, 4},
    view=(-45, 28),
)

# ── draw ──────────────────────────────────────────────────────────────────
ACCENT = "#522aca"

for name, spec in ELEMS.items():
    nodes = spec["nodes"]
    edges = spec["edges"]
    corners = spec["corners"]
    is3d = len(nodes[0]) == 3
    if is3d:
        az, ti = spec["view"]
        pts = [proj(p, az, ti) for p in nodes]
        # depth-sort edges: far edges lighter
        edges_sorted = sorted(edges, key=lambda e: -(pts[e[0]][2] + pts[e[1]][2]))
    else:
        pts = [(x, y, 0.0) for x, y in nodes]
        edges_sorted = edges

    fig, ax = plt.subplots(figsize=(4.6, 3.6), dpi=160)
    ax.set_aspect("equal")
    ax.axis("off")

    for i, e in enumerate(edges_sorted):
        (x1, y1, z1), (x2, y2, z2) = pts[e[0]], pts[e[1]]
        depth = 0.5 + 0.5 * (z1 + z2) / 2.0 if is3d else 1.0
        ax.plot([x1, x2], [y1, y2], color=str(0.75 * depth + 0.15),
                linewidth=1.6, zorder=1, solid_capstyle="round")

    for i, (x, y, z) in enumerate(pts):
        is_corner = i in corners
        ax.scatter([x], [y], s=60 if is_corner else 34,
                   facecolor=ACCENT if is_corner else "white",
                   edgecolor=ACCENT, linewidth=1.2, zorder=3)
        ax.annotate(str(i), (x, y),
                    xytext=(6, 5), textcoords="offset points",
                    fontsize=9.5, fontweight="bold" if is_corner else "normal",
                    color="#201e1c", zorder=4)

    ax.set_xlim(min(p[0] for p in pts) - 0.25, max(p[0] for p in pts) + 0.35)
    ax.set_ylim(min(p[1] for p in pts) - 0.25, max(p[1] for p in pts) + 0.30)
    ax.set_title(name, fontsize=13, fontweight="bold", color=ACCENT, pad=6)
    fig.tight_layout(pad=0.6)
    out = os.path.join(OUT_DIR, name + "_nodes.png")
    fig.savefig(out, facecolor="white")
    plt.close(fig)
    print("saved", out)
