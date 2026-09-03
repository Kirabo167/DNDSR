#!/usr/bin/env python3
"""Render the ACM2D pressure history over a selected pseudo-step range."""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np

from render_acm2d_history import build_triangulation, encode_video, step_from_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_directory", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--prefix", default="acm2D_step")
    parser.add_argument("--max-step", type=int, default=2000)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--xlim", nargs=2, type=float, default=(-5.0, 35.0))
    parser.add_argument("--ylim", nargs=2, type=float, default=(-6.0, 6.0))
    parser.add_argument("--contours", action="store_true")
    parser.add_argument("--contour-levels", type=int, default=21)
    return parser.parse_args()


def pressure_limit(paths: list[str]) -> float:
    absolute_maximum = 0.0
    for path in paths:
        with h5py.File(path, "r") as handle:
            pressure = handle["VTKHDF/CellData/Pressure"][:]
        absolute_maximum = max(absolute_maximum, float(np.nanmax(np.abs(pressure))))
    return max(1.02 * absolute_maximum, 1.0e-6)


def render_frame(
    path: str,
    output: Path,
    triangulation: mtri.Triangulation,
    owners: np.ndarray,
    limit: float,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    contour_count: int,
) -> None:
    with h5py.File(path, "r") as handle:
        pressure = handle["VTKHDF/CellData/Pressure"][:]

    step = step_from_path(path)
    figure, axis = plt.subplots(figsize=(12.8, 7.2), dpi=100, constrained_layout=True)
    image = axis.tripcolor(
        triangulation,
        facecolors=pressure[owners],
        shading="flat",
        cmap="RdBu_r",
        vmin=-limit,
        vmax=limit,
        rasterized=True,
    )
    if contour_count > 0:
        triangle_pressure = pressure[owners]
        triangle_nodes = triangulation.triangles
        node_sum = np.bincount(
            triangle_nodes.ravel(),
            weights=np.repeat(triangle_pressure, 3),
            minlength=triangulation.x.size,
        )
        node_count = np.bincount(
            triangle_nodes.ravel(),
            minlength=triangulation.x.size,
        )
        node_pressure = np.divide(
            node_sum,
            node_count,
            out=np.zeros_like(node_sum),
            where=node_count > 0,
        )
        levels = np.linspace(-limit, limit, contour_count)
        axis.tricontour(
            triangulation,
            node_pressure,
            levels=levels,
            colors="black",
            linewidths=0.5,
            alpha=0.55,
        )
    axis.set_xlim(*xlim)
    axis.set_ylim(*ylim)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("x / D", fontsize=13)
    axis.set_ylabel("y / D", fontsize=13)
    axis.set_title(f"ACM2D pressure evolution — pseudo-step {step}", fontsize=17)
    axis.tick_params(labelsize=11)
    colorbar = figure.colorbar(image, ax=axis, pad=0.025, fraction=0.04)
    colorbar.set_label(r"Pressure $p$", fontsize=13)
    colorbar.ax.tick_params(labelsize=11)
    figure.savefig(output, facecolor="white")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    pattern = str(args.input_directory / f"{args.prefix}_*.vtkhdf")
    paths = [
        path
        for path in sorted(glob.glob(pattern), key=step_from_path)
        if step_from_path(path) <= args.max_step
    ]
    if not paths:
        raise SystemExit(f"No frames at or before step {args.max_step} match {pattern}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame_directory = args.output.parent / f".{args.output.stem}_frames"
    frame_directory.mkdir(parents=True, exist_ok=True)
    _, triangulation, owners = build_triangulation(paths[0])
    limit = pressure_limit(paths)

    frame_paths: list[Path] = []
    for index, path in enumerate(paths):
        frame_path = frame_directory / f"frame_{index:05d}.png"
        render_frame(
            path,
            frame_path,
            triangulation,
            owners,
            limit,
            tuple(args.xlim),
            tuple(args.ylim),
            args.contour_levels if args.contours else 0,
        )
        frame_paths.append(frame_path)
        print(f"Rendered {index + 1}/{len(paths)}: step {step_from_path(path)}", flush=True)

    preview = args.output.with_suffix(".png")
    preview.write_bytes(frame_paths[-1].read_bytes())
    encode_video(frame_paths, args.output, args.fps)
    print(f"Pressure range: {-limit:.8g} to {limit:.8g}")
    print(preview)
    print(args.output)


if __name__ == "__main__":
    main()
