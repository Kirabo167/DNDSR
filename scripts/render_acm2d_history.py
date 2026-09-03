#!/usr/bin/env python3
"""Render ACM2D VTK-HDF history as a density/u/v comparison video."""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys
import time
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_directory", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--prefix", default="acm2D_step")
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--xlim", nargs=2, type=float, default=(-5.0, 35.0))
    parser.add_argument("--ylim", nargs=2, type=float, default=(-6.0, 6.0))
    parser.add_argument("--preview-only", action="store_true")
    parser.add_argument("--wait-for-step", type=int, default=0)
    parser.add_argument("--max-step", type=int)
    parser.add_argument("--u-range", nargs=2, type=float)
    parser.add_argument("--v-range", nargs=2, type=float)
    parser.add_argument("--contours", action="store_true")
    parser.add_argument("--contour-levels", type=int, default=41)
    return parser.parse_args()


def step_from_path(path: str) -> int:
    match = re.search(r"_(\d+)\.vtkhdf$", path)
    if not match:
        raise ValueError(f"Cannot obtain step from {path}")
    return int(match.group(1))


def build_triangulation(path: str) -> tuple[np.ndarray, mtri.Triangulation, np.ndarray]:
    with h5py.File(path, "r") as handle:
        group = handle["VTKHDF"]
        points = group["Points"][:, :2]
        connectivity = group["Connectivity"][:]
        offsets = group["Offsets"][:]
        cell_types = group["Types"][:]

    triangles: list[list[int]] = []
    owners: list[int] = []
    for cell, vtk_type in enumerate(cell_types):
        nodes = connectivity[offsets[cell] : offsets[cell + 1]]
        if vtk_type == 5 and len(nodes) == 3:  # VTK_TRIANGLE
            triangles.append([int(nodes[0]), int(nodes[1]), int(nodes[2])])
            owners.append(cell)
        elif vtk_type == 9 and len(nodes) == 4:  # VTK_QUAD
            triangles.append([int(nodes[0]), int(nodes[1]), int(nodes[2])])
            triangles.append([int(nodes[0]), int(nodes[2]), int(nodes[3])])
            owners.extend((cell, cell))
        else:
            raise ValueError(f"Unsupported VTK cell type {vtk_type} with {len(nodes)} nodes")

    triangulation = mtri.Triangulation(points[:, 0], points[:, 1], np.asarray(triangles))
    return points, triangulation, np.asarray(owners, dtype=np.int64)


def velocity_ranges(paths: list[str]) -> tuple[tuple[float, float], tuple[float, float]]:
    u_min, u_max = np.inf, -np.inf
    v_abs_max = 0.0
    for path in paths:
        with h5py.File(path, "r") as handle:
            velocity = handle["VTKHDF/CellData/Velocity"][:]
        u_min = min(u_min, float(np.nanmin(velocity[:, 0])))
        u_max = max(u_max, float(np.nanmax(velocity[:, 0])))
        v_abs_max = max(v_abs_max, float(np.nanmax(np.abs(velocity[:, 1]))))
    u_padding = max(0.02 * (u_max - u_min), 1e-3)
    v_abs_max = max(v_abs_max * 1.02, 1e-3)
    return (u_min - u_padding, u_max + u_padding), (-v_abs_max, v_abs_max)


def render_frame(
    path: str,
    frame_path: Path,
    triangulation: mtri.Triangulation,
    owners: np.ndarray,
    u_range: tuple[float, float],
    v_range: tuple[float, float],
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    contour_count: int,
) -> None:
    with h5py.File(path, "r") as handle:
        velocity = handle["VTKHDF/CellData/Velocity"][:]
    step = step_from_path(path)
    rho = np.ones(velocity.shape[0])
    fields = (
        (rho, r"Density $\rho$ (constant)", "viridis", (0.9995, 1.0005)),
        (velocity[:, 0], r"Velocity $u$", "turbo", u_range),
        (velocity[:, 1], r"Velocity $v$", "RdBu_r", v_range),
    )

    figure, axes = plt.subplots(1, 3, figsize=(16, 5.1), dpi=120, constrained_layout=True)
    figure.patch.set_facecolor("white")
    for axis, (values, title, cmap, limits) in zip(axes, fields):
        image = axis.tripcolor(
            triangulation,
            facecolors=values[owners],
            shading="flat",
            cmap=cmap,
            vmin=limits[0],
            vmax=limits[1],
            rasterized=True,
        )
        if contour_count > 0 and float(np.nanmax(values) - np.nanmin(values)) > 1.0e-14:
            triangle_values = values[owners]
            triangle_nodes = triangulation.triangles
            node_sum = np.bincount(
                triangle_nodes.ravel(),
                weights=np.repeat(triangle_values, 3),
                minlength=triangulation.x.size,
            )
            node_count = np.bincount(
                triangle_nodes.ravel(),
                minlength=triangulation.x.size,
            )
            node_values = np.divide(
                node_sum,
                node_count,
                out=np.zeros_like(node_sum),
                where=node_count > 0,
            )
            axis.tricontour(
                triangulation,
                node_values,
                levels=np.linspace(limits[0], limits[1], contour_count),
                colors="black",
                linewidths=0.45,
                alpha=0.55,
            )
        axis.set_xlim(*xlim)
        axis.set_ylim(*ylim)
        axis.set_aspect("equal", adjustable="box")
        axis.set_title(title, fontsize=15)
        axis.set_xlabel("x / D")
        axis.set_ylabel("y / D")
        axis.tick_params(labelsize=9)
        colorbar = figure.colorbar(image, ax=axis, orientation="horizontal", pad=0.12, fraction=0.06)
        colorbar.ax.tick_params(labelsize=9)
        if float(np.nanmax(values) - np.nanmin(values)) <= 1.0e-14:
            colorbar.set_ticks([float(values[0])])
            colorbar.set_ticklabels([f"{float(values[0]):.1f}"])
    figure.suptitle(f"ACM2D cylinder flow evolution — pseudo-step {step}", fontsize=17)
    figure.savefig(frame_path, facecolor="white")
    plt.close(figure)


def encode_video(frame_paths: list[Path], output: Path, fps: int) -> None:
    sys.path.insert(0, "/tmp/dndsr_video_deps")
    import imageio_ffmpeg

    first = np.asarray(Image.open(frame_paths[0]).convert("RGB"))
    height, width = first.shape[:2]
    writer = imageio_ffmpeg.write_frames(
        str(output),
        (width, height),
        fps=fps,
        codec="libx264",
        quality=7,
        pix_fmt_in="rgb24",
        pix_fmt_out="yuv420p",
        output_params=["-movflags", "+faststart"],
    )
    writer.send(None)
    try:
        for frame_path in frame_paths:
            frame = np.asarray(Image.open(frame_path).convert("RGB"))
            writer.send(frame)
        for _ in range(fps):
            writer.send(np.asarray(Image.open(frame_paths[-1]).convert("RGB")))
    finally:
        writer.close()


def main() -> None:
    args = parse_args()
    pattern = str(args.input_directory / f"{args.prefix}_*.vtkhdf")
    paths = sorted(glob.glob(pattern), key=step_from_path)
    if args.max_step is not None:
        paths = [path for path in paths if step_from_path(path) <= args.max_step]
    if not paths:
        raise SystemExit(f"No frames match {pattern}")

    _, triangulation, owners = build_triangulation(paths[0])
    detected_u_range, detected_v_range = velocity_ranges(paths)
    u_range = tuple(args.u_range) if args.u_range else detected_u_range
    v_range = tuple(args.v_range) if args.v_range else detected_v_range
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame_directory = args.output.parent / f".{args.output.stem}_frames"
    frame_directory.mkdir(parents=True, exist_ok=True)

    if args.preview_only:
        preview = args.output.with_suffix(".png")
        render_frame(
            paths[-1], preview, triangulation, owners, u_range, v_range,
            tuple(args.xlim), tuple(args.ylim),
            args.contour_levels if args.contours else 0,
        )
        print(preview)
        return

    frame_paths: list[Path] = []
    rendered_paths: set[str] = set()
    while True:
        paths = sorted(glob.glob(pattern), key=step_from_path)
        if args.max_step is not None:
            paths = [path for path in paths if step_from_path(path) <= args.max_step]
        for path in paths:
            if path in rendered_paths:
                continue
            frame_path = frame_directory / f"frame_{len(frame_paths):05d}.png"
            try:
                render_frame(
                    path, frame_path, triangulation, owners, u_range, v_range,
                    tuple(args.xlim), tuple(args.ylim),
                    args.contour_levels if args.contours else 0,
                )
            except (OSError, KeyError):
                continue
            rendered_paths.add(path)
            frame_paths.append(frame_path)
            print(f"Rendered {len(frame_paths)}: step {step_from_path(path)}", flush=True)

        latest_step = max((step_from_path(path) for path in rendered_paths), default=-1)
        if args.wait_for_step <= 0 or latest_step >= args.wait_for_step:
            break
        time.sleep(5)

    encode_video(frame_paths, args.output, args.fps)
    print(args.output)


if __name__ == "__main__":
    main()
