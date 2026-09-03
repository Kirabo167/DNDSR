#!/usr/bin/env python3
"""Render separate density, velocity-magnitude, and pressure ACM2D videos."""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
from PIL import Image

from render_acm2d_history import build_triangulation, step_from_path


FIELD_LABELS = {
    "density": r"Density $\rho$ (constant model value, not solved)",
    "velocity": r"Velocity magnitude $|\mathbf{V}|$",
    "pressure": r"Pressure $p$",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create separate ACM2D density, velocity-magnitude, and pressure videos."
    )
    parser.add_argument("input_directory", type=Path)
    parser.add_argument("output_directory", type=Path)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--case-label", required=True)
    parser.add_argument("--video-prefix", required=True)
    parser.add_argument("--max-step", type=int, default=4000)
    parser.add_argument("--physical-time-step", type=float)
    parser.add_argument("--rho0", type=float, default=1.0)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--xlim", nargs=2, type=float, default=(-5.0, 35.0))
    parser.add_argument("--ylim", nargs=2, type=float, default=(-6.0, 6.0))
    parser.add_argument("--contour-levels", type=int, default=41)
    return parser.parse_args()


def read_series_times(input_directory: Path, prefix: str) -> dict[str, float]:
    series_path = input_directory / f"{prefix}.vtkhdf.series"
    if not series_path.is_file():
        return {}
    with series_path.open("r", encoding="utf-8") as stream:
        series = json.load(stream)
    times = {
        Path(str(entry["name"])).name: float(entry["time"])
        for entry in series.get("files", [])
        if "name" in entry and "time" in entry
    }
    if not all(np.isfinite(time) for time in times.values()):
        raise ValueError(f"Non-finite time found in {series_path}")
    return times


def read_field(path: str, field: str, rho0: float) -> np.ndarray:
    with h5py.File(path, "r") as handle:
        if field == "pressure":
            return np.asarray(handle["VTKHDF/CellData/Pressure"][:], dtype=float)
        velocity = np.asarray(handle["VTKHDF/CellData/Velocity"][:], dtype=float)
    if field == "velocity":
        return np.linalg.norm(velocity[:, :2], axis=1)
    if field == "density":
        return np.full(velocity.shape[0], rho0, dtype=float)
    raise ValueError(f"Unsupported field: {field}")


def global_ranges(
    paths: list[str], rho0: float
) -> tuple[dict[str, tuple[float, float]], dict[str, tuple[float, float]]]:
    minimum_speed = np.inf
    maximum_speed = 0.0
    maximum_pressure = 0.0
    for path in paths:
        with h5py.File(path, "r") as handle:
            velocity = np.asarray(handle["VTKHDF/CellData/Velocity"][:, :2], dtype=float)
            pressure = np.asarray(handle["VTKHDF/CellData/Pressure"][:], dtype=float)
        speed = np.linalg.norm(velocity, axis=1)
        minimum_speed = min(minimum_speed, float(np.nanmin(speed)))
        maximum_speed = max(maximum_speed, float(np.nanmax(speed)))
        maximum_pressure = max(maximum_pressure, float(np.nanmax(np.abs(pressure))))
    density_padding = max(abs(rho0) * 5.0e-4, 5.0e-4)
    display_limits = {
        "density": (rho0 - density_padding, rho0 + density_padding),
        "velocity": (0.0, max(1.02 * maximum_speed, 1.0e-6)),
        "pressure": (-max(1.02 * maximum_pressure, 1.0e-6),
                     max(1.02 * maximum_pressure, 1.0e-6)),
    }
    contour_limits = {
        "density": (rho0, rho0),
        "velocity": (minimum_speed, maximum_speed),
        "pressure": (-maximum_pressure, maximum_pressure),
    }
    return display_limits, contour_limits


def cell_to_node(
    triangulation: mtri.Triangulation,
    owners: np.ndarray,
    cell_values: np.ndarray,
) -> np.ndarray:
    # A quad is represented by two plotting triangles. Deduplicate (cell,node)
    # pairs so the diagonal vertices do not receive twice the cell's weight.
    cell_node_pairs = np.column_stack(
        (
            np.repeat(owners, 3),
            triangulation.triangles.ravel(),
        )
    )
    cell_node_pairs = np.unique(cell_node_pairs, axis=0)
    cells = cell_node_pairs[:, 0]
    nodes = cell_node_pairs[:, 1]
    node_sum = np.bincount(
        nodes,
        weights=cell_values[cells],
        minlength=triangulation.x.size,
    )
    node_count = np.bincount(
        nodes,
        minlength=triangulation.x.size,
    )
    return np.divide(
        node_sum,
        node_count,
        out=np.zeros_like(node_sum),
        where=node_count > 0,
    )


def render_frame(
    source: str,
    output: Path,
    field: str,
    triangulation: mtri.Triangulation,
    owners: np.ndarray,
    limits: tuple[float, float],
    contour_limits: tuple[float, float],
    rho0: float,
    physical_time: float,
    case_label: str,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    contour_count: int,
) -> None:
    values = read_field(source, field, rho0)
    step = step_from_path(source)
    color_maps = {
        "density": "viridis",
        "velocity": "turbo",
        "pressure": "RdBu_r",
    }

    figure, axis = plt.subplots(figsize=(12.8, 7.2), dpi=100, constrained_layout=True)
    image = axis.tripcolor(
        triangulation,
        facecolors=values[owners],
        shading="flat",
        cmap=color_maps[field],
        vmin=limits[0],
        vmax=limits[1],
        rasterized=True,
    )
    if contour_count > 1 and contour_limits[1] - contour_limits[0] > 1.0e-14:
        axis.tricontour(
            triangulation,
            cell_to_node(triangulation, owners, values),
            levels=np.linspace(contour_limits[0], contour_limits[1], contour_count),
            colors="black",
            linewidths=0.45,
            alpha=0.55,
        )
    axis.set_xlim(*xlim)
    axis.set_ylim(*ylim)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("x / D", fontsize=13)
    axis.set_ylabel("y / D", fontsize=13)
    axis.set_title(
        f"{case_label} — {FIELD_LABELS[field]}\n"
        f"physical step {step},  t* = {physical_time:.2f}",
        fontsize=16,
    )
    axis.tick_params(labelsize=11)
    colorbar = figure.colorbar(image, ax=axis, pad=0.025, fraction=0.04)
    colorbar.set_label(FIELD_LABELS[field], fontsize=13)
    colorbar.ax.tick_params(labelsize=11)
    if field == "density":
        colorbar.set_ticks([rho0])
        colorbar.set_ticklabels([f"{rho0:g} (constant)"])
    figure.savefig(output, facecolor="white")
    plt.close(figure)


def encode_video(frame_paths: list[Path], output: Path, fps: int) -> None:
    if fps <= 0:
        raise ValueError("fps must be positive")
    try:
        import imageio_ffmpeg
    except ModuleNotFoundError:
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
            if frame.shape != first.shape:
                raise ValueError(f"Frame size mismatch: {frame_path}")
            writer.send(frame)
        for _ in range(fps):
            writer.send(first if len(frame_paths) == 1 else frame)
    finally:
        writer.close()


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

    if step_from_path(paths[-1]) != args.max_step:
        raise SystemExit(
            f"Last available step is {step_from_path(paths[-1])}, expected {args.max_step}"
        )

    if step_from_path(paths[0]) != 0:
        raise SystemExit(f"First available step is {step_from_path(paths[0])}, expected 0")
    if args.physical_time_step is not None and args.physical_time_step <= 0:
        raise SystemExit("--physical-time-step must be positive")

    args.output_directory.mkdir(parents=True, exist_ok=True)
    _, triangulation, owners = build_triangulation(paths[0])
    display_limits, contour_limits = global_ranges(paths, args.rho0)
    series_times = read_series_times(args.input_directory, args.prefix)

    physical_times: dict[str, float] = {}
    for path in paths:
        name = Path(path).name
        if name in series_times:
            physical_times[name] = series_times[name]
        elif args.physical_time_step is not None:
            physical_times[name] = step_from_path(path) * args.physical_time_step
        else:
            raise SystemExit(
                f"No series time for {name}; provide --physical-time-step as a fallback"
            )
    ordered_times = [physical_times[Path(path).name] for path in paths]
    if any(right <= left for left, right in zip(ordered_times, ordered_times[1:])):
        raise SystemExit("Physical times must be strictly increasing")

    for field in ("density", "velocity", "pressure"):
        output = args.output_directory / (
            f"{args.video_prefix}_{field}_contours_steps_0000_{args.max_step:04d}.mp4"
        )
        frame_directory = args.output_directory / f".{output.stem}_frames"
        frame_directory.mkdir(parents=True, exist_ok=True)
        frame_paths: list[Path] = []
        for index, path in enumerate(paths):
            frame_path = frame_directory / f"frame_{index:05d}.png"
            render_frame(
                path,
                frame_path,
                field,
                triangulation,
                owners,
                display_limits[field],
                contour_limits[field],
                args.rho0,
                physical_times[Path(path).name],
                args.case_label,
                tuple(args.xlim),
                tuple(args.ylim),
                args.contour_levels,
            )
            frame_paths.append(frame_path)
            print(
                f"Rendered {field} {index + 1}/{len(paths)}: step {step_from_path(path)}",
                flush=True,
            )

        preview = output.with_suffix(".png")
        preview.write_bytes(frame_paths[-1].read_bytes())
        encode_video(frame_paths, output, args.fps)
        print(
            f"{field} display range: {display_limits[field][0]:.8g} "
            f"to {display_limits[field][1]:.8g}"
        )
        print(
            f"{field} contour range: {contour_limits[field][0]:.8g} "
            f"to {contour_limits[field][1]:.8g}"
        )
        print(preview)
        print(output)


if __name__ == "__main__":
    main()
