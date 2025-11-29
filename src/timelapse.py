from typing import List, Tuple, Optional
import tempfile
import os

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import imageio


def prepare_rgb_stack(
    dataset: xr.Dataset,
    bands: List[str],
) -> xr.DataArray:
    """
    Returns a DataArray with dimensions: (band, time, y, x)
    """
    rgb_ds = dataset[bands]
    stack = rgb_ds.to_array("band")  # dims: band, time, y, x
    return stack


def render_frames_rgb(
    rgb_stack: xr.DataArray,
    mode_label: str,
) -> Tuple[List[np.ndarray], np.ndarray]:
    """
    Creates RGB frames from a band stack.
    """
    frames: List[np.ndarray] = []
    times = rgb_stack["time"].values

    for idx in range(rgb_stack.sizes["time"]):
        # select (band, y, x)
        slice_ = rgb_stack.isel(time=idx)
        # reorder to (y, x, band)
        array = slice_.transpose("y", "x", "band").values
        array = np.clip(array / 4000.0, 0, 1)
        rgb_image = (array * 255).astype(np.uint8)

        fig, ax = plt.subplots(figsize=(5, 5), dpi=100)
        ax.imshow(rgb_image)
        ax.axis("off")
        title_date = str(np.datetime_as_string(times[idx], unit="D"))
        ax.set_title(f"{mode_label} – {title_date}", fontsize=11)

        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        rgba = np.frombuffer(fig.canvas.renderer.buffer_rgba(), dtype=np.uint8)
        frame = rgba.reshape((h, w, 4))[..., :3]
        frames.append(frame)
        plt.close(fig)

    return frames, times


def render_frames_ndvi(
    ndvi: xr.DataArray,
    mode_label: str = "NDVI",
) -> Tuple[List[np.ndarray], np.ndarray]:
    """
    Creates colored NDVI frames (using RDYlGn colormap).
    """
    frames: List[np.ndarray] = []
    times = ndvi["time"].values

    for idx in range(ndvi.sizes["time"]):
        slice_ = ndvi.isel(time=idx).values
        slice_ = np.clip(slice_, 0, 1)

        cmapped = plt.cm.RdYlGn(slice_)[..., :3]
        image = (cmapped * 255).astype(np.uint8)

        fig, ax = plt.subplots(figsize=(5, 5), dpi=100)
        ax.imshow(image)
        ax.axis("off")
        title_date = str(np.datetime_as_string(times[idx], unit="D"))
        ax.set_title(f"{mode_label} – {title_date}", fontsize=11)

        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        rgba = np.frombuffer(fig.canvas.renderer.buffer_rgba(), dtype=np.uint8)
        frame = rgba.reshape((h, w, 4))[..., :3]
        frames.append(frame)
        plt.close(fig)

    return frames, times


def export_video(frames: List[np.ndarray], fps: int) -> str:
    """
    Writes frames to a temporary MP4 and returns the file path.
    """
    if not frames:
        raise ValueError("No frames to write.")

    tmp_file = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp_path = tmp_file.name
    tmp_file.close()

    writer = imageio.get_writer(
        tmp_path,
        fps=fps,
        codec="libx264",
        macro_block_size=None,
    )
    for frame in frames:
        writer.append_data(frame)
    writer.close()

    return tmp_path
