# Take the  binarized images as inputs, assign to black conductivity, to white around 0
# Give this conductivity grid to differentiable Fourier solver, periodic boundaries everywhere, difference of 1K top to bottom
# Evaluate the solution and then the effective conductivity of the structure


# Exper9memt using the augmented dataset. Compute effective conductivity for all of them. Find max and min of this dataset. Plot the final temperature solution and above the effective conductivity for the max and min

# We will add this to the loss, if the reconstructed value has effective conductiivty less than the minimum found in the dataset then penalize


# Example (kept for reference only, now superseded by real implementation below).
'''
from matinverse import Geometry2D, BoundaryConditions, Fourier
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import time
import numpy as np
import matplotlib.colors as mcolors
'''

from typing import Union, Tuple, Any

import numpy as np
from PIL import Image
import os


def _to_gray_batch(design: Union[np.ndarray, Any]) -> np.ndarray:
    """Convert input designs to a NumPy batch of grayscale images in [0,1].

    Accepts:
      - np.ndarray or torch.Tensor
      - shapes: (H,W), (H,W,C), (C,H,W), (B,H,W), (B,H,W,C), (B,C,H,W)
    Returns:
      - gray: np.ndarray, shape (B, H, W), dtype float32 in [0,1].
    """
    # Lazy import to avoid making torch a hard dependency for callers
    try:
        import torch  # type: ignore
    except Exception:  # pragma: no cover - torch may not be installed
        torch = None

    x = design
    if torch is not None and isinstance(x, torch.Tensor):  # type: ignore[name-defined]
        x = x.detach().cpu().numpy()
    else:
        x = np.asarray(x)

    # Ensure float32
    x = x.astype(np.float32)

    if x.ndim == 2:
        # (H, W) -> (1, H, W, 1)
        x = x[None, :, :, None]
    elif x.ndim == 3:
        # Three cases:
        # (H, W, C): single image with channels last
        # (C, H, W): single image with channels first
        # (B, H, W): batch of grayscale images
        if x.shape[-1] in (1, 3):
            # (H, W, C)
            x = x[None, ...]  # -> (1, H, W, C)
        elif x.shape[0] in (1, 3) and x.shape[2] not in (1, 3):
            # (C, H, W)
            x = np.transpose(x, (1, 2, 0))  # -> (H, W, C)
            x = x[None, ...]                # -> (1, H, W, C)
        else:
            # (B, H, W): batch of grayscale images
            x = x[..., None]                # -> (B, H, W, 1)
    elif x.ndim == 4:
        # (B, H, W, C) or (B, C, H, W)
        if x.shape[-1] in (1, 3):
            # already (B, H, W, C)
            pass
        else:
            # assume (B, C, H, W)
            x = np.transpose(x, (0, 2, 3, 1))  # -> (B, H, W, C)
    else:
        raise ValueError(f"Unsupported design array with shape {x.shape}")

    # Now x is (B, H, W, C)
    if x.shape[-1] == 1:
        gray = x[..., 0]
    else:
        gray = x.mean(axis=-1)

    # Normalize/clamp to [0,1]
    gray = np.clip(gray, 0.0, 1.0).astype(np.float32)
    return gray


def _resize_to_square(gray: np.ndarray, size: int = 128) -> np.ndarray:
    """Resize each grayscale image in a batch to (size, size) via PIL.

    gray: (B, H, W) in [0,1]
    returns: (B, size, size) in [0,1]
    """
    if gray.ndim != 3:
        raise ValueError(f"Expected gray of shape (B,H,W), got {gray.shape}")

    B = gray.shape[0]
    out = np.zeros((B, size, size), dtype=np.float32)
    for i in range(B):
        im = Image.fromarray((gray[i] * 255.0).astype(np.uint8))
        im_resized = im.resize((size, size), resample=Image.BILINEAR)
        out[i] = np.asarray(im_resized).astype(np.float32) / 255.0
    return out


def _fourier_solver_single(conductivity_matrix: np.ndarray) -> Tuple[np.ndarray, float]:
    """Solve potential and effective conductivity for a single 2D field.

    Args:
        conductivity_matrix: 2D array (N, N) with conductivity values.

    Returns:
        potential: (N, N) temperature / potential field.
        kappa_eff: scalar effective conductivity.
    """
    from matinverse import Geometry2D, BoundaryConditions, Fourier  # type: ignore
    import jax.numpy as jnp  # type: ignore

    cond = np.asarray(conductivity_matrix, dtype=np.float32)
    if cond.ndim != 2:
        raise ValueError(f"Expected 2D conductivity matrix, got {cond.shape}")

    H, W = cond.shape
    if H != W:
        N = min(H, W)
        cond = cond[:N, :N]
    else:
        N = H

    # Optionally downsample very large grids for speed (not critical here)
    if N > 128:
        factor = max(1, N // 128)
        cond = cond[::factor, ::factor]
        N = cond.shape[0]

    cond_flat = cond.flatten()
    cond_flat = jnp.array(cond_flat, dtype=jnp.float32)

    size = [1.0, 1.0]
    grid = [N, N]

    geo = Geometry2D(grid, size, periodic=[True, True])
    fourier = Fourier(geo)

    bcs = BoundaryConditions(geo)
    bcs.periodic('y', lambda batch, space, t: 1.0)
    bcs.periodic('x', lambda batch, space, t: 0.0)

    cond_field = cond_flat

    def kappa_map(batch, space, temp, t):
        sigma = cond_field[space]
        return jnp.array([[sigma, 0.0], [0.0, sigma]], dtype=jnp.float32)

    output = fourier(kappa_map, bcs, batch_size=1)

    potential = np.array(output['T']).reshape((N, N))
    kappa_eff = float(output['kappa_effective'])
    return potential, kappa_eff


def fourier_solver(conductivity: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Run the Fourier solver on one or more conductivity fields.

    Args:
      conductivity: np.ndarray of shape (N, N) or (B, N, N).
    Returns:
      T: np.ndarray of shape (B, N, N) with temperature fields.
      kappa_effective: np.ndarray of shape (B,) effective conductivities.
    """
    cond = np.asarray(conductivity, dtype=np.float32)

    if cond.ndim == 2:
        potential, kappa_eff = _fourier_solver_single(cond)
        return potential[None, ...], np.array([kappa_eff], dtype=np.float32)

    if cond.ndim != 3 or cond.shape[1] != cond.shape[2]:
        raise ValueError(f"Expected conductivity of shape (B,N,N) or (N,N), got {cond.shape}")

    B, N, _ = cond.shape
    potentials = []
    kappas = []
    for i in range(B):
        pot_i, k_i = _fourier_solver_single(cond[i])
        potentials.append(pot_i)
        kappas.append(k_i)

    T = np.stack(potentials, axis=0)
    kappa_effective = np.array(kappas, dtype=np.float32)
    return T, kappa_effective


def plot_temperature_fourier(Temperatures: np.ndarray,
                             base_conductivities: np.ndarray,
                             index: int = 0,
                             out_path: str = None,
                             title_prefix: str = "") -> None:
    """Plot masked temperature field for a single design.

    Temperatures: (B, N, N)
    base_conductivities: (B, N, N)
    index: which sample to plot
    out_path: optional path to save the figure; if None, just shows it.
    """
    import matplotlib.pyplot as plt  # local import to avoid issues in headless envs
    import matplotlib.colors as mcolors
    import numpy as np

    cmap = plt.cm.viridis
    norm = mcolors.Normalize(vmin=Temperatures[index].min(), vmax=Temperatures[index].max())

    threshold = np.min(base_conductivities[index]) + 1e-6
    masked_T = np.ma.masked_where(base_conductivities[index] < threshold, Temperatures[index])

    plt.figure(figsize=(6, 5))
    im = plt.imshow(masked_T, cmap=cmap, norm=norm, interpolation="nearest")
    plt.colorbar(im, label="Temperature")
    plt.contour(
        masked_T,
        levels=np.linspace(Temperatures[index].min(), Temperatures[index].max(), 25),
        colors="white",
        linewidths=0.5,
    )

    title = f"{title_prefix}Heatmap of T (Index {index})"
    plt.title(title)
    plt.xlabel("x direction")
    plt.ylabel("y direction")
    plt.tight_layout()

    if out_path is not None:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()
    else:  # pragma: no cover - interactive usage
        plt.show()


def evaluate_design_conductivity(
    design: Union[np.ndarray, Any],
    solver_res: int = 128,
    kappa_solid: float = 1.0,
    kappa_void: float = 1e-3,
    binarize: bool = True,
    threshold: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate effective conductivity of one or more designs via Fourier solver.

    Args:
      design: image or batch of images, arbitrary layout, values in [0,1].
      solver_res: size of square homogenization cell (N = solver_res).
      kappa_solid: conductivity of solid (black) phase.
      kappa_void: conductivity of void (white) phase.
      binarize: if True, threshold the solid fraction.
      threshold: threshold for binarization in [0,1].

    Returns:
      T: temperature fields, shape (B, N, N).
      kappa_eff: effective conductivities, shape (B,).
      cond: conductivity grids used, shape (B, N, N).
    """
    gray = _to_gray_batch(design)          # (B, H, W) in [0,1], black ~ 0, white ~ 1
    gray_sq = _resize_to_square(gray, solver_res)  # (B, N, N)

    # Map black (0) -> solid fraction 1, white (1) -> solid fraction 0
    phi = 1.0 - gray_sq

    if binarize:
        phi = (phi >= threshold).astype(np.float32)

    cond = kappa_void + phi * (kappa_solid - kappa_void)

    T, kappa_eff = fourier_solver(cond)
    return T, kappa_eff, cond


def _load_augmented_images(dirs, max_images=None):
    """Load binarized augmented PNGs as grayscale arrays in [0,1]."""
    import os
    from glob import glob
    from PIL import Image

    paths = []
    for d in dirs:
        d = str(d)
        # Prefer binarized images if present
        paths += glob(os.path.join(d, "**", "*bin*.png"), recursive=True)
        if not paths:
            paths += glob(os.path.join(d, "**", "*.png"), recursive=True)

    paths = sorted(set(paths))
    if max_images is not None:
        paths = paths[:max_images]

    imgs = []
    for p in paths:
        try:
            im = Image.open(p).convert("L")  # grayscale
            arr = np.asarray(im).astype(np.float32) / 255.0
            imgs.append(arr)
        except Exception as e:  # pragma: no cover - IO issues
            print(f"Skipping {p}: {e}")

    if not imgs:
        raise RuntimeError(f"No images found in {dirs}")

    return paths, np.stack(imgs, axis=0)


if __name__ == "__main__":
    # Minimal smoke test: check that the module compiles and the API runs on dummy data.
    # For full experiments, adapt AUGMENTED_DIRS to your dataset folders.
    import os

    # Dummy checker on random binary designs
    dummy = np.random.rand(4, 128, 256).astype(np.float32)
    T, kappa_eff, cond = evaluate_design_conductivity(dummy, solver_res=128)
    print("Dummy kappa_eff:", kappa_eff)

    # Configuration for dataset experiment on generated designs
    DESIGN_DIRS = [
        "outputs/designs/mbb_beam_384x64_0.4-20260111-164330/images",
    ]

    if DESIGN_DIRS:
        paths, imgs = _load_augmented_images(DESIGN_DIRS)
        T_all, kappa_all, cond_all = evaluate_design_conductivity(imgs, solver_res=128)

        # Summary statistics over all designs
        mean_kappa = float(np.mean(kappa_all))
        std_kappa = float(np.std(kappa_all))
        print(f"Evaluated {len(paths)} designs from {DESIGN_DIRS[0]}")
        print(f"kappa_eff mean = {mean_kappa:.6f}, std = {std_kappa:.6f}")

        # Min/max designs
        idx_min = int(np.argmin(kappa_all))
        idx_max = int(np.argmax(kappa_all))

        print("Min kappa:", float(kappa_all[idx_min]), "at", paths[idx_min])
        print("Max kappa:", float(kappa_all[idx_max]), "at", paths[idx_max])

        out_dir = os.path.join("figures", "conductivity_eval")
        os.makedirs(out_dir, exist_ok=True)

        plot_temperature_fourier(
            T_all,
            cond_all,
            index=idx_min,
            out_path=os.path.join(out_dir, "min_kappa_T.png"),
            title_prefix=f"k_eff={float(kappa_all[idx_min]):.4f} | ",
        )
        plot_temperature_fourier(
            T_all,
            cond_all,
            index=idx_max,
            out_path=os.path.join(out_dir, "max_kappa_T.png"),
            title_prefix=f"k_eff={float(kappa_all[idx_max]):.4f} | ",
        )

        # Connectivity cut experiment: take the max-conductivity design
        # and introduce a thick white (void) band across the middle.
        max_img = imgs[idx_max].copy()  # (H, W) in [0,1]
        H, W = max_img.shape
        band_thickness = max(1, H // 16)
        band_start = H // 2 - band_thickness // 2
        band_end = band_start + band_thickness

        max_img_cut = max_img.copy()
        # White = void (low conductivity)
        max_img_cut[band_start:band_end, :] = 1.0

        T_cut, kappa_cut, cond_cut = evaluate_design_conductivity(
            max_img_cut[None, ...], solver_res=128
        )

        print("Connectivity cut experiment:")
        print(
            f"Original max kappa_eff = {float(kappa_all[idx_max]):.6f}, "
            f"with middle cut = {float(kappa_cut[0]):.6f}"
        )

        plot_temperature_fourier(
            T_cut,
            cond_cut,
            index=0,
            out_path=os.path.join(out_dir, "max_kappa_with_cut_T.png"),
            title_prefix=f"k_eff={float(kappa_cut[0]):.4f} | ",
        )



