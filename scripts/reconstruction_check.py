#!/usr/bin/env python3
"""
reconstruction_check.py

Load the full ViTVAE (saved with torch.save(model.state_dict(), ...)),
load images from a folder (same resize as training), run them through the
model with identical preprocessing, and save Orig / Recon / Diff for inspection.

Important: the NHWC -> NCHW permutation must match training:
  training: NHWC (N,256,128,3) -> NCHW (N,3,128,256) via permute(0,3,2,1)
"""
import os
from glob import glob
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt

# --- Config: adjust to your environment ---
MODEL_PATH = "models/vitvae_thaw2_latent16_decodersizexlarge_20260223-102323_GOOD_EPOCH15.pt"   # full model (state_dict)
IMAGE_DIR  = "outputs/designs/mbb_beam_384x64_0.4-20260111-164330/images"
OUT_DIR    = f"recon_check/{os.path.splitext(os.path.basename(MODEL_PATH))[0]}"  
BATCH_SIZE = 8
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
RESIZE     = (128, 256)  # (width, height) used in training
LATENT_DIM = 16     # must match what you trained
SIZE_DECODER = 'xlarge'

os.makedirs(OUT_DIR, exist_ok=True)

# -------------------------
# Helper functions
# -------------------------
def psnr_from_mse(mse, max_val=1.0):
    if mse <= 1e-12:
        return 99.0
    return 10.0 * np.log10((max_val ** 2) / mse)

def load_images_as_nhwc(dirpath, resize=(128,256), max_images=None):
    paths = sorted(glob(os.path.join(dirpath, "*.png")))
    if max_images:
        paths = paths[:max_images]
    imgs = []
    for p in paths:
        im = Image.open(p).convert("RGB")
        im = im.resize(resize, Image.BILINEAR)
        arr = np.asarray(im).astype(np.float32) / 255.0   # [0,1], NHWC (H=256,W=128,C=3)
        imgs.append(arr)
    if not imgs:
        raise RuntimeError("No images found in " + dirpath)
    return paths, np.stack(imgs, axis=0)  # (N, H, W, C)

def binarize_nhwc(x, thresh=0.5):
    """Binarize NHWC float images in [0,1] with a threshold."""
    return (x >= thresh).astype(np.float32)


def save_side_by_side(orig_nhwc, recon_nhwc, out_path, nshow=8):
    n = min(len(orig_nhwc), nshow)
    fig, axes = plt.subplots(3, n, figsize=(1.6*n, 4.8))
    for i in range(n):
        axes[0, i].imshow(orig_nhwc[i])
        axes[0, i].set_title("Orig")
        axes[0, i].axis("off")
        axes[1, i].imshow(recon_nhwc[i])
        axes[1, i].set_title("Recon")
        axes[1, i].axis("off")
        diff = np.clip(np.abs(orig_nhwc[i] - recon_nhwc[i]), 0, 1)
        axes[2, i].imshow(diff)
        axes[2, i].set_title("Abs diff")
        axes[2, i].axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()


def stretch(x_nhwc, target_height=128, target_width=256):
    """
    Stretch images to a fixed rectangular size (target_height x target_width).
    Input:  x_nhwc: (N, H, W, C)
    Output: (N, target_height, target_width, C)
    """
    N, H, W, C = x_nhwc.shape
    out = np.zeros((N, target_height, target_width, C), dtype=x_nhwc.dtype)
    
    for i in range(N):
        # Scale pixel values to 0-255 for PIL
        im = Image.fromarray((x_nhwc[i] * 255).astype(np.uint8))
        # Resize to target height & width
        im_resized = im.resize((target_width, target_height), resample=Image.BILINEAR)
        # Scale back to 0-1
        out[i] = np.asarray(im_resized).astype(np.float32) / 255.0
        
    return out



def save_side_by_side_2x4(orig_nhwc, recon_nhwc, out_path, thresh=0.5, binarize =True):
    """
    Plot 4 originals on the top row and binarized reconstructions on the bottom row.
    Layout: 2 x 4 grid.
    """
    n = min(4, len(orig_nhwc))
    if binarize:
        orig_bin =  binarize_nhwc(orig_nhwc, thresh=thresh)

        recon_bin = binarize_nhwc(recon_nhwc, thresh=thresh)
    else:
        orig_bin = orig_nhwc
        recon_bin = recon_nhwc

    fig, axes = plt.subplots(2, 4, figsize=(12, 6))

    for i in range(4):
        if i < n:
            axes[0, i].imshow(orig_bin[i])
            axes[0, i].set_title(f"Orig {i}")
            axes[1, i].imshow(recon_bin[i])
            axes[1, i].set_title(f"Recon ≥ {thresh}")
        axes[0, i].axis("off")
        axes[1, i].axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()



if __name__ == '__main__':

    # -------------------------
    # Load model (same ViTVAE class as training)
    # -------------------------
    from train_hilab import ViTVAE  # adjust if module path differs

    model = ViTVAE(latent_dim=LATENT_DIM, size_decoder=SIZE_DECODER)
    state = torch.load(MODEL_PATH, map_location="cpu")
    # If you saved {"meta":..., "state":...} for decoder-only, load appropriately.
    # Here we expect full model.state_dict() was saved.
    try:
        model.load_state_dict(state)
    except RuntimeError:
        # maybe state is a dict with keys 'meta'/'state' (decoder-only); try to handle gracefully
        if isinstance(state, dict) and "state" in state:
            st = state["state"]
            model_state = model.state_dict()
            # copy matching shapes
            for k,v in st.items():
                if k in model_state and model_state[k].shape == v.shape:
                    model_state[k] = v
            model.load_state_dict(model_state)
        else:
            raise

    model = model.to(DEVICE)
    model.eval()
    print("Loaded model:", MODEL_PATH)

    # -------------------------
    # Load images
    # -------------------------
    paths, orig_nhwc = load_images_as_nhwc(IMAGE_DIR, resize=RESIZE)
    print("Found images:", len(paths), "example shape (H,W,C):", orig_nhwc.shape[1:])

    # Convert to NCHW using the SAME permutation used for training:
    # training used: (N, H=256, W=128, C=3) -> (N, 3, 128, 256) via permute(0,3,2,1)
    orig_nchw = torch.from_numpy(orig_nhwc).permute(0,3,2,1).contiguous().float().to(DEVICE)
    print("Converted to NCHW for model:", orig_nchw.shape, "  (N,C,H,W)")

    # -------------------------
    # Run model (same path as training: encode -> reparameterize -> decode)
    # -------------------------
    recons = []
    mus = []
    logvars = []
    with torch.no_grad():
        for i in range(0, len(orig_nchw), BATCH_SIZE):
            batch = orig_nchw[i:i+BATCH_SIZE]
            # use the exact forward behavior from training:
            # forward() does encode->reparam->decode, but call encode/reparam/decode explicitly for diagnostics
            mu, logvar = model.encode(batch)
            z = model.reparameterize(mu, logvar)
            recon = model.decode(z)   # (B,3,128,256)
            recons.append(recon.cpu())
            mus.append(mu.cpu())
            logvars.append(logvar.cpu())

    recon_nchw = torch.cat(recons, dim=0)
    mu_all = torch.cat(mus, dim=0)
    logvar_all = torch.cat(logvars, dim=0)
    print("recon_nchw.shape:", recon_nchw.shape)
    print("mu mean/std:", float(mu_all.mean()), float(mu_all.std()))
    print("logvar mean:", float(logvar_all.mean()))

    # ── Latent space health check ──────────────────────────────────────────────
    print("\n── Latent dimension diagnostics ──")
    mu_np      = mu_all.numpy()          # (N, latent_dim)
    logvar_np  = logvar_all.numpy()

    dim_std    = mu_np.std(axis=0)       # std of mu across dataset, per dim
    dim_logvar = logvar_np.mean(axis=0)  # mean posterior logvar per dim

    print(f"{'Dim':>4}  {'mu_std':>8}  {'mean_logvar':>12}  {'status':>12}")
    for i, (s, lv) in enumerate(zip(dim_std, dim_logvar)):
        if s < 0.1:
            status = "DEAD ❌"
        elif s < 0.3:
            status = "weak ⚠️"
        else:
            status = "active ✅"
        print(f"{i:>4}  {s:>8.4f}  {lv:>12.4f}  {status:>12}")

    active = (dim_std >= 0.1).sum()
    print(f"\nActive dims: {active}/{len(dim_std)}")
    print(f"Overall mu std:     {mu_np.std():.4f}  (healthy ~0.5-1.5)")
    print(f"Overall logvar mean:{logvar_np.mean():.4f}  (healthy ~0.0 ± 2)")

    # Convert recon back to ORIGINAL disk NHWC layout:
    # recon_nchw is (N,3,128,256); to get disk layout (N,256,128,3) do permute(0,3,2,1)
    recon_nhwc = recon_nchw.permute(0,3,2,1).numpy()

    # -------------------------
    # Sanity checks & metrics
    # -------------------------
    assert orig_nhwc.shape == recon_nhwc.shape, f"shape mismatch orig={orig_nhwc.shape} recon={recon_nhwc.shape}"

    mse = float(np.mean((orig_nhwc - recon_nhwc) ** 2))
    psnr = psnr_from_mse(mse)
    print(f"Overall MSE: {mse:.6e}  PSNR: {psnr:.2f} dB")

    # Save side-by-side comparison
    # Stretch width to make square for visualization
    orig_sq  = stretch(orig_nhwc, target_height=128, target_width=256)
    recon_sq = stretch(recon_nhwc, target_height=128, target_width=256)

    cmp_path = os.path.join(OUT_DIR, "comparison_fixed.png")
    save_side_by_side_2x4(orig_sq, recon_sq, cmp_path)

    print("Saved comparison to:", cmp_path)

    # Save individual latent vectors and recon images (optional)
    for i, p in enumerate(paths):
        base = os.path.splitext(os.path.basename(p))[0]
        np.savetxt(os.path.join(OUT_DIR, f"{i:02d}_{base}_latent.txt"), mu_all[i].numpy(), fmt="%.6f")
        # save recon as PNG
        im = Image.fromarray((np.clip(recon_nhwc[i], 0, 1) * 255).astype(np.uint8))
        im.save(os.path.join(OUT_DIR, f"{i:02d}_{base}_recon.png"))

    print("Saved latents + per-image reconstructions.")



