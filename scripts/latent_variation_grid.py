#!/usr/bin/env python3
"""
Generate a sensitivity grid for a decoder's latent space.

Usage examples:
  python scripts/latent_variation_grid.py --decoder models/vitvae_decoder_thaw2_latent8.pt --outdir outputs/latent_grid
  python scripts/latent_variation_grid.py --decoder models/vitvae_decoder_thaw2_latent8.pt \
      --base 0.5,0.0,-1.0,0,0,0,0,0 --schedule -2,-1,1,2 --mode replace

This will save `base.png` and `grid.png` in the output folder.
"""
import os
import argparse
from glob import glob

import numpy as np
import torch
from PIL import Image

try:
    from scripts.train_hilab import ViTVAE
except Exception:
    from train_hilab import ViTVAE


def stretch_to_rect(img, target_height=128, target_width=256):
    """
    Stretch an image to a fixed rectangular size.
    Args:
        img: HxWxC float in [0,1]
        target_height: desired height
        target_width: desired width
    Returns:
        HxWxC float in [0,1]
    """
    im = Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8))
    im_resized = im.resize((target_width, target_height), resample=Image.BILINEAR)
    return np.asarray(im_resized).astype(np.float32) / 255.0


def load_decoder(path):
    ck = torch.load(path, map_location='cpu')
    meta = ck.get('meta', {})
    state = ck.get('state', ck)
    size_decoder = meta.get('size_decoder', 'xlarge')
    latent_dim = int(meta.get('latent_dim', 16))
    model = ViTVAE(latent_dim=latent_dim, size_decoder=size_decoder)
    model_state = model.state_dict()
    for k, v in state.items():
        if k in model_state and model_state[k].shape == v.shape:
            model_state[k] = v
    model.load_state_dict(model_state)
    model.eval()
    return model, latent_dim


def decode_to_rgb(decoder, z: np.ndarray):
    """
    z: 1D numpy array shape (latent_dim,)
    returns: HxWx3 float in [0,1]
    """
    zt = torch.from_numpy(z.astype(np.float32)).unsqueeze(0)
    with torch.no_grad():
        recon = decoder.decode(zt)
    recon = recon[0].cpu().numpy()  # (3, H_dec, W_dec)
    rgb = np.clip(recon.transpose(2, 1, 0), 0, 1)  # (H_dec, W_dec, 3)
    return rgb


def binarize_rgb(rgb: np.ndarray, threshold: float = 0.5):
    gray = rgb.mean(axis=2)
    bin_mask = (gray >= threshold).astype(np.float32)
    return np.stack([bin_mask] * 3, axis=2)


def save_rgb(img_arr, path):
    im = Image.fromarray((np.clip(img_arr, 0, 1) * 255).astype(np.uint8))
    im.save(path)


def prepare_base_rgb(decoder, base_latent, binarize=True, threshold=0.5):
    recon_raw = decode_to_rgb(decoder, base_latent)
    recon_stretched = stretch_to_rect(recon_raw, target_height=128, target_width=256)
    recon_for_mse = recon_stretched.copy()
    if binarize:
        base_rgb = binarize_rgb(recon_stretched, threshold=threshold)
    else:
        base_rgb = recon_stretched
    return base_rgb, recon_for_mse


def build_variations(decoder, base_latent, schedule_vals, mode='add', binarize=True, threshold=0.5):
    latent_dim = base_latent.shape[0]
    variations = []
    for i in range(latent_dim):
        row = []
        for v in schedule_vals:
            z = base_latent.copy()
            if mode == 'add':
                z[i] += v
            else:
                z[i] = v
            rgb_raw = decode_to_rgb(decoder, z)
            rgb_stretched = stretch_to_rect(rgb_raw, target_height=128, target_width=256)
            if binarize:
                rgb = binarize_rgb(rgb_stretched, threshold=threshold)
            else:
                rgb = rgb_stretched
            row.append(rgb)
        variations.append(row)
    return variations


def make_grid(base_rgb, variations, outpath, row_spacing=10, col_spacing=10):
    rows = len(variations)
    cols = len(variations[0]) if rows > 0 else 0
    H, W, C = base_rgb.shape
    grid_w = cols * W + (cols - 1) * col_spacing
    grid_h = rows * H + (rows - 1) * row_spacing
    grid = Image.new('RGB', (grid_w, grid_h), color=(255, 255, 255))
    for r in range(rows):
        for c in range(cols):
            arr = variations[r][c]
            im = Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8))
            paste_x = c * (W + col_spacing)
            paste_y = r * (H + row_spacing)
            grid.paste(im, (paste_x, paste_y))
    grid.save(outpath)


def parse_latent_list(s: str):
    parts = [p.strip() for p in s.split(',') if p.strip()]
    return np.array([float(x) for x in parts], dtype=np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--decoder', type=str, default=None)
    parser.add_argument('--base', type=str, default=None)
    parser.add_argument('--base_img', type=str, default=None)
    parser.add_argument('--full_model', type=str, default=None)
    parser.add_argument('--schedule', nargs='+', default=['-3, -2, -1.0, -0.5, 1.0, 2.0, 3.0'])
    parser.add_argument('--mode', choices=['add', 'replace'], default='replace')
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--binarize_threshold', type=float, default=0.5)
    parser.add_argument('--no_binarize', action='store_true')
    parser.add_argument('--outdir', type=str, default='outputs/latent_grid')
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    decoder_path = args.decoder
    print('Loading decoder:', decoder_path)
    decoder, latent_dim = load_decoder(decoder_path)
    print('Detected latent_dim =', latent_dim)

    # Prepare base latent
    base = None
    if args.base is not None:
        base_vec = parse_latent_list(args.base)
        if base_vec.shape[0] != latent_dim:
            raise ValueError(f'Provided base length {base_vec.shape[0]} does not match latent_dim={latent_dim}')
        base = base_vec.astype(np.float32)

    # TODO: handle --base_img encoding if needed (kept original logic)
    if base is None:
        rng = np.random.RandomState(args.seed)
        base = rng.randn(latent_dim).astype(np.float32)
        print('Sampled random base latent (seed=%d)' % args.seed)

    # Parse schedule
    schedule_tokens = []
    for tok in args.schedule:
        schedule_tokens += [t for t in str(tok).split(',') if t.strip()]
    schedule_vals = [float(s) for s in schedule_tokens]

    # Base image
    base_rgb, _ = prepare_base_rgb(decoder, base, binarize=not args.no_binarize, threshold=args.binarize_threshold)
    save_rgb(base_rgb, os.path.join(args.outdir, 'base.png'))
    print('Saved base image to', os.path.join(args.outdir, 'base.png'))

    # Build variations and grid
    variations = build_variations(
        decoder, base, schedule_vals, mode=args.mode,
        binarize=not args.no_binarize, threshold=args.binarize_threshold
    )
    grid_path = os.path.join(args.outdir, 'grid.png')
    make_grid(base_rgb, variations, grid_path)
    print('Saved grid to', grid_path)


if __name__ == '__main__':
    main()

# --base "-0.205460,-1.144199, -1.789775, -0.902603, 1.777870, 0.828746, 0.444054, 1.197009"\
# --base "0.305828, 1.073675, 0.639515, 0.339767, 2.682443, 0.515682, 1.378971, -0.196818"

# NOT BAD!!!

#--base "-0.142843, -0.030754, -0.093442, -0.596072, 0.198657, 1.288095, -0.307844, 1.894389, 0.697677, 1.200793, -1.004479, 0.319866, -1.811832, 0.530297, -1.017856, 1.801798"

# --base "0.378275, 0.005934, -0.821723, -1.323568, 1.853147, 0.018200, 0.325669, 1.070722, 0.039563, -0.877092, -1.023597, 0.792667, 1.459755, 1.191377, -0.159492, 1.013047"