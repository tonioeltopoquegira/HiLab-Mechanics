#!/usr/bin/env python3
"""
Plot the best design from a BO results JSON produced by `scripts/bo_decoder_opt.py`.

Usage: python scripts/plot_bo_result.py [path/to/bo_result.json]
If no path is provided, the script picks the newest file in `bo_results/`.
"""
import os
import sys
import json
from glob import glob
from datetime import datetime

import numpy as np
import torch
from PIL import Image

try:
    from scripts.train_hilab import ViTVAE
except Exception:
    from train_hilab import ViTVAE

from neural_structural_optimization import topo_physics


def find_latest_bo_result(folder='bo_results'):
    files = sorted(glob(os.path.join(folder, 'bo_result_*.json')))
    if not files:
        raise FileNotFoundError(f'No bo_result_*.json found in {folder}')
    return files[-1]


def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)


def extract_best_params(data):
    # Try common layouts
    if 'best' in data:
        best = data['best']
        # best might be [compliance, params] or {'0':..}
        if isinstance(best, list) and len(best) >= 2 and isinstance(best[1], dict):
            return best[1]
        if isinstance(best, dict) and 'parameters' in best:
            return best['parameters']
    # fallback: look for 'trace' and take best entry
    trace = data.get('trace', [])
    if trace:
        # find minimal compliance
        best_entry = min(trace, key=lambda e: float(e.get('compliance', float('inf'))))
        return best_entry.get('params') or best_entry.get('parameters') or best_entry.get('params')
    raise RuntimeError('Could not extract best parameters from BO result JSON')


def params_to_latent(params_dict, latent_dim=16):
    z = np.zeros(latent_dim, dtype=np.float32)
    for i in range(latent_dim):
        key = f'x{i}'
        if key in params_dict:
            z[i] = float(params_dict[key])
        elif str(i) in params_dict:
            z[i] = float(params_dict[str(i)])
        else:
            raise KeyError(f'Missing latent parameter {key} in params dict')
    return z


def load_decoder(path, latent_dim=16):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    ck = torch.load(path, map_location='cpu')
    state = ck.get('state', ck)
    model = ViTVAE(latent_dim=latent_dim)
    model_state = model.state_dict()
    for k, v in state.items():
        if k in model_state and model_state[k].shape == v.shape:
            model_state[k] = v
    model.load_state_dict(model_state)
    model.eval()
    return model


def latent_to_images(decoder, z: np.ndarray):
    zt = torch.from_numpy(z.astype(np.float32)).unsqueeze(0)
    with torch.no_grad():
        recon = decoder.decode(zt)
    recon = recon[0].cpu().numpy()  # (3,H,W)
    rgb = np.clip(recon.transpose(1, 2, 0), 0, 1)
    gray = rgb.mean(axis=2)
    return rgb, gray


def resize_for_physics(gray, nelx, nely):
    pil = Image.fromarray((gray * 255).astype(np.uint8))
    pil = pil.resize((nelx, nely), resample=Image.BILINEAR)
    arr = np.asarray(pil).astype(np.float32) / 255.0
    return arr


def main(path=None):
    if path is None:
        path = find_latest_bo_result()
    data = load_json(path)
    params = extract_best_params(data)
    if params is None:
        raise RuntimeError('No parameters found in BO result')

    latent_dim = int(data.get('latent_dim', 16))
    z = params_to_latent(params, latent_dim=latent_dim)

    # decoder checkpoint
    decoder_ckpt = data.get('decoder_ckpt') or 'models/vitvae_decoder_thaw2.pt'
    if not os.path.exists(decoder_ckpt):
        print(f"Warning: decoder checkpoint {decoder_ckpt} not found. Using default models/... if available.")
    decoder = load_decoder(decoder_ckpt, latent_dim=latent_dim)

    rgb, gray = latent_to_images(decoder, z)

    # physics grid
    args = topo_physics.default_args()
    nely = int(args['nely'])
    nelx = int(args['nelx'])
    phys = resize_for_physics(gray, nelx, nely)

    outdir = os.path.dirname(path)
    base = os.path.splitext(os.path.basename(path))[0]
    rgb_path = os.path.join(outdir, base + '_best_recon_rgb.png')
    gray_path = os.path.join(outdir, base + '_best_recon_gray.png')
    phys_path = os.path.join(outdir, base + '_best_physics_density.png')

    Image.fromarray((rgb * 255).astype(np.uint8)).save(rgb_path)
    Image.fromarray((gray * 255).astype(np.uint8)).save(gray_path)
    Image.fromarray((phys * 255).astype(np.uint8)).save(phys_path)

    print('Saved:', rgb_path)
    print('Saved:', gray_path)
    print('Saved:', phys_path)


if __name__ == '__main__':
    arg = sys.argv[1] if len(sys.argv) > 1 else None
    main(arg)
