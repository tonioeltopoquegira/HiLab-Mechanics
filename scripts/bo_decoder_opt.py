#!/usr/bin/env python3
"""
Bayesian optimization over a pretrained ViTVAE decoder's 16-d latent input to minimize
topology-compliance evaluated by the repo's autograd-based physics (`topo_physics.objective`).

Usage: edit CONFIG below or run as script and point to a decoder checkpoint.
"""

import os
import time
import json
from datetime import datetime
from typing import Dict

import numpy as np
import torch
from PIL import Image

# Robust Ax imports
AX_FALLBACK = False
try:
    from ax.service.client import AxClient
    print("Using ax.service.client.AxClient")
except Exception:
    try:
        from ax.service.ax_client import AxClient
        print("Using ax.service.ax_client.AxClient (fallback)")
    except Exception:
        raise ImportError("Could not import AxClient. Please install ax-platform.")

# Import ViTVAE from the training script (it defines the model architecture)
try:
    # prefer package-style import when running from repo root
    from scripts.train_hilab import ViTVAE
except Exception:
    try:
        from train_hilab import ViTVAE
    except Exception as e:
        raise ImportError("Could not import ViTVAE from train_hilab.py. Make sure you run this script from the repo root and that scripts/train_hilab.py is available.")
# Physics
from neural_structural_optimization import topo_physics


CONFIG = {
    "decoder_checkpoint": "models/vitvae_decoder_thaw2.pt",
    "latent_dim": 16,
    "bounds": (-10.0, 10.0),
    "max_trials": 100,
    "sobol_steps": 5,
    "save_dir": "bo_results",
}


def load_decoder_from_checkpoint(path: str, latent_dim: int):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    ck = torch.load(path, map_location="cpu")
    meta = ck.get("meta", {})
    state = ck.get("state", ck)

    model = ViTVAE(latent_dim=latent_dim)
    # load only matching keys
    model_state = model.state_dict()
    for k, v in state.items():
        if k in model_state and model_state[k].shape == v.shape:
            model_state[k] = v
    model.load_state_dict(model_state)
    model.eval()
    return model


def latent_to_design(decoder_model: ViTVAE, z: np.ndarray):
    """z: (latent_dim,) in numpy -> design array (nely, nelx) as float in [0,1]"""
    zt = torch.from_numpy(z.astype(np.float32)).unsqueeze(0)  # (1, latent)
    with torch.no_grad():
        # Feed latent vector directly to the decoder (avoid calling forward which expects images)
        recon = decoder_model.decode(zt)
    # recon shape: (1,3,128,256) in [0,1]
    img = recon[0].cpu().numpy()  # (3,H,W)
    gray = img.mean(axis=0)  # (H,W)
    # convert to PIL and resize to physics grid
    args = topo_physics.default_args()
    nely = int(args['nely'])
    nelx = int(args['nelx'])
    pil = Image.fromarray((np.clip(gray, 0, 1) * 255).astype(np.uint8))
    pil = pil.resize((nelx, nely), resample=Image.BILINEAR)
    arr = np.asarray(pil).astype(np.float32) / 255.0
    # arr shape (nely, nelx)
    return arr


def params_to_latent(params: Dict, latent_dim: int):
    z = np.zeros(latent_dim, dtype=np.float32)
    for i in range(latent_dim):
        key = f"x{i}"
        if key in params:
            z[i] = float(params[key])
        elif str(i) in params:
            z[i] = float(params[str(i)])
        else:
            raise KeyError(f"Missing latent parameter {key} in params dict")
    return z


def latent_to_images(decoder_model: ViTVAE, z: np.ndarray):
    """Return (rgb, gray) where rgb is HxWx3 in [0,1] and gray is HxW in [0,1]."""
    zt = torch.from_numpy(z.astype(np.float32)).unsqueeze(0)
    with torch.no_grad():
        recon = decoder_model.decode(zt)
    recon = recon[0].cpu().numpy()  # (3,H,W)
    rgb = np.clip(recon.transpose(1, 2, 0), 0, 1)
    gray = rgb.mean(axis=2)
    return rgb, gray


def resize_for_physics(gray: np.ndarray, nelx: int, nely: int):
    pil = Image.fromarray((gray * 255).astype(np.uint8))
    pil = pil.resize((nelx, nely), resample=Image.BILINEAR)
    arr = np.asarray(pil).astype(np.float32) / 255.0
    return arr


def evaluate_compliance(design_arr: np.ndarray, args=None):
    if args is None:
        args = topo_physics.default_args()
    ke = topo_physics.get_stiffness_matrix(args['young'], args['poisson'])
    # objective expects array shape (nely, nelx)
    c = float(topo_physics.objective(design_arr, ke, args))
    return c


def run_bo():
    os.makedirs(CONFIG['save_dir'], exist_ok=True)
    start_time = time.time()

    # Load decoder model
    decoder = load_decoder_from_checkpoint(CONFIG['decoder_checkpoint'], CONFIG['latent_dim'])

    # Create Ax client
    client = AxClient()
    # define 16 continuous parameters
    params = [{
        "name": f"x{i}",
        "type": "range",
        "bounds": [CONFIG['bounds'][0], CONFIG['bounds'][1]],
    } for i in range(CONFIG['latent_dim'])]

    # Create experiment with robust handling for different Ax versions.
    from inspect import signature

    def _safe_create_experiment(ax_client, name, parameters, objective_name="compliance"):
        sig = signature(ax_client.create_experiment)
        pnames = list(sig.parameters.keys())
        # Try older/newer signatures
        if 'objective_name' in pnames:
            # signature supports objective_name, minimize
            try:
                ax_client.create_experiment(name=name, parameters=parameters, objective_name=objective_name, minimize=True)
                return
            except Exception:
                pass
        if 'objectives' in pnames:
            # newer signature expecting objectives dict
            try:
                # try to import ObjectiveProperties if available
                try:
                    from ax.service.utils.instantiation import ObjectiveProperties
                    objectives = {objective_name: ObjectiveProperties(minimize=True)}
                except Exception:
                    objectives = {objective_name: {"minimize": True}}
                ax_client.create_experiment(name=name, parameters=parameters, objectives=objectives)
                return
            except Exception:
                pass
        # fallback minimal attempt
        try:
            ax_client.create_experiment(name=name, parameters=parameters)
            return
        except Exception as ex:
            raise RuntimeError(f"Failed to create experiment on AxClient: {ex}")

    _safe_create_experiment(client, name="decoder_latent_bo", parameters=params, objective_name="compliance")

    best = None
    trace = []

    for t in range(CONFIG['max_trials']):
        parameters, trial_index = client.get_next_trial()
        # parameters: dict of name->value
        z = np.array([float(parameters[f"x{i}"]) for i in range(CONFIG['latent_dim'])], dtype=np.float32)
        t0 = time.time()
        design = latent_to_design(decoder, z)
        c = evaluate_compliance(design)
        elapsed = time.time() - t0

        client.complete_trial(trial_index=trial_index, raw_data={"compliance": (c, 0.0)})
        trace.append({"trial": t, "params": parameters, "compliance": c, "time_s": elapsed})

        if best is None or c < best[0]:
            best = (c, parameters)

        print(f"Trial {t+1}/{CONFIG['max_trials']}: compliance={c:.6f} time={elapsed:.2f}s best={best[0]:.6f}")

        # No manual manipulation of the client's internal generation strategy.
        # Ax will transition from Sobol to BoTorch automatically based on its
        # configured strategy. Avoid clearing internal state.

    total_time = time.time() - start_time

    # Save results
    out = {
        "best": best,
        "trace": trace,
        "total_time_s": total_time,
        "timestamp": datetime.now().isoformat(),
        "decoder_ckpt": CONFIG['decoder_checkpoint'],
        "best_params": best[1] if best is not None else None,
    }
    out_fname = os.path.join(CONFIG['save_dir'], f"bo_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(out_fname, 'w') as fh:
        json.dump(out, fh, indent=2)

    # Save images for the best found latent (reconstruction + physics density)
    try:
        if best is not None and best[1] is not None:
            best_params = best[1]
            try:
                z = params_to_latent(best_params, CONFIG['latent_dim'])
                rgb, gray = latent_to_images(decoder, z)
                args = topo_physics.default_args()
                nely = int(args['nely'])
                nelx = int(args['nelx'])
                phys = resize_for_physics(gray, nelx, nely)

                base = os.path.splitext(os.path.basename(out_fname))[0]
                rgb_path = os.path.join(CONFIG['save_dir'], base + '_best_recon_rgb.png')
                gray_path = os.path.join(CONFIG['save_dir'], base + '_best_recon_gray.png')
                phys_path = os.path.join(CONFIG['save_dir'], base + '_best_physics_density.png')

                Image.fromarray((rgb * 255).astype(np.uint8)).save(rgb_path)
                Image.fromarray((gray * 255).astype(np.uint8)).save(gray_path)
                Image.fromarray((phys * 255).astype(np.uint8)).save(phys_path)

                print('Saved best reconstruction RGB:', rgb_path)
                print('Saved best reconstruction gray:', gray_path)
                print('Saved best physics density:', phys_path)
            except Exception as e:
                print('Failed to save best-images:', e)
    except Exception:
        pass

    print("BO finished. Best compliance:", best)
    print("Saved results to", out_fname)
    return out


if __name__ == '__main__':
    run_bo()
