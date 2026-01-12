import os
import math
import random
from typing import List, Tuple, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
try:
    from transformers import ViTModel
    _USE_HF = True
except Exception:
    _USE_HF = False
    try:
        import timm
    except Exception:
        timm = None
from typing import Tuple
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

# =========================
#  Utilities
# =========================
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def psnr_from_mse(mse: float, max_val: float = 1.0) -> float:
    if mse <= 1e-12:
        return 99.0
    return 10.0 * math.log10((max_val ** 2) / mse)


def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# =========================
#  Model
# =========================
class ViTVAE(nn.Module):
    """
    ViT encoder (CLS token -> latent) + lightweight ConvTranspose decoder to (3, 128, 256).
    Supports either HuggingFace `transformers` ViTModel or a `timm` ViT fallback.
    """
    def __init__(self, latent_dim=16, vit_name='google/vit-base-patch16-224-in21k'):
        super().__init__()
        self.resize_for_vit = nn.Upsample(size=(224, 224), mode='bilinear', align_corners=False)

        if _USE_HF:
            self.encoder = ViTModel.from_pretrained(vit_name)
            self._enc_type = "hf"
            self.encoder_hidden_size = self.encoder.config.hidden_size
        else:
            if timm is None:
                raise ImportError("Neither 'transformers' nor 'timm' available. Install one to use ViT encoder.")
            # use timm's ViT model as a fallback
            self.encoder = timm.create_model("vit_base_patch16_224", pretrained=True)
            self._enc_type = "timm"
            if hasattr(self.encoder, "head") and hasattr(self.encoder.head, "in_features"):
                hidden = self.encoder.head.in_features
            elif hasattr(self.encoder, "embed_dim"):
                hidden = getattr(self.encoder, "embed_dim")
            else:
                hidden = 768
            self.encoder_hidden_size = hidden

        self.fc_mu = nn.Linear(self.encoder_hidden_size, latent_dim)
        self.fc_logvar = nn.Linear(self.encoder_hidden_size, latent_dim)

        self.decoder_input = nn.Linear(latent_dim, 512 * 4 * 8)   # seed (512, 4, 8)
        self.decoder = nn.Sequential(
            nn.Unflatten(dim=1, unflattened_size=(512, 4, 8)),                 # (512, 4, 8)
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),  # (256, 8, 16)
            nn.BatchNorm2d(256), nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # (128, 16, 32)
            nn.BatchNorm2d(128), nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),   # (64, 32, 64)
            nn.BatchNorm2d(64), nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),    # (32, 64, 128)
            nn.BatchNorm2d(32), nn.ReLU(True),
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),     # (3, 128, 256)
            nn.Sigmoid()
        )

    # ---- freezing helpers ----
    def freeze_all_vit(self):
        for p in self.encoder.parameters():
            p.requires_grad = False

    def unfreeze_last_k_vit_blocks(self, k: int, also_unfreeze_ln_head: bool = True, also_unfreeze_embeddings: bool = False):
        # Freeze everything first
        self.freeze_all_vit()
        if self._enc_type == "hf":
            # Optionally unfreeze embeddings
            if also_unfreeze_embeddings and hasattr(self.encoder, "embeddings"):
                for p in self.encoder.embeddings.parameters():
                    p.requires_grad = True
            # Unfreeze last-k blocks
            if hasattr(self.encoder, "encoder") and hasattr(self.encoder.encoder, "layer"):
                blocks = self.encoder.encoder.layer
                if k > 0:
                    for blk in blocks[-k:]:
                        for p in blk.parameters():
                            p.requires_grad = True
            # Optionally unfreeze final LN / pooler
            if also_unfreeze_ln_head:
                if hasattr(self.encoder, "layernorm"):
                    for p in self.encoder.layernorm.parameters():
                        p.requires_grad = True
                if hasattr(self.encoder, "pooler") and self.encoder.pooler is not None:
                    for p in self.encoder.pooler.parameters():
                        p.requires_grad = True
        else:
            # timm variant
            if hasattr(self.encoder, "blocks") and k > 0:
                for blk in self.encoder.blocks[-k:]:
                    for p in blk.parameters():
                        p.requires_grad = True
            if also_unfreeze_ln_head:
                if hasattr(self.encoder, "norm"):
                    for p in self.encoder.norm.parameters():
                        p.requires_grad = True

    # ---- VAE core ----
    def encode(self, x):
        # x: (B, C, H, W)
        # Pad to square (preserve aspect ratio) before resizing to ViT input size.
        import torch.nn.functional as F
        B, C, H, W = x.shape
        s = max(H, W)
        pad_h = s - H
        pad_w = s - W
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        # F.pad expects (left, right, top, bottom)
        x_padded = F.pad(x, (pad_left, pad_right, pad_top, pad_bottom), value=0.0)
        x224 = self.resize_for_vit(x_padded)  # (B,3,224,224)

        if self._enc_type == "hf":
            enc_out = self.encoder(pixel_values=x224)
            cls = enc_out.last_hidden_state[:, 0, :]
        else:
            feats = self.encoder.forward_features(x224)
            if feats.dim() == 3:
                cls = feats[:, 0, :]
            else:
                cls = feats
        mu = self.fc_mu(cls)
        logvar = self.fc_logvar(cls)
        return mu, logvar

    @staticmethod
    def reparameterize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.decoder_input(z)
        return self.decoder(h)  # (B,3,128,256)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar


# =========================
#  Loss
# =========================
class VAELoss(nn.Module):
    def __init__(self, recon_type="mse", kl_weight=1e-3, recon_weight=1.0, binarization_weight=0.0):
        super().__init__()
        self.recon_type = recon_type
        self.kl_weight = kl_weight
        self.recon_weight = recon_weight
        self.binarization_weight = binarization_weight

    def forward(self, recon, x, mu, logvar):
        if self.recon_type == "mse":
            recon_loss = F.mse_loss(recon, x, reduction="mean")
        elif self.recon_type == "l1":
            recon_loss = F.l1_loss(recon, x, reduction="mean")
        elif self.recon_type == "bce":
            recon_loss = F.binary_cross_entropy(recon, x, reduction="mean")
        else:
            raise ValueError("Unknown recon_type")

        kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        bin_reg = torch.mean(recon * (1 - recon)) if self.binarization_weight > 0 else 0.0
        loss = self.recon_weight * recon_loss + self.kl_weight * kld + self.binarization_weight * bin_reg
        return loss, recon_loss.detach(), kld.detach()


# =========================
#  Data: from your arrays
# =========================
def make_loaders_from_arrays_flexible(
    train_images,          # np.ndarray or torch.Tensor
    test_images,           # np.ndarray or torch.Tensor
    batch_size: int = 16,
    num_workers: int = 4,
    normalize_from_uint8: bool = False,
) -> Tuple[DataLoader, DataLoader]:
    """
    Accepts:
      - NumPy or Torch arrays
      - Either NHWC (N, H, W, C) with C=3 and H=256, W=128
        or   NCHW (N, 3, 128, 256)
    Returns NCHW float tensors in [0,1].
    """

    def to_tensor_nchw(x):
        # Convert to torch tensor (no copy if already tensor)
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        elif not isinstance(x, torch.Tensor):
            raise TypeError("Expected np.ndarray or torch.Tensor")

        # If uint8 0..255 and user didnâ€™t request normalization, warn-raise
        if x.dtype == torch.uint8 and not normalize_from_uint8:
            raise ValueError("Input is uint8; pass normalize_from_uint8=True or normalize beforehand.")

        # Normalize if requested
        if normalize_from_uint8 and x.dtype == torch.uint8:
            x = x.float() / 255.0
        else:
            x = x.float()

        # Detect shape layout
        if x.ndim != 4:
            raise ValueError(f"Expected 4D tensor/array, got shape {tuple(x.shape)}")

        if x.shape[1] == 3 and x.shape[2] == 128 and x.shape[3] == 256:
            # Already NCHW
            pass
        elif x.shape[-1] == 3 and x.shape[1] == 256 and x.shape[2] == 128:
            # NHWC -> NCHW with (H=256, W=128) -> (C=3, H=128, W=256)
            x = x.permute(0, 3, 2, 1)
        else:
            raise ValueError(
                f"Unexpected shape {tuple(x.shape)}. "
                "Supported: (N,3,128,256) or (N,256,128,3)."
            )

        # Ensure range [0,1]
        if x.min() < -1e-6 or x.max() > 1.0 + 1e-6:
            raise ValueError("Values out of [0,1]. Normalize your inputs or set normalize_from_uint8=True.")

        return x.contiguous()

    train_t = to_tensor_nchw(train_images)
    test_t  = to_tensor_nchw(test_images)

    # dummy labels so loop can unpack (imgs, _)
    train_ds = TensorDataset(train_t, torch.zeros(len(train_t)))
    test_ds  = TensorDataset(test_t,  torch.zeros(len(test_t)))

    pin = torch.cuda.is_available()
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=num_workers, pin_memory=pin)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin)
    return train_loader, test_loader
# =========================
#  Train / Eval
# =========================
def make_optimizer(model, lr=1e-4, weight_decay=0.0):
    params = [p for p in model.parameters() if p.requires_grad]
    return optim.Adam(params, lr=lr, weight_decay=weight_decay)


def run_one_epoch(model, loader, optimizer, criterion, device, train=True):
    model.train() if train else model.eval()

    running_loss = 0.0
    running_mse = 0.0
    running_psnr = 0.0
    n_imgs = 0

    pbar = tqdm(loader, leave=False)
    for batch in pbar:
        if isinstance(batch, (list, tuple)) and len(batch) == 2:
            imgs = batch[0]
        else:
            imgs = batch  # fallback

        imgs = imgs.to(device, non_blocking=True)

        if train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train):
            recon, mu, logvar = model(imgs)
            loss, recon_mse, kld = criterion(recon, imgs, mu, logvar)
            if train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

        bsz = imgs.size(0)
        running_loss += loss.item() * bsz
        batch_mse = F.mse_loss(recon, imgs, reduction="mean").item()
        running_mse += batch_mse * bsz
        running_psnr += psnr_from_mse(batch_mse) * bsz
        n_imgs += bsz

        pbar.set_description(f"{'Train' if train else 'Eval'} loss {loss.item():.4f} | MSE {batch_mse:.4f}")

    avg_loss = running_loss / max(1, n_imgs)
    avg_mse = running_mse / max(1, n_imgs)
    avg_psnr = running_psnr / max(1, n_imgs)
    return {"loss": avg_loss, "mse": avg_mse, "psnr": avg_psnr}
def save_grid(imgs, path, nrow=4, rotate90: bool = False):
    """imgs: Tensor (B,3,H,W) in [0,1]"""
    b, c, h, w = imgs.shape
    ncol = nrow
    nrow = int(math.ceil(b / ncol))
    # infer one image size after conversion to original layout to set figure aspect
    if h == 128 and w == 256:
        sample_h, sample_w = w, h  # original layout is (256,128) => (H_orig=W, W_orig=H)
    else:
        sample_h, sample_w = h, w
    # each tile: base size 2 inches height; adjust width to keep aspect ratio
    tile_h_in = 2.0
    tile_w_in = tile_h_in * (sample_w / sample_h)
    fig_w = ncol * tile_w_in
    fig_h = nrow * tile_h_in
    fig, axes = plt.subplots(nrow, ncol, figsize=(max(1.0, fig_w), max(1.0, fig_h)))
    axes = axes.flatten()
    for i in range(len(axes)):
        axes[i].axis("off")
        if i < b:
            # Convert internal tensor (C,H,W) back to original saved layout (H_orig= W, W_orig= H, C)
            # Our dataset originally produced NHWC images of shape (256,128,3) which were
            # converted to NCHW (3,128,256). To show the original orientation we need
            # to permute (2,1,0) -> (W,H,C) == (256,128,3).
            if h == 128 and w == 256:
                img = imgs[i].permute(2, 1, 0).cpu().numpy()
            else:
                img = imgs[i].permute(1, 2, 0).cpu().numpy()
            # Optionally rotate 90 degrees counter-clockwise for visualization
            if rotate90:
                import numpy as _np
                img = _np.rot90(img, k=1)
            axes[i].imshow(img, aspect='equal')
    # Save without cropping the tiles
    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()

# =========================
#  Sweep
# =========================
def sweep_thaw_depths_with_loaders(
    train_loader: DataLoader,
    val_loader: DataLoader,
    thaw_depths: List[int],
    epochs_per_setting: int = 5,
    latent_dim: int = 16,
    recon_type: str = "mse",
    kl_weight: float = 1e-3,
    lr: float = 1e-4,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    save_grid_examples: bool = True,
):
    set_seed(123)

    results = []
    all_curves: Dict[int, List[float]] = {}

    for k in thaw_depths:
        print("\n" + "=" * 80)
        print(f"Setting: Unfreeze last {k} ViT blocks")
        print("=" * 80)

        model = ViTVAE(latent_dim=latent_dim)
        model.unfreeze_last_k_vit_blocks(k=k, also_unfreeze_ln_head=True, also_unfreeze_embeddings=False)
        model = model.to(device)

        print(f"Trainable parameters: {count_trainable_params(model):,}")

        criterion = VAELoss(recon_type=recon_type, kl_weight=kl_weight)
        optimizer = make_optimizer(model, lr=lr)

        val_mse_curve = []

        for epoch in range(1, epochs_per_setting + 1):
            print(f"\n[Thaw={k}] Epoch {epoch}/{epochs_per_setting}")
            train_metrics = run_one_epoch(model, train_loader, optimizer, criterion, device, train=True)
            val_metrics   = run_one_epoch(model, val_loader,   optimizer, criterion, device, train=False)

            print(f"Train: loss={train_metrics['loss']:.4f} | MSE={train_metrics['mse']:.6f} | PSNR={train_metrics['psnr']:.2f} dB")
            print(f"Val  : loss={val_metrics['loss']:.4f} | MSE={val_metrics['mse']:.6f} | PSNR={val_metrics['psnr']:.2f} dB")

            val_mse_curve.append(val_metrics["mse"])

            if save_grid_examples and epoch == epochs_per_setting:
                model.eval()
                with torch.no_grad():
                    imgs, _ = next(iter(val_loader))
                    imgs = imgs[:8].to(device)
                    recon, _, _ = model(imgs)
                    os.makedirs("examples", exist_ok=True)
                    save_grid(imgs, f"examples/input_thaw{k}.png")
                    save_grid(recon, f"examples/recon_thaw{k}.png")
                    print(f"Saved examples: examples/input_thaw{k}.png, examples/recon_thaw{k}.png")

            # Save model parameters for this thaw setting
            os.makedirs("models", exist_ok=True)
            full_path = os.path.join("models", f"vitvae_thaw{k}.pt")
            try:
                torch.save(model.state_dict(), full_path)
                # Also save decoder-only weights for quick loading in downstream BO
                decoder_path = os.path.join("models", f"vitvae_decoder_thaw{k}.pt")
                dec_state = {kname: v for kname, v in model.state_dict().items() if kname.startswith("decoder") or kname.startswith("decoder_input")}
                # include metadata
                meta = {"latent_dim": latent_dim, "recon_size": (128, 256)}
                torch.save({"meta": meta, "state": dec_state}, decoder_path)
                print(f"Saved model state: {full_path} and decoder: {decoder_path}")
            except Exception as e:
                print(f"[WARN] Failed to save model for thaw={k}: {e}")

        all_curves[k] = val_mse_curve
        agg = sum(val_mse_curve[-2:]) / 2.0 if len(val_mse_curve) >= 2 else val_mse_curve[-1]
        results.append({"thaw_blocks": k, "val_mse": agg})

    # Plot standard curve
    xs = [r["thaw_blocks"] for r in results]
    ys = [r["val_mse"] for r in results]
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    xs = [xs[i] for i in order]; ys = [ys[i] for i in order]

    os.makedirs("figures", exist_ok=True)
    plt.figure(figsize=(7,5))
    plt.plot(xs, ys, marker="o")
    plt.xlabel("Number of unfrozen ViT encoder blocks (last-k)")
    plt.ylabel("Validation reconstruction MSE (lower is better)")
    plt.title("Effect of partial ViT fine-tuning on reconstruction quality (your dataset)")
    plt.grid(True); plt.tight_layout()
    plt.savefig("figures/standard_curve_vit_thaw_vs_mse.png", dpi=180)
    print("\nSaved curve to figures/standard_curve_vit_thaw_vs_mse.png")

    # Per-setting curves
    plt.figure(figsize=(7,5))
    for k, curve in sorted(all_curves.items(), key=lambda kv: kv[0]):
        plt.plot(range(1, len(curve)+1), curve, marker=".", label=f"thaw={k}")
    plt.xlabel("Epoch"); plt.ylabel("Val MSE")
    plt.title("Per-setting validation curves (your dataset)")
    plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig("figures/val_curves_by_thaw.png", dpi=180)
    print("Saved per-setting curves to figures/val_curves_by_thaw.png")

    print("\nSummary (smaller MSE is better):")
    for r in sorted(results, key=lambda d: d["thaw_blocks"]):
        print(f"  thaw={r['thaw_blocks']:>2d} -> Val MSE ~ {r['val_mse']:.6f}")

    return results, all_curves


# =========================
#  Example main for your arrays
# =========================
if __name__ == "__main__":
    """
    Replace the two lines below with your actual arrays, or import them.
    Expected shapes:
      train_images: (N, 256, 128, 3)  | test_images: (M, 256, 128, 3)
      values in [0,1] or set normalize_from_uint8=True
    """
    # --- Configuration: edit these paths to point to your image folders ---
    CONFIG = {
        # list of directories containing PNG/JPEG images for training
        'train_dirs':['outputs/augmented/designs/mbb_beam_384x64_0.4-20260111-164330'],
        # list of directories for validation / holdout
        'val_dirs': ['outputs/designs/mbb_beam_384x64_0.4-20260111-164330/images'],
        # desired in-memory image size: (width, height) for PIL resize
        'resize': (128, 256),
        # whether to treat input files as uint8 and normalize automatically
        'normalize_from_uint8': True,
        'batch_size': 16,
        'num_workers': 4,
    }


    from glob import glob
    from PIL import Image


    def load_images_from_dirs(dirs, resize=(128, 256), max_images=None):
        """Load images from multiple directories, return NHWC uint8 float32 in [0,1].

        Args:
          dirs: list of directory paths (strings or Path) to search recursively.
          resize: (width, height) tuple for PIL.Image.resize.
        Returns:
          NumPy array shaped (N, H, W, C) with dtype float32 and values in [0,1].
        """
        paths = []
        for d in dirs:
            d = str(d)
            paths += glob(os.path.join(d, '**', '*.png'), recursive=True)
            paths += glob(os.path.join(d, '**', '*.jpg'), recursive=True)
            paths += glob(os.path.join(d, '**', '*.jpeg'), recursive=True)

        paths = sorted(set(paths))
        if max_images is not None:
            paths = paths[:max_images]

        imgs = []
        for p in paths:
            try:
                im = Image.open(p).convert('RGB')
                im = im.resize(resize, resample=Image.BILINEAR)
                arr = np.asarray(im).astype(np.float32) / 255.0
                # expected output NHWC: (H, W, C) -> (256,128,3)
                imgs.append(arr)
            except Exception as e:
                print(f"Skipping {p}: {e}")

        if not imgs:
            raise RuntimeError(f'No images found in {dirs}')
        return np.stack(imgs, axis=0)


    # Load training and validation images from configured folders
    train_images = load_images_from_dirs(CONFIG['train_dirs'], resize=CONFIG['resize'])
    # If val_dirs doesn't exist or is empty, reuse a small split from train
    try:
        val_images = load_images_from_dirs(CONFIG['val_dirs'], resize=CONFIG['resize'])
    except RuntimeError:
        # fallback: take last 5% of train as validation
        n = len(train_images)
        split = max(1, int(n * 0.05))
        val_images = train_images[-split:]
        train_images = train_images[:-split]

    print("Train images shape:", train_images.shape)
    print("Val images shape  :", val_images.shape)

    train_loader, val_loader = make_loaders_from_arrays_flexible(
        train_images, val_images,
        batch_size=CONFIG['batch_size'],
        num_workers=CONFIG['num_workers'],
        normalize_from_uint8=CONFIG['normalize_from_uint8'],
    )

    thaw_settings = [2]  # 2 = your current choice
    results, curves = sweep_thaw_depths_with_loaders(
        train_loader, val_loader,
        thaw_depths=thaw_settings,
        epochs_per_setting=20,
        latent_dim=16,
        recon_type="bce",
        kl_weight=1e-2,
        lr=1e-4,
    )
