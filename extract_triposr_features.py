"""
Extract pre-MLP triplane features from TripoSR.

Loads TripoSR, freezes all weights, runs the encoder on an input image,
then extracts the raw 120-dim triplane features that would normally be
fed into the final NeRF MLP.

Usage:
    python extract_triposr_features.py --image path/to/image.png
    python extract_triposr_features.py --image path/to/image.png --resolution 64 --output features.pt
"""

import argparse
import sys
from pathlib import Path
from unittest.mock import MagicMock

# Mock deps that are unused here but cause import errors
sys.modules["trimesh"] = MagicMock()
sys.modules["torchmcubes"] = MagicMock()
sys.modules["rembg"] = MagicMock()

sys.path.insert(0, str(Path(__file__).parent / "TripoSR"))

import torch
import torch.nn.functional as F
from einops import rearrange, reduce
from PIL import Image

from tsr.system import TSR
from tsr.utils import scale_tensor


def query_triplane_features(
    renderer,
    positions: torch.Tensor,
    triplane: torch.Tensor,
) -> torch.Tensor:
    """Sample the triplane at 3D positions and return the raw concatenated
    features (120-dim with the default config) *without* passing through
    the NeRF MLP.

    Args:
        renderer: The TriplaneNeRFRenderer (used for config like radius).
        positions: (*, 3) world-space coordinates.
        triplane: (3, C, H, W) scene code for one sample.

    Returns:
        features: (*, 3*C) raw triplane features per query point.
    """
    input_shape = positions.shape[:-1]
    positions_flat = positions.reshape(-1, 3)

    positions_norm = scale_tensor(
        positions_flat,
        (-renderer.cfg.radius, renderer.cfg.radius),
        (-1, 1),
    )

    def _sample_chunk(x):
        indices2D = torch.stack(
            (x[..., [0, 1]], x[..., [0, 2]], x[..., [1, 2]]),
            dim=-3,
        )
        out = F.grid_sample(
            rearrange(triplane, "Np Cp Hp Wp -> Np Cp Hp Wp", Np=3),
            rearrange(indices2D, "Np N Nd -> Np () N Nd", Np=3),
            align_corners=False,
            mode="bilinear",
        )
        if renderer.cfg.feature_reduction == "concat":
            return rearrange(out, "Np Cp () N -> N (Np Cp)", Np=3)
        else:
            return reduce(out, "Np Cp () N -> N Cp", Np=3, reduction="mean")

    chunk_size = renderer.chunk_size
    if chunk_size > 0:
        chunks = [
            _sample_chunk(positions_norm[i : i + chunk_size])
            for i in range(0, positions_norm.shape[0], chunk_size)
        ]
        features = torch.cat(chunks, dim=0)
    else:
        features = _sample_chunk(positions_norm)

    return features.reshape(*input_shape, -1)


def main():
    parser = argparse.ArgumentParser(description="Extract TripoSR pre-MLP triplane features")
    parser.add_argument("--image", default="examples/chair.png", help="Path to input image (default: examples/chair.png)")
    parser.add_argument(
        "--model",
        default="stabilityai/TripoSR",
        help="HuggingFace model ID or local path (default: stabilityai/TripoSR)",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=64,
        help="Grid resolution per axis for feature extraction (default: 64)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=8192,
        help="Chunk size for triplane sampling (0 = no chunking, default: 8192)",
    )
    parser.add_argument(
        "--output",
        default="features.pt",
        help="Output path for saved features tensor (default: features.pt)",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device to use (default: cuda:0 if available, else cpu)",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify features match the original pipeline by passing them through the MLP",
    )
    args = parser.parse_args()

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    print("Loading TripoSR...")
    model = TSR.from_pretrained(args.model, config_name="config.yaml", weight_name="model.ckpt")
    model.renderer.set_chunk_size(args.chunk_size)
    model.to(device)
    model.eval()

    for param in model.parameters():
        param.requires_grad_(False)

    print(f"Model loaded, all weights frozen.")
    print(f"  NeRF MLP input dim : {model.decoder.cfg.in_channels}")
    print(f"  NeRF MLP hidden    : {model.decoder.cfg.n_neurons} x {model.decoder.cfg.n_hidden_layers} layers")

    # Encode image -> scene codes
    image = Image.open(args.image).convert("RGB")
    print(f"Encoding image: {args.image}")
    with torch.no_grad():
        scene_codes = model([image], device=device)
    print(f"  scene_codes shape: {scene_codes.shape}")  # (1, 3, 40, 64, 64)

    # Build query grid
    resolution = args.resolution
    radius = model.renderer.cfg.radius
    grid_1d = torch.linspace(-radius, radius, resolution, device=device)
    grid_x, grid_y, grid_z = torch.meshgrid(grid_1d, grid_1d, grid_1d, indexing="ij")
    grid_points = torch.stack([grid_x, grid_y, grid_z], dim=-1)  # (R, R, R, 3)
    print(f"Querying {resolution}^3 = {resolution**3} grid points in [-{radius:.3f}, {radius:.3f}]^3")

    # Extract features
    with torch.no_grad():
        features = query_triplane_features(model.renderer, grid_points, scene_codes[0])

    print(f"  features shape : {features.shape}")
    print(f"  min={features.min():.4f}  max={features.max():.4f}  mean={features.mean():.4f}")

    # Optional verification
    if args.verify:
        with torch.no_grad():
            reference = model.renderer.query_triplane(model.decoder, grid_points, scene_codes[0])
            mlp_out = model.decoder(features)
        density_ok = torch.allclose(mlp_out["density"], reference["density"], atol=1e-5)
        color_ok = torch.allclose(mlp_out["features"], reference["features"], atol=1e-5)
        print(f"  Density matches original pipeline : {density_ok}")
        print(f"  Color features match original     : {color_ok}")

    # Save
    torch.save(features.cpu(), args.output)
    print(f"Saved features to: {args.output}")


if __name__ == "__main__":
    main()
