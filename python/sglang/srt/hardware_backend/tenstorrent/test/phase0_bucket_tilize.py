"""Phase 0.1 — bucket Tracy Tilize ops by weight shape.

Usage:
    python3 phase0_bucket_tilize.py <ops_perf_results.csv> <hf_model_dir>

Output: prints (weight_tilize_count, activation_tilize_count, weight_fraction).
"""
import csv
import sys
from pathlib import Path


def load_weight_shapes(hf_model_dir: str) -> set[tuple[int, ...]]:
    """Load all (named_parameters) shapes from an HF model dir."""
    import torch
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(hf_model_dir, torch_dtype=torch.float32)
    shapes = set()
    for _name, param in model.named_parameters():
        if param.numel() < 1000:
            continue
        shapes.add(tuple(param.shape))
    return shapes


def pad_to_tile(dim: int, tile: int = 32) -> int:
    return ((dim + tile - 1) // tile) * tile


def _parse_pad_logical(cell: str) -> int:
    """Parse a Tracy PAD[LOGICAL] cell, e.g. '128[128]' -> 128 (padded value).

    Falls back to a plain int parse for older fixtures.
    """
    cell = cell.strip()
    if "[" in cell:
        cell = cell.split("[", 1)[0]
    return int(cell)


def shape_matches_weight(input_shape: tuple[int, ...],
                         weight_shapes: set[tuple[int, ...]]) -> bool:
    """Match modulo tile-padding. input_shape is from the four W/Z/Y/X cols."""
    # Drop leading 1s (Tracy emits 4D; weights may be 1D/2D/3D)
    trimmed = tuple(d for d in input_shape if d != 1)
    for w in weight_shapes:
        padded = tuple(pad_to_tile(d) for d in w)
        if trimmed == padded:
            return True
        if trimmed == w:
            return True
    return False


def main(csv_path: str, hf_model_dir: str) -> None:
    weight_shapes = load_weight_shapes(hf_model_dir)
    print(f"Loaded {len(weight_shapes)} unique weight shapes from {hf_model_dir}")

    weight_count = 0
    activation_count = 0

    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Tracy emits the op as "TilizeWithValPaddingDeviceOperation".
            # We match by prefix to stay robust to suffix drift.
            op_code = row.get("OP CODE", "").strip()
            if not op_code.startswith("TilizeWithValPadding"):
                continue
            try:
                w = _parse_pad_logical(row["INPUT_0_W_PAD[LOGICAL]"])
                z = _parse_pad_logical(row["INPUT_0_Z_PAD[LOGICAL]"])
                y = _parse_pad_logical(row["INPUT_0_Y_PAD[LOGICAL]"])
                x = _parse_pad_logical(row["INPUT_0_X_PAD[LOGICAL]"])
            except (KeyError, ValueError):
                continue
            input_shape = (w, z, y, x)
            if shape_matches_weight(input_shape, weight_shapes):
                weight_count += 1
            else:
                activation_count += 1

    total = weight_count + activation_count
    if total == 0:
        print("No TilizeWithValPadding ops found in CSV.")
        return
    fraction = weight_count / total
    print(f"weight_tilize:     {weight_count}")
    print(f"activation_tilize: {activation_count}")
    print(f"weight_fraction:   {fraction:.2%}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(2)
    main(sys.argv[1], sys.argv[2])
