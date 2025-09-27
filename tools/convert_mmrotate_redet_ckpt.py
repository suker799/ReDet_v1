#!/usr/bin/env python3
"""Convert mmrotate ReDet checkpoints to ReDet repository format."""
import argparse
import os
from typing import Dict, Any

import torch


SUFFIX_WEIGHTS = ".conv.weight"
SUFFIX_BIAS = ".conv.bias"
TARGET_WEIGHTS = ".conv.filter"
TARGET_BIAS = ".conv.expanded_bias"


def remap_key(key: str) -> str:
    """Remap mmrotate convolution parameter keys to match ReDet naming."""
    if key.endswith(SUFFIX_WEIGHTS):
        return key[: -len(SUFFIX_WEIGHTS)] + TARGET_WEIGHTS
    if key.endswith(SUFFIX_BIAS):
        return key[: -len(SUFFIX_BIAS)] + TARGET_BIAS
    return key


def convert_state_dict(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Convert keys inside the state dict to the expected ReDet format."""
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = remap_key(key)
        if new_key in new_state_dict:
            raise KeyError(
                f"Key collision detected when remapping '{key}' to '{new_key}'."
            )
        new_state_dict[new_key] = value
    return new_state_dict


def convert_checkpoint(src_path: str, dst_path: str) -> None:
    """Load a checkpoint, convert its keys, and save it back to disk."""
    checkpoint = torch.load(src_path, map_location="cpu")

    if "state_dict" not in checkpoint:
        raise KeyError(
            "The provided checkpoint does not contain a 'state_dict' entry."
        )

    checkpoint["state_dict"] = convert_state_dict(checkpoint["state_dict"])

    dst_dir = os.path.dirname(dst_path)
    if dst_dir:
        os.makedirs(dst_dir, exist_ok=True)
    torch.save(checkpoint, dst_path)



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert an mmrotate ReDet checkpoint so it can be used "
            "with the ReDet repository."
        )
    )
    parser.add_argument(
        "src",
        type=str,
        help=(
            "Path to the mmrotate checkpoint file, e.g. "
            "ReDet_re50_refpn_3x_hrsc2016-d1b4bd29.pth"
        ),
    )
    parser.add_argument(
        "dst",
        type=str,
        nargs="?",
        default=os.path.join(
            "checkpoints", "redet_re50_refpn_3x_hrsc2016_mmrotate.pth"
        ),
        help=(
            "Destination path for the converted checkpoint. "
            "Defaults to checkpoints/redet_re50_refpn_3x_hrsc2016_mmrotate.pth"
        ),
    )
    return parser.parse_args()



def main() -> None:
    args = parse_args()
    convert_checkpoint(args.src, args.dst)


if __name__ == "__main__":
    main()
