#!/usr/bin/env python3
"""Compare two HuggingFace model weights by keys and parameter shapes."""

import argparse
from collections import OrderedDict

from safetensors import safe_open
from transformers import AutoConfig


def load_state_dict(model_path: str) -> OrderedDict:
    """Load state dict from a HuggingFace model path."""
    import os
    import glob
    
    state_dict = OrderedDict()
    
    # Check for safetensors files first
    safetensor_files = glob.glob(os.path.join(model_path, "*.safetensors"))
    if safetensor_files:
        for f in sorted(safetensor_files):
            with safe_open(f, framework="pt", device="cpu") as st:
                for key in st.keys():
                    state_dict[key] = st.get_tensor(key)
        return state_dict
    
    # Fall back to pytorch bin files
    import torch
    bin_files = glob.glob(os.path.join(model_path, "*.bin"))
    if bin_files:
        for f in sorted(bin_files):
            state_dict.update(torch.load(f, map_location="cpu", weights_only=True))
        return state_dict
    
    # Try loading as a single model
    from transformers import AutoModel
    model = AutoModel.from_pretrained(model_path)
    return model.state_dict()


def compare_weights(path1: str, path2: str, show_matching: bool = False):
    """Compare two model weights and report differences."""
    print(f"Loading model 1: {path1}")
    state_dict1 = load_state_dict(path1)
    print(f"  Loaded {len(state_dict1)} parameters\n")
    
    print(f"Loading model 2: {path2}")
    state_dict2 = load_state_dict(path2)
    print(f"  Loaded {len(state_dict2)} parameters\n")
    
    keys1 = set(state_dict1.keys())
    keys2 = set(state_dict2.keys())
    
    # Keys only in model 1
    only_in_1 = keys1 - keys2
    if only_in_1:
        print(f"{'='*60}")
        print(f"Keys ONLY in model 1 ({len(only_in_1)}):")
        print(f"{'='*60}")
        for key in sorted(only_in_1):
            print(f"  {key}: {tuple(state_dict1[key].shape)}")
    
    # Keys only in model 2
    only_in_2 = keys2 - keys1
    if only_in_2:
        print(f"\n{'='*60}")
        print(f"Keys ONLY in model 2 ({len(only_in_2)}):")
        print(f"{'='*60}")
        for key in sorted(only_in_2):
            print(f"  {key}: {tuple(state_dict2[key].shape)}")
    
    # Common keys with shape/dtype mismatches
    common_keys = keys1 & keys2
    shape_mismatches = []
    dtype_mismatches = []
    shape_matches = []
    
    for key in sorted(common_keys):
        tensor1 = state_dict1[key]
        tensor2 = state_dict2[key]
        shape1 = tuple(tensor1.shape)
        shape2 = tuple(tensor2.shape)
        dtype1 = tensor1.dtype
        dtype2 = tensor2.dtype
        
        if shape1 != shape2:
            shape_mismatches.append((key, shape1, shape2))
        else:
            # Compute mean difference for tensors with same shape
            mean1 = tensor1.float().mean().item()
            mean2 = tensor2.float().mean().item()
            mean_diff = abs(mean1 - mean2)
            shape_matches.append((key, shape1, dtype1, dtype2, mean1, mean2, mean_diff))
        
        if dtype1 != dtype2:
            dtype_mismatches.append((key, dtype1, dtype2))
    
    if shape_mismatches:
        print(f"\n{'='*60}")
        print(f"Shape MISMATCHES ({len(shape_mismatches)}):")
        print(f"{'='*60}")
        for key, shape1, shape2 in shape_mismatches:
            print(f"  {key}:")
            print(f"    Model 1: {shape1}")
            print(f"    Model 2: {shape2}")
    
    if dtype_mismatches:
        print(f"\n{'='*60}")
        print(f"Dtype MISMATCHES ({len(dtype_mismatches)}):")
        print(f"{'='*60}")
        for key, dtype1, dtype2 in dtype_mismatches:
            print(f"  {key}:")
            print(f"    Model 1: {dtype1}")
            print(f"    Model 2: {dtype2}")
    
    if show_matching and shape_matches:
        print(f"\n{'='*60}")
        print(f"Matching parameters ({len(shape_matches)}):")
        print(f"{'='*60}")
        for key, shape, dtype1, dtype2, mean1, mean2, mean_diff in shape_matches:
            dtype_str = f"{dtype1}" if dtype1 == dtype2 else f"{dtype1} vs {dtype2}"
            print(f"  {key}: {shape} [{dtype_str}]")
            print(f"    Mean: {mean1:.6e} vs {mean2:.6e} (diff: {mean_diff:.6e})")
    
    # Show mean differences summary for non-zero diffs
    significant_diffs = [(k, s, d) for k, s, _, _, _, _, d in shape_matches if d > 1e-6]
    if significant_diffs:
        print(f"\n{'='*60}")
        print(f"Parameters with significant mean difference (>{1e-6}) ({len(significant_diffs)}):")
        print(f"{'='*60}")
        for key, shape, mean_diff in sorted(significant_diffs, key=lambda x: -x[2]):
            print(f"  {key}: {shape}, mean_diff={mean_diff:.6e}")
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"  Model 1 total params: {len(keys1)}")
    print(f"  Model 2 total params: {len(keys2)}")
    print(f"  Only in model 1:      {len(only_in_1)}")
    print(f"  Only in model 2:      {len(only_in_2)}")
    print(f"  Common keys:          {len(common_keys)}")
    print(f"  Shape mismatches:     {len(shape_mismatches)}")
    print(f"  Dtype mismatches:     {len(dtype_mismatches)}")
    print(f"  Shape matches:        {len(shape_matches)}")
    print(f"  Significant mean diff:{len(significant_diffs)}")
    
    if not only_in_1 and not only_in_2 and not shape_mismatches:
        print("\n✓ Models have identical structure!")
        if not dtype_mismatches and not significant_diffs:
            print("✓ Models have identical dtypes and values!")
    else:
        print("\n✗ Models have structural differences.")


def main():
    parser = argparse.ArgumentParser(
        description="Compare two HuggingFace model weights by keys and shapes"
    )
    parser.add_argument(
        "model1",
        type=str,
        help="Path to first model (local path or HuggingFace model ID)",
    )
    parser.add_argument(
        "model2",
        type=str,
        help="Path to second model (local path or HuggingFace model ID)",
    )
    parser.add_argument(
        "--show-matching",
        action="store_true",
        help="Also show parameters that match between models",
    )
    args = parser.parse_args()
    
    compare_weights(args.model1, args.model2, args.show_matching)


if __name__ == "__main__":
    main()
