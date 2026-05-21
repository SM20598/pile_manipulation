#!/usr/bin/env python3
"""
Convert old pickle data format (list-of-dicts) to new torch.save format (dict-of-tensors).

Old format:
    data.pkl = [{"state": tensor, "state_": tensor, "action": (p_start, p_stop, angle)}, ...]

New format:
    data.pt = {"states": tensor, "states_": tensor, "p_starts": tensor, "p_stops": tensor, "angles": tensor}

Usage:
    python convert_pkl_to_pt.py /path/to/data/directory [--delete-old]

Example:
    python convert_pkl_to_pt.py training/ --delete-old
    python convert_pkl_to_pt.py training/  # keeps old files
"""

import pickle
import torch
from pathlib import Path
import argparse
import sys


def convert_pkl_to_pt(pkl_path, keep_old=True):
    """
    Convert a single pickle file to torch format.
    
    Args:
        pkl_path: Path to .pkl file
        keep_old: If False, delete the original .pkl file after conversion
    
    Returns:
        Tuple of (num_samples_converted, pt_path)
    """
    pkl_path = Path(pkl_path)
    
    if not pkl_path.exists():
        print(f"❌ File not found: {pkl_path}")
        return 0, None
    
    print(f"📖 Reading: {pkl_path.name}")
    
    try:
        with open(pkl_path, 'rb') as f:
            old_data = pickle.load(f)
    except Exception as e:
        print(f"❌ Error loading pickle: {e}")
        return 0, None
    
    if not isinstance(old_data, list) or len(old_data) == 0:
        print(f"❌ Invalid format: expected non-empty list, got {type(old_data)}")
        return 0, None
    
    # Check structure of first sample
    first_sample = old_data[0]
    if not isinstance(first_sample, dict) or "state" not in first_sample:
        print(f"❌ Invalid sample format: {first_sample.keys()}")
        return 0, None
    
    # Convert list-of-dicts to dict-of-tensors
    print(f"🔄 Converting {len(old_data)} samples...")
    
    states_list = []
    states__list = []
    p_starts_list = []
    p_stops_list = []
    angles_list = []
    
    for sample in old_data:
        states_list.append(sample["state"])
        states__list.append(sample["state_"])
        
        action = sample["action"]
        p_start, p_stop, angle = action
        p_starts_list.append(p_start)
        p_stops_list.append(p_stop)
        angles_list.append(angle)
    
    # Stack into tensors (on CPU)
    new_data = {
        "states": torch.stack(states_list),
        "states_": torch.stack(states__list),
        "p_starts": torch.stack(p_starts_list),
        "p_stops": torch.stack(p_stops_list),
        "angles": torch.stack(angles_list),
    }
    
    # Save as torch format
    pt_path = pkl_path.with_suffix('.pt')
    print(f"💾 Saving: {pt_path.name}")
    torch.save(new_data, str(pt_path))
    
    # Print stats
    print(f"   ✓ states:   {new_data['states'].shape}")
    print(f"   ✓ states_:  {new_data['states_'].shape}")
    print(f"   ✓ p_starts: {new_data['p_starts'].shape}")
    print(f"   ✓ p_stops:  {new_data['p_stops'].shape}")
    print(f"   ✓ angles:   {new_data['angles'].shape}")
    
    # Delete old pickle if requested
    if not keep_old:
        pkl_path.unlink()
        print(f"🗑️  Deleted: {pkl_path.name}")
    
    return len(old_data), pt_path


def main():
    parser = argparse.ArgumentParser(
        description="Convert old pickle data format to new torch.save format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
            python convert_pkl_to_pt.py training/
            python convert_pkl_to_pt.py training/ --delete-old
            python convert_pkl_to_pt.py training/ --pattern "_data.pkl"
                    """,
    )
    parser.add_argument("directory", type=str, help="Directory containing .pkl files")
    parser.add_argument(
        "--delete-old", 
        action="store_true", 
        help="Delete original .pkl files after conversion"
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="_data.pkl",
        help="Pickle file pattern to match (default: '_data.pkl')",
    )
    
    args = parser.parse_args()
    
    data_dir = Path(args.directory)
    if not data_dir.is_dir():
        print(f"❌ Not a directory: {data_dir}")
        sys.exit(1)
    
    # Find all matching pickle files
    pkl_files = sorted(data_dir.glob(f"*{args.pattern}"))
    
    if not pkl_files:
        print(f"⚠️  No files matching '*{args.pattern}' found in {data_dir}")
        sys.exit(0)
    
    print(f"🎯 Found {len(pkl_files)} file(s) to convert\n")
    
    total_samples = 0
    converted_files = 0
    
    for pkl_path in pkl_files:
        print(f"\n{'='*60}")
        n_samples, pt_path = convert_pkl_to_pt(pkl_path, keep_old=not args.delete_old)
        if pt_path is not None:
            total_samples += n_samples
            converted_files += 1
    
    print(f"\n{'='*60}")
    print(f"✅ Conversion complete!")
    print(f"   Files converted: {converted_files}")
    print(f"   Total samples: {total_samples}")
    if args.delete_old:
        print(f"   Old .pkl files: DELETED")
    else:
        print(f"   Old .pkl files: KEPT")


if __name__ == "__main__":
    main()
