"""
Feature Extraction Pipeline using Grounding DINO and SAM
Extracts visual features from images for training purposes.

Usage:
    python extract_features.py --dataset robo_demo --batch_size 1 --split train
"""
print("started")
import os
import time
import copy
import argparse
from typing import Dict, List, Tuple
import numpy as np

def setup_args():
    """
    Parse command line arguments for feature extraction pipeline.
    """
    parser = argparse.ArgumentParser('Argparse options for feature extraction pipeline')
    parser.add_argument('--dataset', default='rap', help='Dataset to process')
    parser.add_argument('--features_folder', type=str,
                        default='./features_folder/',
                        help='Features storage directory')
    parser.add_argument('--data_folder', type=str,
                        default='/fsx/ad/vlm/github_datasets_test/',
                        help='Main dataset storage directory')
    parser.add_argument('--save_folder', type=str,
                        default='./save_folder/',
                        help='Results storage directory')

    # Device configuration: allow user to pass a comma-separated list of visible device IDs
    parser.add_argument('--device', type=str, default=None, help='Device to process (e.g. cuda:0). If None, derived from --device_ids')
    parser.add_argument('--device_ids', type=str, default="0,1,2,3",
                        help='Comma-separated list of physical device ids to expose (e.g. "0,1")')

    # Data loading configuration
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--split', default='train', help='Data split (train/val/test)')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of dataloader workers')
    parser.add_argument('--shuffle', action='store_true', help='Shuffle the dataloader indices')
    parser.add_argument('--multi_concept', action='store_true', help='For This-is-My-Img dataset !')

    # Model configuration - DINO
    parser.add_argument('--dino_img_size', type=int, default=518, help='DINO image size')
    parser.add_argument('--dino_patch_size', type=int, default=14, help='DINO transformer patch size')
    parser.add_argument('--dino_embedding_size', type=int, default=1024, help='DINO transformer embedding size')

    # Model configuration - VLM
    parser.add_argument('--vlm_model', default="OpenGVLab/InternVL3-14B", help='Which VLM model to use')

    # Training views configuration
    parser.add_argument('--task', default='extraction', help='Task to be solved')
    parser.add_argument('--training_views_path', default=None, help='Path to training features for each personalized object')
    parser.add_argument('--n_training_views', default=5, type=int, help='Number of training views for each personalized object')

    # Detection and masking parameters
    parser.add_argument('--min_mask_area', type=int, default=9000, help='Ignore SAM masks smaller than this size in pixels')
    parser.add_argument('--grounding_sam', action='store_true', help='Whether to use Grounding SAM for mask extraction')

    # Augmentation configuration
    parser.add_argument('--variation', default='normal', help='Training views to use: normal, augmented or 1.1')
    parser.add_argument('--n_augment', type=int, default=9, help='Number of augmentations per training view')

    # Visualization
    parser.add_argument('--show', action='store_true', help='Show image and corresponding labels')

    args, _ = parser.parse_known_args()
    # convert device_ids string -> list[int]
    args.device_ids = [int(x.strip()) for x in str(args.device_ids).split(',') if x.strip() != ""]
    return args


def configure_environment(args) -> None:
    """
    Configure CUDA environment based on device IDs from arguments.
    This MUST be called BEFORE importing torch or any library that queries CUDA.
    It also sets args.device to the in-process visible device index (e.g. 'cuda:0').
    """
    if hasattr(args, 'device_ids') and args.device_ids:
        # Convert list of device IDs to comma-separated string (no spaces)
        device_ids_str = ','.join(map(str, args.device_ids))
        os.environ["CUDA_VISIBLE_DEVICES"] = device_ids_str
        # Inside this Python process, visible GPUs will be reindexed 0..(n-1).
        # Use the first visible device as default unless user specified --device explicitly.
        if args.device is None:
            args.device = "cuda:0"
        print(f"✓ CUDA_VISIBLE_DEVICES set to: {device_ids_str}")
        print(f"✓ Using {len(args.device_ids)} GPU(s) (process-visible indices 0..{len(args.device_ids)-1})")
    else:
        # Default to first GPU
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        if args.device is None:
            args.device = "cuda:0"
        print("✓ CUDA_VISIBLE_DEVICES set to: 0 (default)")
        print("✓ Using 1 GPU (process-visible index 0)")

# ------------------ Now parse args and configure env BEFORE torch ------------------
args = setup_args()

# Example modifications you had in your script (kept)
#args.dataset='myvlm'
#args.task = 'extraction'
#args.device_ids=[0, 1, 2, 3]
#args.vlm_model="OpenGVLab/InternVL3-14B"
#args.n_training_views = 1
#args.variation="augment"
#args.n_augment=1
#args.grounding_sam = True
if args.variation=='normal':
    args.n_augment=1

# configure environment BEFORE importing torch / cuda-dependent libs
configure_environment(args)

# Now it is safe to import torch and other cuda-using libraries
import torch
import matplotlib.pyplot as plt

# Optionally set torch to use the selected device index (in-process index)
try:
    # pick numeric index from args.device (e.g. 'cuda:0' -> 0)
    if args.device and args.device.startswith("cuda:"):
        device_idx = int(args.device.split(':')[1])
        torch.cuda.set_device(device_idx)
except Exception as e:
    print(f"Warning: couldn't set torch.cuda device explicitly: {e}")

# now do the rest of the imports that may depend on torch/cuda availability
from utils import (
    apply_mask_dino,
    save_sample_features,
    check_save
)
from loader import get_dataloader
from model import Train_Extractor
print("end includes")

# Constants
PROGRESS_LOG_INTERVAL = 50

# ... (rest of your functions remain unchanged) ...

def initialize_feature_dict() -> Dict[str, List]:
    return {
        "npy_image": [],
        "class_label": [],
        "img_path": [],
        "feat_path": [],
        "dino_img_features": [],
        "dino_obj_features": [],
        "dino_mask": [],
    }

def process_labels(labels: List[str]) -> List[str]:
    return [label.replace('-', ' ') for label in labels]

def extract_batch_features(
    data_batch: Tuple,
    extractor: Train_Extractor,
    args
) -> Dict[str, List]:
    img, img_dino, path, label, *_ = data_batch
    image = np.stack(img)
    image_dino = np.stack(img_dino)
    features = initialize_feature_dict()
    class_labels = process_labels(label)
    use_grounding_sam = bool(args.grounding_sam)
    print("Using Grounding SAM? ", use_grounding_sam)
    dino_mask = extractor.forward_grounding_dino(image_dino, class_labels, use_g_sam=use_grounding_sam)
    dino_img_features = extractor.forward_dino(image_dino)
    for batch_idx in range(args.batch_size):
        if check_save(args, path[batch_idx]):
            features["dino_img_features"].append(dino_img_features[batch_idx])
            features["npy_image"].append(image_dino[batch_idx])
            features["class_label"].append(class_labels[batch_idx])
            features["img_path"].append(path[batch_idx])
            features["dino_mask"].append(copy.deepcopy(dino_mask))
            dino_obj_features = apply_mask_dino(args, dino_mask, dino_img_features[batch_idx])
            features["dino_obj_features"].append(dino_obj_features[batch_idx])
    return features

def stack_features(features: Dict[str, List]) -> Dict[str, np.ndarray]:
    if len(features["npy_image"]) > 0:
        features["npy_image"] = np.stack(features["npy_image"])
        features["dino_img_features"] = torch.stack(features["dino_img_features"])
        features["dino_obj_features"] = torch.stack(features["dino_obj_features"])
        features["dino_mask"] = np.stack(features["dino_mask"])
    return features

def print_config(args) -> None:
    print("\n" + "="*70)
    print("FEATURE EXTRACTION CONFIGURATION")
    print("="*70)
    print(f"Dataset:           {args.dataset}")
    print(f"Data folder:       {args.data_folder}")
    print(f"Features folder:   {args.features_folder}")
    print(f"Split:             {args.split}")
    print(f"Device:            {args.device}")
    print(f"Batch size:        {args.batch_size}")
    print(f"Grounding SAM:     {args.grounding_sam}")
    print(f"Variation:         {args.variation}")
    print(f"Augmentations:     {args.n_augment}")
    print(f"Min mask area:     {args.min_mask_area}")
    print("="*70 + "\n")

# Main pipeline
print('Started - Grounding DINO/SAM Feature Extraction')
start_time = time.time()

print_config(args)

print("Initializing data loader and model...")
data_loader, _, _ = get_dataloader(args=args)

# IMPORTANT: make sure Train_Extractor uses args.device (now e.g. 'cuda:0')
extractor = Train_Extractor(args)  # inside it, .to(args.device) should succeed
print(f"Data loader ready with {len(data_loader)} batches\n")

# Process data with augmentations
if args.variation == 'augment':
    # First pass: extract features from normal (non-augmented) images
    total_batches = len(data_loader) * (args.n_augment + 1)  # +1 for normal pass
    processed_batches = 0
    
    print(f'Processing normal (non-augmented) images...')
    # Temporarily disable augmentation
    original_variation = args.variation
    args.variation = 'normal'
    
    for batch_idx, data_batch in enumerate(data_loader):
        if batch_idx % PROGRESS_LOG_INTERVAL == 0:
            progress = (processed_batches / total_batches) * 100
            print(f'  Batch {batch_idx}/{len(data_loader)} ({progress:.1f}% complete)')
        features = extract_batch_features(data_batch, extractor, args)
        if len(features["npy_image"]) > 0:
            features = stack_features(features)
            features = save_sample_features(args, features, 0)  # Save with suffix _0
        processed_batches += 1
    
    # Restore augmentation setting
    args.variation = original_variation
    
    # Second pass: extract features from augmented images
    for augment_idx in range(args.n_augment):
        print(f'Processing augmentation {augment_idx + 1}/{args.n_augment}')
        for batch_idx, data_batch in enumerate(data_loader):
            if batch_idx % PROGRESS_LOG_INTERVAL == 0:
                progress = (processed_batches / total_batches) * 100
                print(f'  Batch {batch_idx}/{len(data_loader)} ({progress:.1f}% complete)')
            features = extract_batch_features(data_batch, extractor, args)
            if len(features["npy_image"]) > 0:
                features = stack_features(features)
                features = save_sample_features(args, features, augment_idx + 1)
            processed_batches += 1
else:
    # Normal behavior for non-augment variations
    total_batches = len(data_loader) * args.n_augment
    processed_batches = 0
    
    for augment_idx in range(args.n_augment):
        print(f'Processing augmentation {augment_idx + 1}/{args.n_augment}')
        for batch_idx, data_batch in enumerate(data_loader):
            if batch_idx % PROGRESS_LOG_INTERVAL == 0:
                progress = (processed_batches / total_batches) * 100
                print(f'  Batch {batch_idx}/{len(data_loader)} ({progress:.1f}% complete)')
            features = extract_batch_features(data_batch, extractor, args)
            if len(features["npy_image"]) > 0:
                features = stack_features(features)
                features = save_sample_features(args, features, augment_idx + 1)
            processed_batches += 1

elapsed_time = time.time() - start_time
hours, remainder = divmod(elapsed_time, 3600)
minutes, seconds = divmod(remainder, 60)
print("\n" + "="*70)
print("EXTRACTION COMPLETE")
print("="*70)
print(f"Total time: {int(hours):02d}:{int(minutes):02d}:{seconds:.2f}")
print(f"Processed {processed_batches} batches")
print("="*70 + "\n")
