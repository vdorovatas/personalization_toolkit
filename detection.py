"""
Unified Object Detection and Evaluation Script

This script evaluates personalized object detection using DINO features and SAM masks.
Supports both single-concept and multi-concept evaluation modes.
Processes images, detects objects, and calculates precision and recall metrics.

Usage:
    Single-concept: python detection_eval.py --dataset yollava --split validation
    Multi-concept:  python detection_eval.py --dataset rap --split validation --multi_concept
"""

import sys
import copy
from pathlib import Path
import argparse
import os

import matplotlib.pyplot as plt
from PIL import Image
##temporary
import random
import numpy as np
import torch

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

######


from utils import (
    draw_bounding_box,
    mask_to_normalized_bbox,
    nested_list_contains,
    get_objects_features,
    apply_mask_dino,
    gather_masks,
    check_bbox,
    substring_before_last_dash
)
from loader import get_dataloader
from model import Extractor






# Constants
COLORS = ['green', 'blue', 'purple', 'orange', 'pink', 'gray', 'yellow', 'red']

# Dataset-specific minimum mask areas
DATASET_MIN_MASK_AREAS = {
    'myvlm': 2000,
    'this-is-my': 2000,
    'yollava': 9000,
    'rap': 9000
}

FIXED_THRESHOLDS = [0.75]


def setup_args():
    """
    Parse command line arguments for detection evaluation pipeline.
    
    Returns:
        Parsed arguments object
    """
    parser = argparse.ArgumentParser('Object Detection Evaluation')
    
    # Dataset configuration
    parser.add_argument('--dataset', default='yollava', 
                        help='Dataset to process')
    parser.add_argument('--features_folder', type=str, 
                        default='./features_folder/', 
                        help='Features storage directory')
    parser.add_argument('--data_folder', type=str, 
                        default='/fsx/ad/vlm/github_datasets_test/', 
                        help='Main dataset storage directory')
    
    # Device configuration
    parser.add_argument('--device', type=str, default='cuda:1',
                        help='Device to process')
    parser.add_argument('--device_ids', default=[1],
                        help='Visible devices to process')
    
    # Data loading configuration
    parser.add_argument('--batch_size', type=int, default=1, 
                        help='Batch size')
    parser.add_argument('--split', default='test', 
                        help='Data split (train/test)')
    parser.add_argument('--test_split', default=None, 
                        choices=["Positive","Fake","Negative (Hard)","Negative (Other)"],
                        help='Test Split for This-is-My-Img dataset')
    parser.add_argument('--num_workers', type=int, default=8, 
                        help='Number of dataloader workers')
    parser.add_argument('--shuffle', action='store_true', 
                        help='Shuffle the dataloader indices')
    
    # Model configuration - DINO
    parser.add_argument('--dino_img_size', type=int, default=518, 
                        help='DINO image size')
    parser.add_argument('--dino_patch_size', type=int, default=14, 
                        help='DINO transformer patch size')
    parser.add_argument('--dino_embedding_size', type=int, default=1024, 
                        help='DINO transformer embedding size')
    
    # Training views configuration
    parser.add_argument('--task', default='detection', help='Task to be solved')
    parser.add_argument('--n_training_views', type=int, default=None,
                        help='Number of training views for each personalized object')
    
    # Detection and masking parameters
    parser.add_argument('--detect_thresh', type=float, default=0.75, 
                        help='Detection threshold for personalized objects')
    parser.add_argument('--min_mask_area', type=int, default=9000, 
                        help='Ignore SAM masks smaller than this size in pixels')
    parser.add_argument('--box_overlap_thresh', type=float, default=100, 
                        help='Box overlap threshold for filtering')
    parser.add_argument('--grounding_sam', action='store_true',
                        help='Whether to use Grounding SAM for mask extraction')
    
    # Augmentation configuration
    parser.add_argument('--variation', default='normal', 
                        help='Training views variation: normal or augment')
    
    # Multi-concept mode
    parser.add_argument('--multi_concept', action='store_true',
                        help='Enable multi-concept evaluation mode (for category pairs)')
    
    # Visualization
    parser.add_argument('--show', action='store_true', 
                        help='Show image and corresponding labels')
    
    args, _ = parser.parse_known_args()
    return args


def configure_environment(args) -> None:
    """
    Configure CUDA environment based on device IDs from arguments.
    
    Args:
        args: Arguments object containing device_ids list
    """
    if hasattr(args, 'device_ids') and args.device_ids:
        device_ids_str = ','.join(map(str, args.device_ids))
        os.environ["CUDA_VISIBLE_DEVICES"] = device_ids_str
        print(f"✓ CUDA_VISIBLE_DEVICES set to: {device_ids_str}")
        print(f"✓ Using {len(args.device_ids)} GPU(s)")
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        print("✓ CUDA_VISIBLE_DEVICES set to: 0 (default)")


def setup_thresholds(args):
    """Get list of thresholds to evaluate."""
    return FIXED_THRESHOLDS


def prepare_object_names(args, my_objects):
    """
    Prepare object names based on dataset type.
    
    Args:
        args: Configuration arguments
        my_objects: List of object names
        
    Returns:
        list: Processed object names
    """
    if args.dataset in ['yollava', 'this-is-my', 'rap']:
        return [substring_before_last_dash(obj) for obj in my_objects]
    else:
        return [obj.replace(' ', '_') for obj in my_objects]


def extract_labels_from_path(path, dataset, multi_concept=False):
    """
    Extract object labels from image path.
    
    Args:
        path: Image file path
        dataset: Dataset name
        multi_concept: Whether to handle multi-concept cases with "_and_" pattern
        
    Returns:
        list: Extracted labels
    """
    if dataset == 'yollava' or dataset=='myvlm':
        folder_name = path.split('/')[-2]
    else:
        folder_name = path.split('/')[-2].split('-')[0]
    
    # Check if this is a multi-object case with "_and_" pattern
    if multi_concept and "_and_" in folder_name:
        labels = folder_name.split("_and_")
    else:
        labels = [folder_name]
    
    # Handle double-labeled images (legacy support for this-is-my)
    if path.split('_')[-1] == 'double.png':
        label = labels[0]
        double_label_pairs = {
            'Reynards keyboard': 'Reynards work chair',
            'Reynards work chair': 'Reynards keyboard',
            'Nikkis car': 'Nikkis camper bag',
            'Nikkis camper bag': 'Nikkis car'
        }
        if label in double_label_pairs:
            labels.append(double_label_pairs[label])
    
    return labels


def process_single_image(args, data_batch, mask_extractor, all_obj_features, my_objects_raw):
    """
    Process a single image batch and detect objects.
    
    Args:
        args: Configuration arguments
        data_batch: Batch of data containing images and paths
        mask_extractor: Model for feature extraction
        all_obj_features: Pre-computed object features
        my_objects_raw: List of object names
        
    Returns:
        list: Detection results including personalized_objects and labels
    """
    image, dino_image, path, *_ = data_batch

    print("\n" + "="*80)
    print("IMAGE IDENTITY CHECK - Batch 0")
    print("="*80)
    
    # 1. Check paths
    print(f"\n1. IMAGE PATH:")
    print(f"   path[0]: {path[0]}")
    
    # 2. Check raw image data
    print(f"\n2. RAW IMAGE DATA:")
    print(f"   image type: {type(image)}")
    print(f"   image length: {len(image)}")
    print(f"   image[0] shape: {image[0].shape}")
    print(f"   image[0] dtype: {image[0].dtype}")
    print(f"   image[0] min: {image[0].min()}")
    print(f"   image[0] max: {image[0].max()}")
    print(f"   image[0] mean: {image[0].mean():.6f}")
    print(f"   image[0] sum: {image[0].sum()}")
    
    # 3. Check DINO processed image
    print(f"\n3. DINO IMAGE DATA:")
    print(f"   dino_image type: {type(dino_image)}")
    print(f"   dino_image length: {len(dino_image)}")
    print(f"   dino_image[0] shape: {dino_image[0].shape}")
    print(f"   dino_image[0] dtype: {dino_image[0].dtype}")
    print(f"   dino_image[0] min: {dino_image[0].min()}")
    print(f"   dino_image[0] max: {dino_image[0].max()}")
    print(f"   dino_image[0] mean: {dino_image[0].mean():.6f}")
    print(f"   dino_image[0] sum: {dino_image[0].sum()}")
    
    # 4. Hash the image data to verify it's identical
    import hashlib
    img_bytes = image[0].tobytes()
    img_hash = hashlib.md5(img_bytes).hexdigest()
    print(f"\n4. IMAGE HASH:")
    print(f"   MD5 hash: {img_hash}")
    
    dino_bytes = dino_image[0].tobytes()
    dino_hash = hashlib.md5(dino_bytes).hexdigest()
    print(f"   DINO image MD5 hash: {dino_hash}")
    
    # 5. Check first few pixel values
    print(f"\n5. SAMPLE PIXEL VALUES:")
    print(f"   image[0] top-left 3x3:")
    print(f"   {image[0][:3, :3, 0]}")  # First channel
    print(f"   dino_image[0] top-left 3x3:")
    print(f"   {dino_image[0][:3, :3, 0]}")  # First channel
    
    print("="*80 + "\n")



    
    # Extract DINO features
    dino_img_features = mask_extractor.forward_dino(dino_image)
    
    # Generate masks using Grounding DINO
    sam_masks = mask_extractor.forward_grounding_dino(dino_image, ['object.'])

    #sam_masks = mask_extractor.forward_sam(dino_image)
    
    results = []
    
    for batch_idx in range(args.batch_size):
        # Extract labels for this image
        labels = extract_labels_from_path(
            path[batch_idx], 
            args.dataset, 
            multi_concept=args.multi_concept
        )
        
        # Get mean DINO features for each mask
        category_features = apply_mask_dino(args, sam_masks, dino_img_features[batch_idx])
        category_features = torch.stack([feat.mean(dim=0) for feat in category_features])

        nan_count = torch.isnan(category_features).any(dim=1).sum().item()
        valid_count = (~torch.isnan(category_features).any(dim=1)).sum().item()
        
        print(f"    Total masks: {len(category_features)}")
        print(f"    NaN masks: {nan_count}")
        print(f"    Valid masks: {valid_count}")
        print(f"    Path: {path[0].split('/')[-1]}")
        
        # Detect and gather masks
        full_masks, full_masks_labels, something_detected, sim_scores = gather_masks(
            args, category_features, sam_masks[batch_idx], all_obj_features
        )


        #Temporary
        # Add this right after gather_masks is called
        if batch_idx == 0:  # Only debug first batch
            print("\n" + "="*80)
            print("DEBUG: First batch detailed analysis")
            print("="*80)
            
            # 1. Check sam_masks
            print(f"\n1. SAM MASKS:")
            print(f"   Type: {type(sam_masks)}")
            print(f"   Length: {len(sam_masks)}")
            print(f"   sam_masks[0] type: {type(sam_masks[0])}")
            print(f"   sam_masks[0] shape: {sam_masks[0].shape}")
            print(f"   Number of masks detected: {sam_masks[0].shape[0]}")
            
            # 2. Check dino_img_features
            print(f"\n2. DINO IMAGE FEATURES:")
            print(f"   Type: {type(dino_img_features)}")
            print(f"   Shape: {dino_img_features.shape}")
            print(f"   dino_img_features[0] shape: {dino_img_features[0].shape}")
            
            # 3. Check category_features BEFORE stacking
            category_features_raw = apply_mask_dino(args, sam_masks, dino_img_features[0])
            print(f"\n3. CATEGORY FEATURES (before stacking):")
            print(f"   Type: {type(category_features_raw)}")
            print(f"   Length: {len(category_features_raw)}")
            print(f"   First item shape: {category_features_raw[0].shape}")
            print(f"   First item mean: {category_features_raw[0].mean().item():.6f}")
            
            # 4. Check category_features AFTER stacking and mean
            category_features_stacked = torch.stack([x.mean(dim=0) for x in category_features_raw])
            print(f"\n4. CATEGORY FEATURES (after stacking & mean):")
            print(f"   Shape: {category_features_stacked.shape}")
            print(f"   First feature vector mean: {category_features_stacked[0].mean().item():.6f}")
            print(f"   First feature vector norm: {torch.norm(category_features_stacked[0]).item():.6f}")
            
            # 5. Check all_obj_features
            print(f"\n5. ALL OBJECT FEATURES:")
            print(f"   Type: {type(all_obj_features)}")
            print(f"   Shape: {all_obj_features.shape}")
            print(f"   First object features shape: {all_obj_features[0].shape}")
            print(f"   First object, first view mean: {all_obj_features[0][0].mean().item():.6f}")
            print(f"   First object, first view norm: {torch.norm(all_obj_features[0][0]).item():.6f}")
            
            # 6. Manual cosine similarity calculation for first mask vs first object
            mask_feat = category_features_stacked[0].to(args.device)  # Shape: (1024,)
            obj_feat = all_obj_features[0].to(args.device)  # Shape: (N_views, 1024)
            
            print(f"\n6. MANUAL COSINE SIMILARITY TEST:")
            print(f"   mask_feat shape: {mask_feat.shape}")
            print(f"   obj_feat shape: {obj_feat.shape}")
            
            # Compute cosine similarity
            sim_manual = torch.nn.functional.cosine_similarity(
                mask_feat.unsqueeze(0).unsqueeze(0),  # (1, 1, 1024)
                obj_feat.unsqueeze(0),  # (1, N_views, 1024)
                dim=2,
                eps=1e-8
            )
            print(f"   Similarity scores: {sim_manual[0].cpu().numpy()}")
            print(f"   Mean similarity: {sim_manual.mean().item():.6f}")
            print(f"   Max similarity: {sim_manual.max().item():.6f}")
            
            # 7. Check what gather_masks actually computes
            print(f"\n7. GATHER_MASKS SIMULATION:")
            for m_idx in range(min(3, len(category_features_stacked))):  # First 3 masks
                mask_feature = category_features_stacked[m_idx]
                
                sim = torch.nn.functional.cosine_similarity(
                    mask_feature.to(args.device), 
                    all_obj_features.to(args.device), 
                    dim=2, 
                    eps=1e-8
                )
                sim[torch.isnan(sim) == True] = 0
                
                print(f"\n   Mask {m_idx}:")
                print(f"     sim shape: {sim.shape}")
                print(f"     sim max per object: {sim.max(dim=1)[0][:5].cpu().numpy()}")  # First 5 objects
                
                sim_detect = (sim > args.detect_thresh).float().mean(dim=1)
                print(f"     Detection scores (thresh={args.detect_thresh}): {sim_detect[:5].cpu().numpy()}")
                
                detections = (sim_detect > 0)
                detected_indice = torch.where(detections == True)[0]
                print(f"     Detected object indices: {detected_indice.cpu().numpy()}")
                
                if len(detected_indice) > 0:
                    d_indice = torch.unravel_index(torch.argmax(sim, axis=None), sim.shape)[0]
                    print(f"     Best match object index: {d_indice.item()}")
                    print(f"     Best match object name: {my_objects_raw[d_indice]}")
                    print(f"     Max similarity score: {sim.max().item():.6f}")
            
            print("\n" + "="*80)
            print("END DEBUG")
            print("="*80 + "\n") 


        
        personalized_objects = []
        bboxes = []
        bbox_scores = []
        
        if something_detected:
            for mask_idx, full_mask in enumerate(full_masks):
                # Convert mask to bounding box
                bbox, bbox_raw = mask_to_normalized_bbox(full_mask)
                candidate_object = my_objects_raw[full_masks_labels[mask_idx]]
                
                personalized_objects, _, bboxes, bbox_scores = check_bbox(
                    args, bbox, bboxes, bbox_scores, sim_scores[mask_idx],
                    personalized_objects, candidate_object
                )
        
        # Visualize if enabled
        if len(personalized_objects) > 0 and args.show:
            visualize_detections(image[batch_idx], bboxes, personalized_objects, labels)
        
        results.append({
            'personalized_objects': personalized_objects,
            'labels': labels,
            'path': path[batch_idx]
        })
    
    return results


def visualize_detections(image, bboxes, detected_objects, labels):
    """
    Visualize detected objects with bounding boxes.
    
    Args:
        image: Input image array
        bboxes: List of bounding boxes
        detected_objects: List of detected object names
        labels: Ground truth labels
    """
    bbox_img = copy.deepcopy(Image.fromarray(image.astype(np.uint8)))
    detected_colors = COLORS[:len(bboxes)]
    bbox_img = draw_bounding_box(bbox_img, bboxes, detected_colors)
    
    print(f'Objects found: {detected_objects}')
    print(f'Labels: {labels}')
    plt.imshow(bbox_img)
    plt.show()


def update_counters_single_concept(detected_objects, labels, my_objects_raw, counters):
    """
    Update detection counters for single-concept mode.
    Evaluates individual object detection.
    
    Args:
        detected_objects: List of detected objects
        labels: Ground truth labels
        my_objects_raw: All possible object names
        counters: Dictionary of counter arrays
    """
    # Update detection counters
    for detected_obj in detected_objects:
        idx = my_objects_raw.index(detected_obj)
        counters['detected'][idx] += 1
    
    # Update ground truth and correct detection counters
    for label in labels:
        idx = my_objects_raw.index(label)
        counters['gt'][idx] += 1
        if label in detected_objects:
            counters['correct'][idx] += 1
    
    # Update negative counters
    for obj in my_objects_raw:
        idx = my_objects_raw.index(obj)
        if obj not in labels:
            counters['gt_neg'][idx] += 1
        if obj not in detected_objects:
            counters['detected_neg'][idx] += 1
            if obj not in labels:
                counters['correct_neg'][idx] += 1


def update_counters_multi_concept(detected_objects, counters, category_idx, 
                                  category_name, all_category_names):
    """
    Update detection counters for multi-concept mode.
    Evaluates category pairs where all objects must be detected together.
    
    Each image is:
    - A positive example for its own category pair
    - A negative example for ALL other category pairs
    
    Args:
        detected_objects: List of detected objects
        counters: Dictionary of counter arrays
        category_idx: Index of the current validation category
        category_name: Name of the category (e.g., "Object1_and_Object2")
        all_category_names: List of all category names
    """
    # Process for ALL categories
    for cat_idx, cat_name in enumerate(all_category_names):
        # Determine expected objects for this category pair
        is_multi_object = "_and_" in cat_name
        if is_multi_object:
            expected_objects = cat_name.split("_and_")
        else:
            expected_objects = [cat_name]
        
        # Check if ALL objects from THIS category pair were detected
        all_objects_detected = all(obj in detected_objects for obj in expected_objects)
        
        # Update GROUND TRUTH counters
        if cat_idx == category_idx:
            # This is the POSITIVE case: the image belongs to this category
            counters['gt'][cat_idx] += 1
        else:
            # This is a NEGATIVE case: the image does NOT belong to this category
            counters['gt_neg'][cat_idx] += 1
        
        # Update DETECTION counters (for ALL pairs, check if detected)
        if all_objects_detected:
            # Both/all objects from this pair were detected
            counters['detected'][cat_idx] += 1
        else:
            # Not all objects from this pair were detected
            counters['detected_neg'][cat_idx] += 1
        
        # Update CORRECTNESS counters
        if cat_idx == category_idx:
            # This is the ground truth category
            if all_objects_detected:
                # Correctly detected the positive pair
                counters['correct'][cat_idx] += 1
        else:
            # This is NOT the ground truth category (negative case)
            if not all_objects_detected:
                # Correctly did NOT detect a negative pair
                counters['correct_neg'][cat_idx] += 1


def update_counters(detected_objects, labels, my_objects_raw, counters, args,
                   category_idx=None, category_name=None, all_category_names=None):
    """
    Router function that calls appropriate counter update implementation.
    
    Args:
        detected_objects: List of detected objects
        labels: Ground truth labels
        my_objects_raw: All possible object names
        counters: Dictionary of counter arrays
        args: Configuration arguments
        category_idx: Index of current category (multi-concept only)
        category_name: Name of current category (multi-concept only)
        all_category_names: All category names (multi-concept only)
    """
    if args.multi_concept:
        if category_idx is None or category_name is None or all_category_names is None:
            raise ValueError("Multi-concept mode requires category_idx, category_name, and all_category_names")
        update_counters_multi_concept(detected_objects, counters, category_idx,
                                     category_name, all_category_names)
    else:
        update_counters_single_concept(detected_objects, labels, my_objects_raw, counters)


def calculate_metrics(counters, args):
    """
    Calculate precision and recall metrics.
    
    Args:
        counters: Dictionary of counter arrays
        args: Configuration arguments (needed for test_split handling)
        
    Returns:
        dict: Calculated metrics
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        precision = counters['correct'] / counters['detected']
        precision_neg = counters['correct_neg'] / counters['detected_neg']
        recall = counters['correct'] / counters['gt']
        recall_neg = counters['correct_neg'] / counters['gt_neg']
    
    # For this-is-my dataset with negative or fake splits, invert metrics
    should_invert = (args.dataset == 'this-is-my' and 
                     args.test_split is not None and 
                     args.test_split.lower() in ['negative (other)', 'negative (hard)', 'fake'])
    
    if should_invert:
        precision = 1 - precision
        precision_neg = 1 - precision_neg
        recall = 1 - recall
        recall_neg = 1 - recall_neg
    
    return {
        'precision': precision,
        'precision_neg': precision_neg,
        'recall': recall,
        'recall_neg': recall_neg
    }


def print_metrics(metrics, skipped_count, args):
    """
    Print evaluation metrics.
    
    Args:
        metrics: Dictionary of calculated metrics
        skipped_count: Number of images with no detections
        args: Configuration arguments
    """
    # Add indicator for inverted metrics
    invert_note = ""
    if (args.dataset == 'this-is-my' and 
        args.test_split is not None and 
        args.test_split.lower() in ['negative (other)', 'negative (hard)', 'fake']):
        invert_note = " (INVERTED for negative/fake split)"
    
    print(f'\nNo detections for {skipped_count} images')
    
    print(f'\nAverage Metrics{invert_note}:')
    print(f'  Positive Precision: {metrics["precision"].mean():.4f}')
    print(f'  Negative Precision: {metrics["precision_neg"].mean():.4f}')
    print(f'  Positive Recall: {metrics["recall"].mean():.4f}')
    print(f'  Negative Recall: {metrics["recall_neg"].mean():.4f}')
    
    # Calculate metrics excluding NaN values
    precision_clean = metrics['precision'][~np.isnan(metrics['precision'])]
    precision_neg_clean = metrics['precision_neg'][~np.isnan(metrics['precision_neg'])]
    recall_clean = metrics['recall'][~np.isnan(metrics['recall'])]
    recall_neg_clean = metrics['recall_neg'][~np.isnan(metrics['recall_neg'])]
    
    print(f'\nMetrics (excluding NaN){invert_note}:')
    print(f'  Positive Precision: {precision_clean.mean():.4f}')
    print(f'  Negative Precision: {precision_neg_clean.mean():.4f}')
    print(f'  Positive Recall: {recall_clean.mean():.4f}')
    print(f'  Negative Recall: {recall_neg_clean.mean():.4f}')
    
    # Print balanced accuracy for multi-concept mode
    if args.multi_concept:
        balanced_acc = (metrics['recall'] + metrics['recall_neg']) / 2
        balanced_acc_clean = balanced_acc[~np.isnan(balanced_acc)]
        print(f'\n  Balanced Accuracy: {balanced_acc_clean.mean():.4f}')


def save_results(metrics, args, threshold):
    """
    Save evaluation results to files.
    
    Args:
        metrics: Dictionary of calculated metrics
        args: Configuration arguments
        threshold: Detection threshold used
    """
    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)
    
    mode_str = 'multi' if args.multi_concept else 'single'
    
    # Add test_split to filename for this-is-my dataset
    if args.dataset == 'this-is-my' and args.test_split is not None:
        test_split_str = args.test_split.lower().replace(' ', '_').replace('(', '').replace(')', '')
        base_filename = f'{args.dataset}-{mode_str}-{test_split_str}-{{metric}}-{threshold}-{args.variation}.npy'
    else:
        base_filename = f'{args.dataset}-{mode_str}-{{metric}}-{threshold}-{args.variation}.npy'
    
    np.save(results_dir / base_filename.format(metric='precisions'), metrics['precision'])
    np.save(results_dir / base_filename.format(metric='neg_precisions'), metrics['precision_neg'])
    np.save(results_dir / base_filename.format(metric='recalls'), metrics['recall'])
    np.save(results_dir / base_filename.format(metric='neg_recalls'), metrics['recall_neg'])


def print_per_object_metrics(display_names, metrics, args):
    """
    Print per-object or per-category precision and recall.
    
    Args:
        display_names: List of object/category names to display
        metrics: Dictionary of calculated metrics
        args: Configuration arguments
    """
    # Add indicator for inverted metrics
    invert_note = ""
    if (args.dataset == 'this-is-my' and 
        args.test_split is not None and 
        args.test_split.lower() in ['negative (other)', 'negative (hard)', 'fake']):
        invert_note = " (INVERTED)"
    
    print('\n' + '='*80)
    if args.multi_concept:
        print(f'PER-CATEGORY RESULTS{invert_note}:')
    else:
        print(f'PER-OBJECT RESULTS{invert_note}:')
    print('='*80)
    
    for idx, name in enumerate(display_names):
        if args.multi_concept:
            # Calculate balanced accuracy
            acc = (metrics['recall'][idx] + metrics['recall_neg'][idx]) / 2
            print(f'{name}: '
                  f'Precision {metrics["precision"][idx]:.3f} '
                  f'Recall {metrics["recall"][idx]:.3f} '
                  f'Neg Recall {metrics["recall_neg"][idx]:.3f} '
                  f'ACC {acc:.3f}')
        else:
            print(f'{name}: '
                  f'Precision {metrics["precision"][idx]:.3f} '
                  f'Recall {metrics["recall"][idx]:.3f}')


def print_config(args) -> None:
    """
    Print configuration summary.
    
    Args:
        args: Configuration arguments
    """
    print("\n" + "="*70)
    print("DETECTION EVALUATION CONFIGURATION")
    print("="*70)
    print(f"Mode:              {'Multi-Concept' if args.multi_concept else 'Single-Concept'}")
    print(f"Dataset:           {args.dataset}")
    print(f"Data folder:       {args.data_folder}")
    print(f"Features folder:   {args.features_folder}")
    print(f"Split:             {args.split}")
    if args.dataset == 'this-is-my' and args.test_split is not None:
        print(f"Test Split:        {args.test_split}")
    print(f"Device:            {args.device}")
    print(f"Batch size:        {args.batch_size}")
    print(f"Grounding SAM:     {args.grounding_sam}")
    print(f"Variation:         {args.variation}")
    print(f"N training views:  {args.n_training_views}")
    print("="*70 + "\n")


def get_validation_info(args):
    """
    Get validation folder information based on mode.
    
    Args:
        args: Configuration arguments
        
    Returns:
        tuple: (validation_folders, display_names) or (None, None) for single-concept
    """
    if not args.multi_concept:
        return None, None
    
    # Get validation category folders for multi-concept mode
    if args.dataset == 'rap':
        validation_path = "/fsx/ad/vlm/clean_datasets_CVPR2026/rap/multi-concept/validation/*"
    elif args.dataset == 'this-is-my':
        # MODIFICATION: Use Multi-concept path structure
        validation_path = os.path.join(args.data_folder, "This-is-My-Img/Multi-concept", args.split, '*')
    else:
        validation_path = os.path.join(args.data_folder, args.dataset, args.split, '*')
    
    import glob
    validation_folders = glob.glob(validation_path)
    validation_folders = [f for f in validation_folders if os.path.isdir(f)]
    
    display_names = [f.split('/')[-1] for f in validation_folders]
    
    print(f"\nFound {len(validation_folders)} validation categories:")
    for name in display_names:
        print(f"  - {name}")
    
    return validation_folders, display_names


def get_category_info(result_path, validation_folders):
    """
    Get category index and name from result path.
    
    Args:
        result_path: Path to the result image
        validation_folders: List of validation folder paths
        
    Returns:
        tuple: (category_idx, category_name) or (None, None) if not found
    """
    category_folder = result_path.split('/')[-2]  # "Blippi_and_Blippis shoes"
    
    for idx, val_folder in enumerate(validation_folders):
        val_folder_name = val_folder.split('/')[-1]
        # MODIFICATION: Use exact match instead of substring
        if category_folder == val_folder_name:
            return idx, val_folder_name
    
    # Debug output if no match found
    print(f"Warning: Could not match category '{category_folder}'")
    print(f"Available categories: {[v.split('/')[-1] for v in validation_folders[:3]]}...")
    
    return None, None


# =============================================================================
# Main Execution
# =============================================================================

if __name__ == "__main__":
    print('Detection Evaluation Script Started')
    
    # Parse arguments
    if 'ipykernel' in sys.modules:  # Running in Jupyter
        sys.argv = [sys.argv[0]]
    
    args = setup_args()
    
    # Set dataset-specific min mask area
    if args.dataset in DATASET_MIN_MASK_AREAS:
        args.min_mask_area = DATASET_MIN_MASK_AREAS[args.dataset]
    
    # Configure environment (must be done before importing torch/cuda)
    configure_environment(args)
    
    # Print configuration
    print_config(args)
    
    # Load data
    data_loader, my_objects, context_pool = get_dataloader(args=args)
    my_objects_raw = prepare_object_names(args, my_objects)
    
    # Initialize segmentation model
    mask_extractor = Extractor(args)


    # Right after: mask_extractor = Extractor(args)
    import hashlib
    print("\n" + "="*80)
    print("GROUNDING DINO MODEL CHECK")
    print("="*80)
    print(f"Model type: {type(mask_extractor.g_dino_model)}")
    print(f"Model device: {next(mask_extractor.g_dino_model.parameters()).device}")
    print(f"Model dtype: {next(mask_extractor.g_dino_model.parameters()).dtype}")

    # Check specific model parameters
    model_params = list(mask_extractor.g_dino_model.parameters())
    if len(model_params) > 0:
        first_param = model_params[0]
        print(f"First parameter shape: {first_param.shape}")
        print(f"First parameter mean: {first_param.mean().item():.6f}")
        print(f"First parameter std: {first_param.std().item():.6f}")
        print(f"First parameter hash: {hashlib.md5(first_param.cpu().detach().numpy().tobytes()).hexdigest()[:16]}")

    # Check processor config
    print(f"\nProcessor type: {type(mask_extractor.g_dino_processor)}")

    print("="*80 + "\n")
    
    # Load pre-computed object features
    all_obj_files, all_obj_features = get_objects_features(args, my_objects)
    
    # Setup thresholds
    thresholds = setup_thresholds(args)
    
    # Get validation info based on mode
    validation_folders, display_names = get_validation_info(args)
    
    # Disable visualization during batch processing
    args.show = False
    
    print(f'\n{len(data_loader)} total images to process')
    
    # Iterate over different thresholds
    for threshold in thresholds:
        args.detect_thresh = threshold
        print(f'\n{"="*60}')
        print(f'Evaluating with threshold: {threshold}')
        print(f'{"="*60}')
        
        # Initialize counters based on mode
        if args.multi_concept:
            num_units = len(validation_folders)
        else:
            num_units = len(my_objects)
        
        counters = {
            'detected': np.zeros(num_units),
            'detected_neg': np.zeros(num_units),
            'gt': np.zeros(num_units),
            'gt_neg': np.zeros(num_units),
            'correct': np.zeros(num_units),
            'correct_neg': np.zeros(num_units)
        }
        skipped_count = 0
        
        # Process each batch
        for batch_idx, data_batch in enumerate(data_loader):
            if batch_idx % 50 == 0:
                print(f'Processing batch {batch_idx}/{len(data_loader)}')
            
            _, _, paths, _, *_ = data_batch
            
            # Skip training views if using MyVLM dataset
            #import pdb;pdb.set_trace()
            if args.dataset == 'myvlm' and nested_list_contains(all_obj_files, paths):
                continue
            
            # Process the batch
            batch_results = process_single_image(
                args, data_batch, mask_extractor, all_obj_features, my_objects_raw
            )
            
            # Update counters for each result in batch
            for result in batch_results:
                if not result['personalized_objects']:
                    skipped_count += 1
                
                if args.multi_concept:
                    # Get category information
                    category_idx, category_name = get_category_info(
                        result['path'], validation_folders
                    )
                    
                    if category_idx is None:
                        print(f"Warning: Could not find category for {result['path']}")
                        continue
                    
                    all_category_names = display_names
                    
                    update_counters(
                        result['personalized_objects'],
                        result['labels'],
                        my_objects_raw,
                        counters,
                        args,
                        category_idx=category_idx,
                        category_name=category_name,
                        all_category_names=all_category_names
                    )
                else:
                    update_counters(
                        result['personalized_objects'],
                        result['labels'],
                        my_objects_raw,
                        counters,
                        args
                    )
        
        # Calculate and display metrics
        metrics = calculate_metrics(counters, args)
        print_metrics(metrics, skipped_count, args)
        
        # Print per-object/category metrics
        if args.multi_concept:
            print_per_object_metrics(display_names, metrics, args)
        else:
            print_per_object_metrics(my_objects, metrics, args)
        
        # Save results
        save_results(metrics, args, threshold)
        
        print(f'\nResults saved to results/ directory')