"""
Simplified VQA validation using dataloader.

This script uses the improved dataloader which builds the dataset
directly from the VQA JSON file, ensuring only images with QA pairs are loaded.
"""

print('started')
import os
import json
import sys
import tempfile
from collections import defaultdict
from utils import setup_args, Azure_evaluate, get_cat_folders_from_features, draw_bounding_box, mask_to_normalized_bbox, get_query, get_objects_features, apply_mask_dino, get_mask_for_gt
from PIL import Image
import torch
import glob
import copy
from loader import get_dataloader
from model import Extractor, Personalized_InternVL
import numpy as np
import matplotlib.pyplot as plt
import logging
from datetime import datetime

print('import finished')

# === Setup ===
args = setup_args()

if args.variation == 'normal':
    args.n_augment = 1

colors = ['blue', 'red', 'green', 'purple', 'orange', 'pink', 'gray', 'yellow']

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(message)s',
    datefmt='%H:%M'
)

# === Load dataloader ===
logging.info("Loading dataloader...")
dataloader, my_objects, context_pool = get_dataloader(args)
logging.info(f"Dataloader loaded with {len(dataloader.dataset)} VQA samples")
logging.info(f"Found {len(my_objects)} object categories")

# === Initialize models ===
logging.info("Loading models...")
mask_extractor = Extractor(args)
azure = Azure_evaluate ()




if 'InternVL' in args.vlm_model:
    vlm_model = Personalized_InternVL(args)
else:
    sys.exit(f"{args.vlm_model} is not supported")

# === Load object features ===
logging.info("Loading object features...")
# Get training views from the feature folder
cat_folders = get_cat_folders_from_features(args)
feature_objects = [os.path.basename(folder) for folder in cat_folders]

all_obj_files, all_obj_features = get_objects_features(args, my_objects)
logging.info(f"Features loaded for {len(feature_objects)} objects")

# Debug: Print feature objects
logging.info(f"Feature objects found: {feature_objects}")
for obj in feature_objects:
    logging.info(f"  - '{obj}'")

# Create a mapping from object name to features
object_to_features = {}
for idx, obj_name in enumerate(feature_objects):
    object_to_features[obj_name] = all_obj_features[idx].unsqueeze(0)

# Create name mapping for multi-concept lookups
# Maps "Casey" -> "Casey-man", "Alex" -> "Alex-woman", etc.
name_to_full_object = {}
for obj in feature_objects:
    # Extract base name (before hyphen)
    if '-' in obj:
        base_name = obj.split('-')[0]
        # Handle possessive forms: "Caseys" -> "Casey"
        if base_name.endswith('s') and len(base_name) > 1:
            base_name_without_s = base_name[:-1]
            if base_name_without_s not in name_to_full_object:
                name_to_full_object[base_name_without_s] = obj
        if base_name not in name_to_full_object:
            name_to_full_object[base_name] = obj
    # Also add the full object name without suffix for object lookups
    # e.g., "Caseys boosted board-skateboard" -> also map "Caseys boosted board"
    if '-' in obj:
        base_obj = '-'.join(obj.split('-')[:-1])  # Remove suffix after last hyphen
        if base_obj not in name_to_full_object:
            name_to_full_object[base_obj] = obj

logging.info(f"Name mapping created: {name_to_full_object}")

# === Setup results tracking ===
results_by_object = defaultdict(lambda: {'correct': 0, 'total': 0})
all_correct = 0
all_total = 0

# Setup save path
current_time = datetime.now()
formatted_time = current_time.strftime("%A_%d_%I_%M_%p")
save_path = os.path.join(args.save_folder, args.task, args.vlm_model, formatted_time)
os.makedirs(save_path, exist_ok=True)
logging.info(f"Results will be saved to: {save_path}")

# === Main evaluation loop ===
logging.info("\n" + "=" * 80)
logging.info("Starting VQA evaluation")
logging.info("=" * 80)

for batch_idx, batch in enumerate(dataloader):
    if batch is None:
        continue
    
    # Unpack batch (batch size is 1)
    # Returns: img, img_dino, path, label, question, answer, None, label_map, open_question, full_answer
    imgs, imgs_dino, paths, labels, questions, answers, _, label_maps, open_questions, full_answers = batch
    
    # Get single items from batch
    img_array = imgs[0]  # Already numpy array
    img_dino_array = imgs_dino[0]  # Already numpy array
    img_path = paths[0]
    obj_name = labels[0]
    question = questions[0]
    answer = answers[0]
    label_map = label_maps[0] if label_maps[0] is not None else {}
    
    if question is None or answer is None:
        logging.warning(f"Skipping {img_path} - missing QA data")
        continue
    
    # Parse multi-concept object names
    # Multi-concept format: "Person_Persons object" (underscore separator)
    # Single concept format: "Object-type" (hyphen separator, no underscore)
    parts = obj_name.split('_', 1)
    
    if len(parts) == 2:
        # Multi-concept case: split into individual objects
        object_names = parts  # e.g., ["Casey", "Caseys boosted board"]
    else:
        # Single concept case
        object_names = [obj_name]  # e.g., ["Alexs everyday bag-bag"]
    
    logging.info(f"\n{'=' * 80}")
    logging.info(f"Processing [{batch_idx + 1}/{len(dataloader)}]")
    logging.info(f"Original label: {obj_name}")
    logging.info(f"Objects to detect: {object_names}")
    logging.info(f"Image: {img_path}")
    
    # Get features for ALL objects
    obj_features_list = []
    valid_objects = []
    valid_object_display_names = []  # For display purposes (with label_map applied)
    
    for obj in object_names:
        # Try direct match first
        matched_obj = obj
        
        # If not found, try to map base name to full name
        if obj not in object_to_features:
            # Try name mapping (e.g., "Casey" -> "Casey-man")
            if obj in name_to_full_object:
                matched_obj = name_to_full_object[obj]
                logging.info(f"  Mapped '{obj}' -> '{matched_obj}'")
            else:
                logging.warning(f"  Object '{obj}' not found in feature list. Skipping this object.")
                continue
        
        if matched_obj not in object_to_features:
            logging.warning(f"  Object '{matched_obj}' not found in feature list. Skipping this object.")
            continue
            
        obj_features_list.append(object_to_features[matched_obj])
        valid_objects.append(matched_obj)
        
        # Get display name (apply label_map if available)
        display_name = label_map.get(obj, obj) if label_map else obj
        valid_object_display_names.append(display_name)
    
    if len(obj_features_list) == 0:
        logging.warning(f"  No valid objects found. Skipping entire sample.")
        continue
    
    logging.info(f"  Found {len(valid_objects)} valid objects: {valid_objects}")
    
    # Concatenate all object features
    obj_features = torch.cat(obj_features_list, dim=0)
    
    # Convert numpy back to PIL for processing
    img_raw = Image.fromarray(img_array.astype(np.uint8))
    
    # Prepare image batches for DINO and SAM
    dino_image_batch = [img_dino_array]
    sam_image_batch = [img_dino_array]
    
    # Get DinoV2 features
    dino_img_features = mask_extractor.forward_dino(dino_image_batch)
    
    # Get SAM masks
    sam_masks = mask_extractor.forward_grounding_dino(sam_image_batch, ['object.'])
    
    # Get the mean Dino features for each mask
    category_features = apply_mask_dino(args, sam_masks, dino_img_features[0])
    category_features = torch.stack([x.mean(dim=0) for x in category_features])
    
    # Find best matching masks for ALL objects
    bboxes = []
    bbox_img = copy.deepcopy(img_raw)
    
    for idx, obj in enumerate(valid_objects):
        obj_feat = obj_features_list[idx]
        final_mask, max_score = get_mask_for_gt(args, category_features, sam_masks[0], obj_feat)
        
        bbox, bbox_raw = mask_to_normalized_bbox(final_mask)
        bboxes.append(bbox)
        
        # Draw bounding box with different color for each object
        bbox_img = draw_bounding_box(bbox_img, [bbox], [colors[idx]])
        logging.info(f"  Detected '{valid_objects[idx]}' with score {max_score:.3f}")
    
    # Generate prompt with ALL detected objects
    # Use display names for the prompt
    prompts = get_query(args, valid_object_display_names, colors[:len(valid_objects)], question, None, None)
    logging.info(f"Question: {prompts}")

    
    # Save bbox image to temporary file
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
        tmp_path = tmp_file.name
        bbox_img.save(tmp_path)
    
    try:
        #import pdb;pdb.set_trace()
        response = vlm_model(tmp_path, [prompts])
    finally:
        # Clean up temporary file
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
    
    # Parse answer - check if multiple choice or open-ended
    if 'A.' in question or 'B.' in question:
        # Multiple choice - extract letter
        predicted_answer = None
        response_lower = response.lower().strip()
        
        # Try to extract answer letter
        if response_lower.startswith('a'):
            predicted_answer = 'A'
        elif response_lower.startswith('b'):
            predicted_answer = 'B'
        elif 'a.' in response_lower or 'a)' in response_lower:
            predicted_answer = 'A'
        elif 'b.' in response_lower or 'b)' in response_lower:
            predicted_answer = 'B'
        
        # Extract correct answer letter from answer string (e.g., "A.option text")
        correct_letter = answer[0] if answer and len(answer) > 0 else None
        is_correct = (predicted_answer == correct_letter)
    else:
        # Open-ended - simple string matching
        llm_answer = azure.evaluate_answer(question, response, answer)
        #print('LLm Answer:',llm_answer)
        if 'yes' in llm_answer.lower():
            is_correct = True
        elif 'no' in llm_answer.lower():
            is_correct = False
    
    correctness = "✓ CORRECT" if is_correct else "✗ INCORRECT"
    logging.info(f"Ground Truth: {answer}")
    logging.info(f"Prediction: {response}")
    logging.info(f"Result: {correctness}")
    
    # Update counters (use original obj_name for tracking)
    results_by_object[obj_name]['correct'] += int(is_correct)
    results_by_object[obj_name]['total'] += 1
    all_correct += int(is_correct)
    all_total += 1
    
    # Save detailed results
    with open(os.path.join(save_path, 'results_vqa.txt'), 'a') as file:
        file.write('#' * 80 + '\n')
        file.write(f'IMG Path: {img_path}\n')
        file.write(f'Objects: {obj_name} -> {valid_objects}\n')
        file.write(f'Question: {question}\n')
        file.write(f'Ground Truth: {answer}\n')
        file.write(f'Prediction: {response}\n')
        file.write(f'Result: {correctness}\n')
    
    if not is_correct and hasattr(args, 'show') and args.show:
        plt.imshow(bbox_img)
        plt.title(f"Q: {question}\nGT: {answer}\nPred: {response}")
        plt.show()

# === Aggregate results ===
logging.info("\n" + "=" * 80)
logging.info("PER-OBJECT RESULTS")
logging.info("=" * 80)

per_object = {}
for obj, res in results_by_object.items():
    acc = res['correct'] / res['total'] * 100 if res['total'] > 0 else 0
    logging.info(f"{obj:40s}: {res['correct']:3d}/{res['total']:3d} = {acc:6.2f}%")
    per_object[obj] = {'correct': res['correct'], 'total': res['total'], 'accuracy': acc}

overall_acc = all_correct / all_total * 100 if all_total > 0 else 0
logging.info(f"\nOVERALL ACCURACY: {all_correct}/{all_total} = {overall_acc:.2f}%")

# === Save results ===
results_dict = {
    'per_object': per_object,
    'overall': {'correct': all_correct, 'total': all_total, 'accuracy': overall_acc}
}

results_json_path = os.path.join(save_path, 'results_summary.json')
with open(results_json_path, 'w') as f:
    json.dump(results_dict, f, indent=2)

logging.info(f"\nResults saved to {results_json_path}")

# Also save in the original format for compatibility
with open(os.path.join(save_path, 'results_vqa.txt'), 'a') as file:
    file.write('\n' + '=' * 80 + '\n')
    file.write('SUMMARY\n')
    file.write('=' * 80 + '\n')
    file.write(f'Total Images: {all_total}\n')
    file.write(f'Correct Images: {all_correct}\n')
    file.write(f'Accuracy: {overall_acc:.2f}%\n')

logging.info("Evaluation complete!")