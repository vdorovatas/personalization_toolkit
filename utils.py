import numpy as np
from PIL import Image, ImageDraw
import pickle
import copy
import random
import argparse
import logging
import os
import glob
import torch
from torch.nn.utils.rnn import pad_sequence
from sklearn.cluster import AgglomerativeClustering
import concurrent.futures
import time
import matplotlib.pyplot as plt
import openai
from langchain_openai import ChatOpenAI
py_logger = logging.getLogger(__name__)
py_logger.setLevel(logging.DEBUG)
# Create input parser, parse arguments and do final configurations
def setup_args():
    parser = argparse.ArgumentParser('Argparse options for feature extraction pipeline')
    parser.add_argument('--dataset', default='MyVLM', help='Dataset to process')
    parser.add_argument('--data_folder', type=str, default='/data/datasets/zeroshot-seg/personalization/', help='Main dataset storage directory')
    parser.add_argument('--save_folder', type=str, default='results/', help='Main dataset storage directory')
    parser.add_argument('--features_folder', type=str, default='/data/datasets/zeroshot-seg/personalization/', help='Main dataset storage directory')
    parser.add_argument('--device', type=str, default='cuda:0',help='Device to process')
    parser.add_argument('--device_ids', default=[0,1,2,3,4,5,6,7],help='Visible Devices to process')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--split', default='train', help='Data Split')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of dataloder workers')
    parser.add_argument('--dino_img_size', type=int, default=518, help='Image size')
    parser.add_argument('--dino_patch_size', type=int, default=14, help='Transformers Patch Size')
    parser.add_argument('--dino_embedding_size', type=int, default=1024, help='Transformers embedding Size')
    parser.add_argument('--shuffle', action='store_true', help='Shuffle the dataloader indices')
    parser.add_argument('--show', action='store_true', help='Show image and corresponding labels')
    parser.add_argument('--vlm_model', default="LLaVA", help='Which VLM model to use.')
    parser.add_argument('--training_views_path', default=None, help='Where to get training features for each personalized object.')
    parser.add_argument('--n_training_views', default=4, help='Number of training views for each personalized object.')
    parser.add_argument('--detect_thresh', default=0.75, help='Detection threshold for personalized objects.')
    parser.add_argument('--clip_vit_model', default="openai/clip-vit-large-patch14-336", help='Which CLIP model to load from HF.')
    parser.add_argument('--clip_img_size', type=int, default=1344, help='Clip image size for training view extraction.')
    parser.add_argument('--min_mask_area', type=int, default=9000, help='Ignore SAM masks smaller than this size in pixels.')
    parser.add_argument('--box_overlap_thresh', type=float, default=80, help='Ignore SAM masks smaller than this size in pixels.')
    parser.add_argument('--variation',  default='normal', help='which training views to use: normal, augmented or 1.1')
    parser.add_argument('--category_specific_thresh',  default=False, help='Do we set a fixed threshold for all objects or set t specifically ?')
    parser.add_argument('--n_augment', type=int, default=9, help='Number of augmentations per training view.')
    parser.add_argument('--inverse_accuracy',  default=False, help='Evaluate on Images without the personalized objects (this_is_my)?')
    parser.add_argument('--gpt_fake',  default=False, help='Use GPT generated images for evaluation (this_is_my)?')
    parser.add_argument('--task',  default='detection', help='What is the task to be solved ?')
    parser.add_argument('--grounding_sam',  default=True, help='Wheter to use grounding sam for mask extraction ?')
    args, _ = parser.parse_known_args()
    args.num_crops=(args.clip_img_size//336)**2
    args.num_patches=args.clip_img_size//args.dino_patch_size
    py_logger.debug("Parsed input arguments and customized parameters")
    return args

def draw_bounding_box(image, bboxes, colors):
    """
    Draws a bounding box on the image.

    :param image_path: Path to the input image.
    :param box_coords: Tuple of bounding box coordinates (x_min, y_min, x_max, y_max) normalized between 0 and 1.
    """
    # Load the image
    #image = Image.fromarray(img.astype(np.uint8))
    #import pdb;pdb.set_trace()
    for box_coords, color in zip(bboxes,colors):
        draw = ImageDraw.Draw(image)

        # Image dimensions
        width, height = image.size

        # Normalize coordinates
        x_min = int(box_coords[0] * width)
        y_min = int(box_coords[1] * height)
        x_max = int(box_coords[2] * width)
        y_max = int(box_coords[3] * height)

        # Define the bounding box
        bounding_box = [(x_min, y_min), (x_max, y_max)]

        # Draw the bounding box
        draw.rectangle(bounding_box, outline=color, width=10)
    return image

def mask_to_normalized_bbox(mask):
    """
    Converts a binary mask to a bounding box with coordinates normalized to the [0, 1] range.

    Args:
        mask (np.ndarray): A binary mask of shape (H, W) with 0s and 1s.

    Returns:
        bbox (tuple): A tuple (x_min, y_min, x_max, y_max) with normalized bounding box coordinates.
    """
    # Find non-zero indices (locations of 1s in the mask)
    indices = np.argwhere(mask)

    if indices.size == 0:
        # If the mask is empty (no 1s), return None or an invalid box
        return None

    # Extract y and x coordinates
    y_coords = indices[:, 0]
    x_coords = indices[:, 1]

    # Calculate the bounding box
    x_min = np.min(x_coords)
    x_max = np.max(x_coords)
    y_min = np.min(y_coords)
    y_max = np.max(y_coords)

    # Normalize the coordinates to the [0, 1] range
    height, width = mask.shape
    x_min_norm = x_min / width
    x_max_norm = x_max / width
    y_min_norm = y_min / height
    y_max_norm = y_max / height

    # Return the bounding box as (x_min, y_min, x_max, y_max)
    return (x_min_norm, y_min_norm, x_max_norm, y_max_norm), (x_min, y_min, x_max, y_max)

def nested_list_contains(nl, target):
    # Extract the base name from target (e.g., "running_shoes_14" from the first example)
    target_base = target[0].split('/')[-1].split('.')[0]
    
    for thing in nl:
        if type(thing) is list:
            if nested_list_contains(thing, target):
                return True
        elif isinstance(thing, str):
            # Extract the base name from thing
            thing_base = thing.split('/')[-1].split('.')[0]
            
            # Check if target_base is a prefix of thing_base (with underscore separator)
            # This ensures running_shoes_14 matches running_shoes_14_1 
            # but NOT running_shoes_1 or running_shoes_1_anything
            if thing_base == target_base or (
                thing_base.startswith(target_base + '_') and 
                len(thing_base) > len(target_base) + 1
            ):
                return True
    return False

def gather_obj_files(obj, args,features_path):
    if args.dataset == 'myvlm':
        obj_files = glob.glob(features_path + "/{}/*.pt".format(obj.replace(' ', '_')), recursive=True)
        random.shuffle(obj_files)
        return obj_files[:args.n_training_views]
    elif args.dataset == 'yollava':
        obj_files = glob.glob(features_path + "/{}/*.pt".format(obj), recursive=True)
        random.shuffle(obj_files)
        return obj_files
    return []

def process_file(o, args,features_path):
    obj_features = []
    for file_path in o:
        instance = torch.load(file_path, pickle_module=pickle)
        if args.dataset == 'MyVLM':
            instance_features = copy.deepcopy(instance['dino_obj_features'][0])
        elif args.dataset == 'yollava':
            try:
                instance_features = torch.stack(instance['dino_obj_features']).squeeze().reshape([-1, 1024])
            except:
                instance_features = copy.deepcopy(instance['dino_obj_features'])
        obj_features.append(instance_features.mean(dim=0))
    return torch.stack(obj_features)

def filter_paths_by_strings(file_list, search_string_a, search_string_b):
    # Filter the file list to include only paths that contain either search_string_a or search_string_b
    matching_files = [
        file_path for file_path in file_list 
        if search_string_a in file_path or search_string_b in file_path
    ]
    return matching_files

def get_objects_features(args, my_objects): #HERE
    #get the view for each object
    #Run the segmentation network on the views and get the mean
    #get the seg masks for the prsonalized objects
    # TODO changes addreses
    all_obj_files=[]
    all_obj_features=[]
    #change features path to args ??
    features_path = os.path.join(args.features_folder,args.dataset, "extracted_features")
    if args.grounding_sam:
        features_path = os.path.join(features_path,"gsam")
    if args.multi_concept:
        features_path=os.path.join(args.features_folder,args.dataset.lower(),'multi_concept','extracted_features/')
    

    #######temporary Vaggelis extractions
    #if args.dataset=='myvlm':
    #    features_path = "/fsx/ad/vlm/myvlm/vdoro_extracted_features/grounding_sam/1.2-augmented/"
    #if args.dataset=='yollava':
    #    features_path = "/fsx/ad/vlm/yollava/vdoro_extracted_features/grounding_sam_face/labelled-1.2-augmented/"
    #if args.dataset=='this-is-my':
    #    features_path="/fsx/ad/vlm/this-is-my/vdoro_extracted_features/grounding_sam/1.2-augmented/"
    ##########################
    
    #For myVLM there is a need to get 4 views as training and gather all augmentations for those views
    for obj in my_objects:
        if args.dataset=="myvlm":
            obj = obj.replace(' ','_')
        elif args.dataset=='yollava':
            obj = substring_before_last_dash(obj)
        #import pdb;pdb.set_trace()
        obj_files=glob.glob(features_path+"/{}/*.pt".format(obj),recursive=True)
        random.shuffle(obj_files)
        if args.n_training_views is not None:
            obj_files = obj_files[:args.n_training_views]
        all_obj_files.append(obj_files)
    #import pdb;pdb.set_trace()
    for o_num,o in enumerate(all_obj_files):
        #print(o_num)
        obj_features=[]
        for i in range(len(o)):
            instance=torch.load(o[i],pickle_module=pickle)
            if args.dataset=='myvlm':
                instance_features=copy.deepcopy(instance['dino_obj_features'][0]).reshape([-1,1024])
            elif args.dataset=='yollava' or args.dataset=='this-is-my':
                #import pdb;pdb.set_trace()
                try:
                    instance_features=torch.stack(instance['dino_obj_features']).squeeze().reshape([-1,1024])
                except:
                    instance_features=copy.deepcopy(instance['dino_obj_features'])
            obj_features.append(instance_features.mean(dim=0))
        obj_features=torch.stack(obj_features)
        #import pdb;pdb.set_trace()
        all_obj_features.append(obj_features)
    
    if args.dataset=='myvlm':
        all_obj_features_final=torch.stack(all_obj_features).cuda()
    elif args.dataset=='yollava' or args.dataset=='this-is-my':
        all_obj_features_final = pad_sequence(all_obj_features, batch_first=True).cuda()
    
    return all_obj_files,all_obj_features_final

# Filter features using input mask
def apply_mask_dino(args, sam_mask, dino_features):
    sam_mask=torch.from_numpy(np.array(sam_mask))
    sam_mask = sam_mask.float().reshape([-1,1,args.dino_img_size,args.dino_img_size])
    #start_time=time.time()
    sam_dino_mask = torch.nn.functional.interpolate(sam_mask,[args.dino_img_size//args.dino_patch_size,
                                                                   args.dino_img_size//args.dino_patch_size]).squeeze(1)
    dino_features_cropped = []
    py_logger.debug("Applied sam mask on dino features")
    for b in range(sam_dino_mask.shape[0]):
        feat=dino_features[sam_dino_mask[b].type(torch.bool)].reshape(-1, args.dino_embedding_size)
        dino_features_cropped.append(feat)
    return dino_features_cropped

def apply_mask_clip(args, sam_mask, clip_features):
    sam_mask=torch.from_numpy(np.array(sam_mask))
    sam_mask = sam_mask.float().reshape([-1,1,args.dino_img_size,args.dino_img_size])
    sam_clip_mask=torch.nn.functional.interpolate(sam_mask,[args.clip_img_size//args.dino_patch_size,
                                                                   args.clip_img_size//args.dino_patch_size]).squeeze(1)
    clip_features_cropped=[]
    for b in range(sam_clip_mask.shape[0]):
        feat_clip=clip_features[sam_clip_mask[b].type(torch.bool)].reshape(-1, 768)
        clip_features_cropped.append(feat_clip)
    return clip_features_cropped


def apply_mask_sam(args, sam_mask, sam_features):
    sam_img_size = 1024
    sam_patch_size = 16
    sam_embedding_size = 256
    sam_mask=torch.from_numpy(np.array(sam_mask))
    sam_mask = sam_mask.float().reshape([-1,1,args.dino_img_size,args.dino_img_size])
    sam_sam_mask = torch.nn.functional.interpolate(sam_mask,[sam_img_size//sam_patch_size,
                                                                    sam_img_size//sam_patch_size]).squeeze(1)
    sam_features_cropped = []
    py_logger.debug("Applied sam mask on SAM features")
    for b in range(sam_sam_mask.shape[0]):
        feat=sam_features[sam_sam_mask[b].type(torch.bool)].reshape(-1, sam_embedding_size)
        sam_features_cropped.append(feat)
    return sam_features_cropped

def gather_masks(args,category_features, sam_masks, all_obj_features):
    already_detected=np.zeros(len(all_obj_features))
    already_detected_mask_indice=(np.ones(len(all_obj_features))*(-1)).astype(int)
    similarity_scores=[]
    full_masks=[]
    full_masks_labels=[]
    full_masks_counter=0
    something_detected=False
    for m_number,mask_feature in enumerate(category_features):
        sim=torch.nn.functional.cosine_similarity(mask_feature.to(args.device), all_obj_features.to(args.device), dim=2, eps=1e-8)
        #import pdb;pdb.set_trace()
        sim[torch.isnan(sim)==True]=0
        #if args.category_specific_thresh:
        #    sim_detect=(sim>args.category_thresh).float().mean(dim=1)
        #else:
        sim_detect=(sim>args.detect_thresh).float().mean(dim=1)
        detections=(sim_detect>0)
        detected_indice=torch.where(detections==True)[0]
        if len(detected_indice)>0:
            something_detected=True
            d_indice=torch.unravel_index(torch.argmax(sim, axis=None), sim.shape)[0]
            if already_detected[d_indice]==1:
                assert already_detected_mask_indice[d_indice]!=-1
                full_masks[already_detected_mask_indice[d_indice]][sam_masks[m_number]]=0
                full_masks[already_detected_mask_indice[d_indice]]=full_masks[already_detected_mask_indice[d_indice]]+sam_masks[m_number]
                #max or average with previous values ?
                similarity_scores[already_detected_mask_indice[d_indice]].append(torch.max(sim[d_indice]))
            else:
                full_masks.append(sam_masks[m_number])
                full_masks_labels.append(d_indice)
                already_detected[d_indice]=1
                already_detected_mask_indice[d_indice]=int(full_masks_counter)
                full_masks_counter=full_masks_counter+1
                similarity_scores.append([torch.max(sim[d_indice])])
    if something_detected==True:
        full_masks=np.stack(full_masks)
        full_masks_labels=torch.stack(full_masks_labels)
    #import pdb;pdb.set_trace()  
    similarity_scores=[torch.stack(x).mean() for x in similarity_scores]
    return full_masks, full_masks_labels, something_detected, similarity_scores




def gather_masks2(args,category_features, sam_masks, all_obj_features):
    sim_detected=np.zeros(len(all_obj_features))
    already_detected=np.zeros(len(all_obj_features))
    already_detected_mask_indice=(np.ones(len(all_obj_features))*(-1)).astype(int)
    similarity_scores=[]
    full_masks=[]
    full_masks_labels=[]
    full_masks_counter=0
    something_detected=False
    for m_number,mask_feature in enumerate(category_features):
        #Check similarity and threshold
        #plt.imshow(sam_masks[m_number])
        #plt.show()
        sim=torch.nn.functional.cosine_similarity(mask_feature.to(args.device), all_obj_features.to(args.device), dim=2, eps=1e-8)
        sim[torch.isnan(sim)==True]=0
        #print(sim)
        #import pdb;pdb.set_trace()
        if args.category_specific_thresh:
            sim_detect=(sim>args.category_thresh).float().mean(dim=1)
        else:
            sim_detect=(sim>args.detect_thresh).float().mean(dim=1)
        detections=(sim_detect>0)
        detected_indice=torch.where(detections==True)[0]
        if len(detected_indice)>0:
            #import pdb;pdb.set_trace()
            d_indice=torch.unravel_index(torch.argmax(sim, axis=None), sim.shape)[0]
            if already_detected[d_indice]==1:
                #import pdb;pdb.set_trace()
                #print(torch.max(sim[d_indice]),similarity_scores[already_detected_mask_indice[d_indice]])
                if torch.max(sim[d_indice]) > similarity_scores[already_detected_mask_indice[d_indice]][0]:
                    #print(similarity_scores[already_detected_mask_indice[d_indice]])
                    assert already_detected_mask_indice[d_indice]!=-1
                    #full_masks[already_detected_mask_indice[d_indice]][sam_masks[m_number]]=0
                    full_masks[already_detected_mask_indice[d_indice]]=copy.deepcopy(sam_masks[m_number])
                    something_detected=True
                    #max or average with previous values ?
                    similarity_scores[already_detected_mask_indice[d_indice]]=[torch.max(sim[d_indice])]
            else:
                something_detected=True
                full_masks.append(sam_masks[m_number])
                full_masks_labels.append(d_indice)
                already_detected[d_indice]=1
                already_detected_mask_indice[d_indice]=int(full_masks_counter)
                full_masks_counter=full_masks_counter+1
                similarity_scores.append([torch.max(sim[d_indice])])
    if something_detected==True:
        full_masks=np.stack(full_masks)
        full_masks_labels=torch.stack(full_masks_labels)
    #import pdb;pdb.set_trace()  
    #similarity_scores=[x.mean() for x in similarity_scores]
    return full_masks, full_masks_labels, something_detected, similarity_scores






def get_mask_for_gt(args,category_features, sam_masks, obj_features):
    max_score=0
    for m_number,mask_feature in enumerate(category_features):
        #Check similarity and threshold
        sim=torch.nn.functional.cosine_similarity(mask_feature.to(args.device), obj_features.to(args.device), dim=2, eps=1e-8)
        sim_detect=(sim>max_score).float().mean(dim=1)
        detections=(sim_detect>0)
        detected=torch.where(detections==True)[0]
        if len(detected)>0:
            final_mask=copy.deepcopy(sam_masks[m_number])
            max_score=torch.max(sim)
    return final_mask,max_score


# Find corresponding features path using relative path between directories
def _find_corresponding_features_path(data_folder, features_folder, data_path, features_ext=".pt"):
    rel_path = data_path.replace(data_folder, '')
    rel_path = os.path.splitext(rel_path)[0]
    features_path = os.path.join(features_folder, rel_path + features_ext)
    return features_path

# Save sample features file
def save_sample_features(args, features,aug_index=0):
    for b in range(args.batch_size):
        #features_path = _find_corresponding_features_path(args.data_folder, args.features_folder, features["img_path"][b])
        features_path=os.path.join(args.features_folder,args.dataset.lower(),'extracted_features/')
        if args.grounding_sam:
            features_path=os.path.join(features_path,'gsam/')
        if args.multi_concept:
            features_path=os.path.join(args.features_folder,args.dataset.lower(),'multi_concept','extracted_features/')
        features_path=features_path+features["img_path"][b].split('/')[-2]+'/'+features["img_path"][b].split('/')[-1].split('.')[0]+f'_{aug_index}.pt'
        features["feat_path"] .append(features_path)
        features_dir = os.path.dirname(features_path)
        os.makedirs(features_dir, exist_ok=True)
        this_features = {"npy_image":None,
            "class_label":None,
            "img_path":None,
            "feat_path":[],
            "class_token":None,
            "clip_patch_features":None,
            "dino_img_features":None,
            "dino_obj_features":[],
            "sam_mask":[],
            }
        
        # Convert to CPU for future loading
        for key in features:
            #import pdb;pdb.set_trace()
            this_features[key]=copy.deepcopy(features[key][b])
            if type(this_features[key]) == torch.Tensor:
                this_features[key] = this_features[key].cpu()
            elif type(this_features[key]) == list:
                for id, _ in enumerate(this_features[key]):
                    if type(this_features[key][id]) == torch.Tensor:
                        this_features[key][id]= this_features[key][id].cpu()
        print(f"Features file: {this_features['feat_path']}")
        torch.save(this_features,
                   this_features["feat_path"],
                   pickle_module=pickle,
                   pickle_protocol=torch.serialization.DEFAULT_PROTOCOL,
                   _use_new_zipfile_serialization=True)
        
        if (os.path.exists(this_features["feat_path"]) and os.path.isfile(this_features["feat_path"]))==False:
            py_logger.error("Unable to save features file {}".format(this_features["feat_path"]))
            raise AssertionError
    return features

def check_save(args,img_path):
    if args.dataset=='yollava':
        return True
    save_path=_find_corresponding_features_path(args.data_folder, args.features_folder, img_path)
    print(f"Save path: {save_path}")
    directory_path=os.path.dirname(save_path)
    file_count = sum([len(files) for _, _, files in os.walk(directory_path)])
    #import pdb;pdb.set_trace()
    if file_count<args.n_training_views:
        return True
    return False

def calculate_inclusion_percentage(box_a, box_b):
    x1_A, y1_A, x2_A, y2_A = box_a
    x1_B, y1_B, x2_B, y2_B = box_b

    # Calculate the intersection coordinates
    x1_inter = max(x1_A, x1_B)
    y1_inter = max(y1_A, y1_B)
    x2_inter = min(x2_A, x2_B)
    y2_inter = min(y2_A, y2_B)

    # Check if there is an intersection
    if x1_inter < x2_inter and y1_inter < y2_inter:
        area_inter = (x2_inter - x1_inter) * (y2_inter - y1_inter)
    else:
        area_inter = 0

    # Calculate the areas of the boxes
    area_A = (x2_A - x1_A) * (y2_A - y1_A)
    area_B = (x2_B - x1_B) * (y2_B - y1_B)

    # Calculate inclusion percentages
    inclusion_A = (area_inter / area_A * 100) if area_A > 0 else 0
    inclusion_B = (area_inter / area_B * 100) if area_B > 0 else 0

    # Return the maximum inclusion percentage
    return max(inclusion_A, inclusion_B)

def check_bbox(args,bbox,bboxes,box_scores,box_score,personalized_objects,candidate_object,candidate_context=None,contexts=None):
    for b,old_bbox in enumerate(bboxes):
        percentage=calculate_inclusion_percentage(bbox, old_bbox)
        #print(percentage)
        if percentage>args.box_overlap_thresh:
            import pdb;pdb.set_trace()
            if box_scores[b]<box_score:
                bboxes[b]=bbox
                box_scores[b]=box_score
                personalized_objects[b]=candidate_object
                if contexts!=None:
                    contexts[b]=candidate_context
            return personalized_objects,contexts,bboxes, box_scores
    bboxes.append(bbox)
    box_scores.append(box_score)
    personalized_objects.append(candidate_object)
    if contexts!=None:
        contexts.append(candidate_context)
    return personalized_objects,contexts,bboxes,box_scores


def check_bbox2(args,bbox,bboxes,box_scores,box_score,personalized_objects,candidate_object,candidate_context=None,contexts=None):
    for b,old_bbox in enumerate(bboxes):
        percentage=calculate_inclusion_percentage(bbox, old_bbox)
        #print(percentage)
        if percentage>args.box_overlap_thresh:
            #import pdb;pdb.set_trace()
            try:
                if box_scores[b]<box_score[0]:
                    bboxes[b]=bbox
                    box_scores[b]=box_score[0]
                    personalized_objects[b]=candidate_object
                    if contexts!=None:
                        contexts[b]=candidate_context
                return personalized_objects,contexts,bboxes, box_scores
            except:
                import pdb;pdb.set_trace()
    bboxes.append(bbox)
    try:
        box_scores.append(box_score[0])
    except:
        box_scores.append(box_score)
    personalized_objects.append(candidate_object)
    if contexts!=None:
        contexts.append(candidate_context)
    return personalized_objects,contexts,bboxes,box_scores
def substring_before_last_dash(s):
    # Find the index of the last '-'
    index = s.rfind('-')
    
    # If '-' is found, return the substring before it
    if index != -1:
        return s[:index]
    else:
        return s  # Return the original string if '-' is not found
def get_random_object(gt_object,my_objects_raw):
    result = random.choice(['Heads', 'Tails'])
    if result=='Heads':
        obj=copy.deepcopy(gt_object[0])
    else:
        obj_index= random.randint(0, len(my_objects_raw)-1)
        obj=my_objects_raw[obj_index]
    return obj



def build_question_prompt(
    entity_name: str,
    color,
    base: bool=False,
    forbidden_hint: str = 'the box and its color',
    question: str = 'What is the primary material visible on the exterior of {entity}?',
    options: tuple[str, str] = ('A. Glass', 'B. Brick'),
) -> str:
    """
    Returns a single, valid string for HumanMessage.content.
    - Ensures consistent casing and instructions.
    - Avoids any role markers like 'ASSISTANT:'.
    """
    entity_disp = entity_name
    # Normalize a lowercase alias if the question might reference it in lowercase
    entity_alias = entity_name.lower()

    prompt = (
        f'In this image, the entity enclosed in a {color} box is called "{entity_disp}". '
        f'Never mention {forbidden_hint}. '
        f'Answer the following question about "{entity_disp}".\n\n'
        f'Question: {question}\n'
        f'Answer with a single letter (A or B) and explain you answer.'
    )
    if base==True:
        prompt = (
            f'This is an image of "{entity_disp}". '
            f'Answer the following question about "{entity_disp}" in the foloowing image.\n\n'
            f'Question: {question}\n'
            f'Answer with a single letter (A or B) and explain you answer.'
        )
    return prompt

def get_query(args,label,color,question=None,context=None, label_map=None, full_question=None):
    if label_map!=None:
        label=label_map[label]
    if full_question!=None:
        prompts=['USER: <image>\nIn this image, the entity enclosed in a {} box is called \"{}\". Never mention the box and its color and answer the following question about \"{}\" : {} .\nASSISTANT:'.format(color,label.upper(),label.upper(),full_question)]
    elif question==None or question=='':
        if args.task=='clip_score':
            #prompts=['USER: <image>\nIn this image, the entity enclosed in a {} box is called \"{}\". Never mention the box and its color and caption the image and make sure to include the word \"{}\" in the caption.\nASSISTANT:'.format(color,label.upper(),label.upper())]
            prompts=['USER: <image>\nIn this image, the entity enclosed in a {} box is called \"{}\". Describe what \"{}\" is doing using their given name. Describe the image too.\nASSISTANT:'.format(color,label.capitalize(),label.capitalize(),label.capitalize())]
        elif args.task=='comparison_yollava':
            if len(labels)>1:
                #prompts = ['USER: <image>\nIn this image, the entities enclosed in {} boxes are called "{}" respectively. Describe what "{}" are doing using their given names. Describe the image too.\nASSISTANT:'.format(
                #    ", ".join(colors), ", ".join(label.capitalize() for label in labels), ", ".join(label.capitalize() for label in labels)
                #)]
                #prompts = ['USER: <image>\n' +"\n".join(f"In this image, the entity enclosed in a {color} box is called \"{label.capitalize()}\"" for color, label in zip(colors, labels)) + '\nDo not mention any of the bounding boxes or their colors.' +'\nDescribe what "{}" are doing using their given names.'.format(", ".join(label.capitalize() for label in labels)) +"\nDescribe the image too.\nASSISTANT:"]
                prompts = ['USER: <image>\n' +"\n".join(f"In this image, the entity enclosed in a {color} box is called \"{label.capitalize()}\"" for color, label in zip(colors, labels)) + '\nDo not mention any of the bounding boxes or their colors.' +'\nDescribe what "{}" are doing using their given names."\nASSISTANT:"'.format(", ".join(label.capitalize() for label in labels))]

                
            else:
                prompts = ['USER: <image>\nIn this image, the entity enclosed in {} box is called "{}".  Do not mention any of the bounding boxes or their colors. Describe what "{}" is doing using its given name.\nASSISTANT:'.format(
                    ", ".join(colors), ", ".join(label.capitalize() for label in labels), ", ".join(label.capitalize() for label in labels)
                )]
        else:
            prompts=['USER: <image>\n Describe the contents of the {} bounding box.\nASSISTANT:'.format(color)]

            #prompts=['USER: <image>\nIn this image, the entity enclosed in a {} box is called \"{}\". Describe what \"{}\" is doing using its given name. Describe the image too.\nASSISTANT:'.format(color,label.capitalize(),label.capitalize(),label.capitalize())]
        #    prompts=['USER: <image>\nIn this image, there is a {} box. Please refer to the entity inside the {} box as \"{}\". Generate a caption describing the image, make sure to use \"{}\" when mentioning the entity in the {} box.\nASSISTANT:'.format(color,color,label.capitalize(),label.capitalize(),label.capitalize(),color)]
    elif args.task=='vqa' or args.task=='vqa_names' or args.task=="vqa_names_ambiguity":
        prompts=['USER: <image>\nIn this image, the entity enclosed in a {} box is called \"{}\". Never mention the box and its color and answer the following question about \"{}\" : {} .\nASSISTANT:'.format(color,label.upper(),label.upper(),question)]
    elif args.task=='tqa':
        prompts=['Read this text: {}\nAnswer the following question about \"{}\" : {}\nASSISTANT:'.format(context,label.upper(),question)]
    elif args.task=='base_vlm_vqa' or args.task=='base_vlm_vqa_names' or args.task=="base_vlm_vqa_names_ambiguity":
        prompts=['USER: <image>\n{}\nASSISTANT:'.format(question)]
    if args.vlm_model=='GPT-4o':
        prompts=build_question_prompt(
            entity_name=label,
            color=color,
            question=question
        )
    return prompts





import os
import toml
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage
import ast
class AzureOpenAILLM:
    _instance = None

    def __new__(cls, config_file):
        if cls._instance is None:
            cls._instance = super(AzureOpenAILLM, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, config_file):
        if self._initialized:
            return

        self.config = toml.load(config_file)
        self.openai_api_type        = self.config['openai']['OPENAI_API_TYPE']
        self.openai_api_version     = self.config['openai']['OPENAI_API_VERSION']
        self.openai_api_base        = self.config['openai']['OPENAI_API_BASE'] 
        self.openai_chat_model      = self.config['openai']['OPENAI_CHAT_MODEL']
        self.openai_api_key         = self.config['openai']['OPENAI_API_KEY']
        self.openai_temp            = self.config['openai']['TEMP']
        self.openai_stream          = self.config['openai']['STREAM']
        self.openai_verbose         = self.config['openai']['VERBOSE']
        
        self.llm = self._create_llm(
            self.openai_api_type,
            self.openai_api_version,
            self.openai_api_base, 
            self.openai_chat_model, 
            self.openai_api_key,
            self.openai_temp,
            self.openai_stream,
            self.openai_verbose,
        )

        self._initialized = True

    def get_azure_openai_llm(self):
        return self.llm
    
    def _create_llm(
        self,
        openai_api_type,
        openai_api_version,
        openai_api_base, 
        openai_chat_model, 
        openai_api_key,
        openai_temp,
        openai_stream,
        openai_verbose,
    ):
        os.environ["OPENAI_API_TYPE"]               = openai_api_type
        os.environ["OPENAI_API_VERSION"]            = openai_api_version
        os.environ["AZURE_OPENAI_ENDPOINT"]         = openai_api_base
        os.environ["AZURE_OPENAI_API_KEY"]          = openai_api_key
        os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"]  = openai_chat_model
        
        return AzureChatOpenAI(
                    deployment_name     = openai_chat_model,
                    azure_endpoint      = openai_api_base,
                    temperature         = openai_temp,
                    streaming           = openai_stream,
                    verbose             = openai_verbose,
                )

#def pil_to_data_url(pil_image, format="PNG"):
#    # Save to an in-memory buffer
#    buf = BytesIO()
#    pil_image.save(buf, format=format)
#    img_bytes = buf.getvalue()

    # Encode to base64
#    img_b64 = base64.b64encode(img_bytes).decode("utf-8")

    # Build data URL
#    mime = f"image/{format.lower()}"
#    return f"data:{mime};base64,{img_b64}"
import base64
from langchain_openai import AzureChatOpenAI
from io import BytesIO
class AzureOpenAILLM:
    _instance = None

    def __new__(cls, config_file):
        if cls._instance is None:
            cls._instance = super(AzureOpenAILLM, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, config_file):
        if self._initialized:
            return

        self.config = toml.load(config_file)
        self.openai_api_type        = self.config['openai']['OPENAI_API_TYPE']
        self.openai_api_version     = self.config['openai']['OPENAI_API_VERSION']
        self.openai_api_base        = self.config['openai']['OPENAI_API_BASE'] 
        self.openai_chat_model      = self.config['openai']['OPENAI_CHAT_MODEL']
        self.openai_api_key         = self.config['openai']['OPENAI_API_KEY']
        self.openai_temp            = self.config['openai']['TEMP']
        self.openai_stream          = self.config['openai']['STREAM']
        self.openai_verbose         = self.config['openai']['VERBOSE']
        
        self.llm = self._create_llm(
            self.openai_api_type,
            self.openai_api_version,
            self.openai_api_base, 
            self.openai_chat_model, 
            self.openai_api_key,
            self.openai_temp,
            self.openai_stream,
            self.openai_verbose,
        )

        self._initialized = True

    def get_azure_openai_llm(self):
        return self.llm
    
    def _create_llm(
        self,
        openai_api_type,
        openai_api_version,
        openai_api_base, 
        openai_chat_model, 
        openai_api_key,
        openai_temp,
        openai_stream,
        openai_verbose,
    ):
        os.environ["OPENAI_API_TYPE"]               = openai_api_type
        os.environ["OPENAI_API_VERSION"]            = openai_api_version
        os.environ["AZURE_OPENAI_ENDPOINT"]         = openai_api_base
        os.environ["AZURE_OPENAI_API_KEY"]          = openai_api_key
        os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"]  = openai_chat_model
        
        return AzureChatOpenAI(
                    deployment_name     = openai_chat_model,
                    azure_endpoint      = openai_api_base,
                    temperature         = openai_temp,
                    streaming           = openai_stream,
                    verbose             = openai_verbose,
                )

class Azure_evaluate:
    def __init__(self):
        config_file = "../config.ini"
        os.environ["CONFIG_FILE"] = config_file
        self.llm_model = AzureOpenAILLM(os.environ["CONFIG_FILE"]).get_azure_openai_llm()
    def evaluate_answer(self, question, pred, answer):
        prompt="You are an intelligent chatbot designed for evaluating the correctness of generative outputs for question-answer pairs. " +\
        "Your task is to compare the predicted answer with the correct answer and determine if they match meaningfully. Here's how you can accomplish the task:"+\
        "------"+\
        "##INSTRUCTIONS: "+\
        "- Focus on the meaningful match between the predicted answer and the correct answer.\n"+\
        "- Consider synonyms or paraphrases as valid matches.\n"+\
        "- Evaluate the correctness of the prediction compared to the answer."+\
        "Please evaluate the following question-answer pair:\n\n"+\
        f"Question: {question}\n"+\
        f"Correct Answer: {answer}\n"+\
        f"Predicted Answer: {pred}\n\n"+\
        "Provide your evaluation only as a Yes/No."+\
        "DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION."
        ans=self.llm_model.invoke([HumanMessage(content=prompt)])
        return ans.content
