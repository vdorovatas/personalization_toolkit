"""
Dataset implementations for image processing with support for multiple datasets and augmentation strategies.

This module provides PyTorch Dataset classes for various image datasets with synchronized
transformations across different image processing pipelines (DINO, original).
Supports VQA (Visual Question Answering) and TQA (Text Question Answering) tasks.
"""

import os
import glob
import json
import pickle
import random
from typing import Any, List, Optional, Tuple, Dict

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

from utils import substring_before_last_dash


# Constants
ROTATION_RANGE = (-30, 30)
PIXEL_SCALE = 255

# Dataset object lists
OBJECTS_MYVLM = [
    'colorful teapot', 'sheep plush', 'gold pineapple', 'espresso cup', 'red piggy bank',
    'billy dog', 'running shoes', 'ceramic head', 'maeve dog', 'rabbit toy', 'skulls mug',
    'iverson funko pop', 'bull', 'green doll', 'red chicken', 'cat statue', 'minion toy',
    'asian doll', 'robot toy', 'elephant sphere', 'elephant statue', 'small penguin',
    'my cat', 'sheep toy', 'sheep pillow', 'dangling child', 'gengar toy', 'chicken bean bag',
    'boy funko pop'
]

OBJECTS_YOLLAVA = [
    'pig-cup-cup', 'thap-cham-temple', 'toodles-galore-cartoon cat', 'denisdang-face',
    'fire-cartoon woman', 'nha-tho-hcm-church', 'water-cartoon man', 'willinvietnam-face',
    'ciin-face', 'marie-cat-cartoon cat', 'phuc-map-face', 'khanhvy-face', 'thap-but-temple',
    'oong-face', 'neurips-cup-cup', 'dug-cartoon dog', 'yellow-duck-duck plush', 'thuytien-face',
    'thao-face', 'mydieu-cat', 'duck-banana-duck plush', 'viruss-face', 'nha-tho-hanoi-church',
    'mam-cat', 'tokyo-keyboard-keyboard', 'brown-duck-duck plush', 'shiba-gray-gray plush',
    'pusheen-cup-cup', 'butin-dog', 'chua-thien-mu-temple', 'bo-dog', 'lamb-lamb plush',
    'cat-cup-cup', 'shiba-yellow-yellow plush', 'henry-cat', 'shiba-sleep-sleeping plush',
    'dragon-dragon plush', 'yuheng-face', 'elephant-elephant plush', 'shiba-black-black plush'
]

OBJECTS_THIS_IS_MY = [
    'Sherrys road bike-bike', 'Blippis shoes-shoes', 'Caseys son-man', 'Gabs puppy lili-dog',
    'Caseys boosted board-skateboard', 'Alexs hat-hat', 'Reynards keyboard-keyboard',
    'Alexs everyday bag-bag', 'Zaks dog kona-dog', 'Zaks dog coffee-dog', 'Reynards work chair-chair',
    'Nikkis camper bag-bag', 'Caseys friend marlan-man', 'Nikkis car-car'
]

OBJECTS_THIS_IS_MY_MULTI = [
    'Sherry-woman', 'Sherrys road bike-bike',
    'Blippi-man', 'Blippis shoes-shoes',
    'Casey-man', 'Caseys son-man', 'Caseys boosted board-skateboard', 'Caseys friend marlan-man',
    'Gab-woman', 'Gabs puppy lili-dog',
    'Alex-woman', 'Alexs hat-hat', 'Alexs everyday bag-bag',
    'Reynards keyboard-keyboard', 'Reynards work chair-chair',
    'Nikki-woman', 'Nikkis camper bag-bag', 'Nikkis car-car',
    'Zak-man', 'Zaks dog kona-dog', 'Zaks dog coffee-dog'
]

OBJECTS_RAP = [
    'Anya-person', 'Baby_Q-person', 'Bingo-plush toy', 'Bluey-plush toy',
    'Bond-dog', 'Bull_dog-dog', 'C_dog-dog',
    'Character_A-cartoon character', 'Character_B-cartoon character', 'G_cat-cat', 'H-person',
    'J-person', 'K-person', 'Parrot_1-parrot', 'Parrot_2-parrot', 'T-person'
]

# Context pools (to be initialized elsewhere)
CONTEXT_POOL_THIS_IS_MY = None
CONTEXT_POOL_YOLLAVA = None
CONTEXT_POOL_MYVLM = None
CONTEXT_POOL_RAP = None


def collate_fn(batch: List) -> Optional[List]:
    """
    Custom collate function for DataLoader that filters out None entries.
    
    Args:
        batch: List of batch items, potentially containing None values
        
    Returns:
        Collated batch with None entries removed, or None if all entries are None
    """
    # Check if batch contains any valid data
    has_valid_data = any(item is not None for item in batch)
    
    if not has_valid_data:
        return None
    
    # Filter out None entries
    valid_batch = [item for item in batch if item is not None]
    
    # Transpose batch: convert list of tuples to tuple of lists
    collated_batch = list(map(list, zip(*valid_batch)))
    
    return collated_batch


class RandomCropSame:
    """Apply the same random crop to multiple images."""
    
    def __init__(self, size: int):
        """
        Initialize RandomCropSame.
        
        Args:
            size: Output size for both height and width
        """
        self.size = (size, size)
        self.i = None
        self.j = None
        self.h = None
        self.w = None
    
    def set_crop(self, img: Image.Image) -> None:
        """
        Set crop parameters based on input image.
        
        Args:
            img: PIL Image to determine crop parameters from
        """
        output_size = (img.size[1] // 2, img.size[0] // 2)
        self.i, self.j, self.h, self.w = transforms.RandomCrop.get_params(
            img, output_size=output_size
        )
    
    def __call__(self, img: Image.Image) -> Image.Image:
        """
        Apply the crop to an image.
        
        Args:
            img: PIL Image to crop
            
        Returns:
            Cropped PIL Image
        """
        return img.crop((self.j, self.i, self.j + self.w, self.i + self.h))


class HorizontalFlipSame:
    """Apply the same horizontal flip decision to multiple images."""
    
    def __init__(self):
        """Initialize HorizontalFlipSame."""
        self.flip = None
    
    def set_flip(self) -> None:
        """Randomly decide whether to flip."""
        self.flip = random.random() > 0.5
    
    def __call__(self, img: Image.Image) -> Image.Image:
        """
        Apply horizontal flip if enabled.
        
        Args:
            img: PIL Image to potentially flip
            
        Returns:
            Flipped or original PIL Image
            
        Raises:
            ValueError: If flip state not set
        """
        if self.flip is None:
            raise ValueError("Flip state not set. Call set_flip() before using.")
        
        return img.transpose(Image.FLIP_LEFT_RIGHT) if self.flip else img


class RandomRotationSame:
    """Apply the same random rotation to multiple images."""
    
    def __init__(self):
        """Initialize RandomRotationSame."""
        self.angle = None
    
    def set_rotation(self) -> None:
        """Set a random rotation angle."""
        self.angle = random.randint(*ROTATION_RANGE)
    
    def __call__(self, img: Image.Image) -> Image.Image:
        """
        Apply rotation to an image.
        
        Args:
            img: PIL Image to rotate
            
        Returns:
            Rotated PIL Image
            
        Raises:
            ValueError: If rotation angle not set
        """
        if self.angle is None:
            raise ValueError("Rotation angle not set. Call set_rotation() before using.")
        
        return img.rotate(self.angle)


class CustomResize:
    """Custom resize transform."""
    
    def __call__(self, img: Image.Image, new_size: Tuple[int, int]) -> Image.Image:
        """
        Resize image to specified dimensions.
        
        Args:
            img: PIL Image to resize
            new_size: Target size as (width, height)
            
        Returns:
            Resized PIL Image
        """
        return img.resize(new_size, Image.BILINEAR)


class CombinedTransformOriginalSize:
    """Combined transform that applies rotation and flip without resizing."""
    
    def __init__(self, random_rotation, random_crop, horizontal_flip):
        """
        Initialize transform.
        
        Args:
            random_rotation: RandomRotationSame instance
            random_crop: RandomCropSame instance (currently unused)
            horizontal_flip: HorizontalFlipSame instance
        """
        self.horizontal_flip = horizontal_flip
        self.random_crop = random_crop
        self.random_rotation = random_rotation
        self.transform = transforms.Compose([transforms.ToTensor()])
    
    def __call__(self, img: Image.Image) -> torch.Tensor:
        """
        Apply transforms and convert to tensor.
        
        Args:
            img: PIL Image to transform
            
        Returns:
            Transformed image as tensor
        """
        img = self.random_rotation(img)
        img = self.horizontal_flip(img)
        return self.transform(img)



class CombinedTransformDINO:
    """Combined transform for DINO model with resizing."""
    
    def __init__(self, args, random_rotation, random_crop, horizontal_flip):
        """
        Initialize transform.
        
        Args:
            args: Arguments containing dino_img_size
            random_rotation: RandomRotationSame instance
            random_crop: RandomCropSame instance (currently unused)
            horizontal_flip: HorizontalFlipSame instance
        """
        self.horizontal_flip = horizontal_flip
        self.random_crop = random_crop
        self.random_rotation = random_rotation
        self.args = args
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.resize = CustomResize()
    
    def __call__(self, img: Image.Image) -> torch.Tensor:
        """
        Apply transforms, resize for DINO, and convert to tensor.
        
        Args:
            img: PIL Image to transform
            
        Returns:
            Transformed image as tensor
        """
        img = self.random_rotation(img)
        img = self.horizontal_flip(img)
        img = self.resize(img, (self.args.dino_img_size, self.args.dino_img_size))
        return self.transform(img)


class BaseImageDataset(Dataset):
    """Base class for image datasets with common functionality."""
    
    def __init__(self, args, transform, transform_dino):
        """
        Initialize base dataset.
        
        Args:
            args: Configuration arguments
            transform: Transform for original size images
            transform_dino: Transform for DINO model
        """
        self.args = args
        self.transform = transform
        self.transform_dino = transform_dino
        
        # Initialize augmentation instances if needed
        if args.variation == 'augment' and args.split=='train':
            self.random_crop_instance = RandomCropSame(size=336)
            self.random_rotation_instance = RandomRotationSame()
            self.horizontal_flip_instance = HorizontalFlipSame()
        
        self.dataset = []
    
    def _setup_augmentation(self, img_raw: Image.Image) -> None:
        """
        Set up augmentation parameters and update transforms.
        
        Args:
            img_raw: Raw PIL Image to base augmentation on
        """
        if self.args.variation != 'augment' or self.args.split!='train':
            return
        
        # Set augmentation parameters
        self.horizontal_flip_instance.set_flip()
        self.random_rotation_instance.set_rotation()
        self.random_crop_instance.set_crop(img_raw)
        
        # Update transform instances
        self.transform = CombinedTransformOriginalSize(
            self.random_rotation_instance,
            self.random_crop_instance,
            self.horizontal_flip_instance
        )
        self.transform_dino = CombinedTransformDINO(
            self.args,
            self.random_rotation_instance,
            self.random_crop_instance,
            self.horizontal_flip_instance
        )
    
    def _convert_to_numpy(self, img_tensor: torch.Tensor) -> Any:
        """
        Convert tensor to numpy array for batch processing.
        
        Args:
            img_tensor: Image tensor in CHW format
            
        Returns:
            Numpy array in HWC format with pixel values [0, 255]
        """
        return (img_tensor.permute([1, 2, 0]) * PIXEL_SCALE).int().numpy()
    
    def __len__(self) -> int:
        """Return the size of the dataset."""
        return len(self.dataset)


class This_is_my_Dataset(BaseImageDataset):
    """Dataset for 'This is My' dataset."""
    
    def __init__(self, args, transform, transform_dino, **kwargs):
        """
        Initialize This-is-my dataset.
        
        Args:
            args: Configuration arguments
            transform: Transform for original size images
            transform_dino: Transform for DINO model
            **kwargs: Additional keyword arguments
        """
        super().__init__(args, transform, transform_dino)
        
        # Construct data path
        assert args.multi_concept is not None, "For This-is-My-Img dataset please provide a value for args.multi_concept flag."
        if args.multi_concept: #HERE
            data_path = os.path.join(args.data_folder, "This-is-My-Img/Multi-concept", args.split)
        else:
            data_path = os.path.join(args.data_folder, "This-is-My-Img/Single-concept", args.split)
        

        #import pdb;pdb.set_trace()
        # Load dataset files
        if args.split=="train":
            # Get all image files
            all_images = glob.glob(os.path.join(data_path, "**", "*.png"), recursive=True)
            
            # Group images by category (subfolder)
            category_images = {}
            for img_path in all_images:
                # Extract category from path (parent directory name)
                category = os.path.basename(os.path.dirname(img_path))
                if category not in category_images:
                    category_images[category] = []
                category_images[category].append(img_path)
            
            # Select n_training_views images per category
            self.dataset = []
            for category, images in category_images.items():
                # Take only n_training_views images from each category
                selected_images = images[:args.n_training_views]
                self.dataset.extend(selected_images)
        elif args.split=="test":
            assert args.test_split is not None, "For This-is-My dataset in test split, args.test_split must be specified (e.g.,'Positive', 'Negative (other), Negative (hard)')."
            if args.dataset=="this-is-my" and not args.multi_concept:
                data_path = os.path.join(data_path,args.test_split)
            #import pdb;pdb.set_trace()
            self.dataset = glob.glob(os.path.join(data_path, "**", "*.png"))
    
    def __getitem__(self, index: int) -> Tuple:
        """
        Get dataset item by index.
        
        Args:
            index: Index of item to retrieve
            
        Returns:
            Tuple of (img, img_dino, path, label, None, None, None, None, None, None)
        """
        # Load image
        img_raw = Image.open(self.dataset[index])
        
        # Set up augmentation if needed
        self._setup_augmentation(img_raw)
        
        # Apply transforms
        img = self.transform(img_raw)
        img_dino = self.transform_dino(img_raw)
        
        # Extract metadata
        path = self.dataset[index]
        label = path.split('/')[-2]
        
        # Convert to numpy for batch processing
        img = self._convert_to_numpy(img)
        img_dino = self._convert_to_numpy(img_dino)
        
        return img, img_dino, path, label, None, None, None, None, None, None


class This_is_my_VQA(Dataset):
    """Visual Question Answering dataset for 'This is My' dataset."""
    
    # Label mapping for proper names with apostrophes
    LABEL_MAP_DEFAULT = {
        "Alexs everyday bag": "Alex's everyday bag",
        "Alexs hat": "Alex's hat",
        "Blippis shoes": "Blippi's shoes",
        "Caseys boosted board": "Casey's boosted board",
        "Caseys friend marlan": "Casey's friend Marlan",
        "Caseys son": "Casey's son",
        "Gabs puppy lili": "Gab's puppy Lili",
        "Nikkis camper bag": "Nikki's camper bag",
        "Nikkis car": "Nikki's car",
        "Reynards keyboard": "Reynard's keyboard",
        "Reynards work chair": "Reynard's work chair",
        "Sherrys road bike": "Sherry's road bike",
        "Zaks dog coffee": "Zak's dog Coffee",
        "Zaks dog kona": "Zak's dog Kona",
    }
    
    # Anonymized label mapping for name-based tasks
    LABEL_MAP_NAMES = {
        "Alexs everyday bag": "Aero",
        "Alexs hat": "Vex",
        "Blippis shoes": "Echo",
        "Caseys boosted board": "Nova",
        "Caseys friend marlan": "Flux",
        "Caseys son": "Zenith",
        "Gabs puppy lili": "Rune",
        "Nikkis camper bag": "Quill",
        "Nikkis car": "Pax",
        "Reynards keyboard": "Blaze",
        "Reynards work chair": "Frost",
        "Sherrys road bike": "Loom",
        "Zaks dog coffee": "Glint",
        "Zaks dog kona": "Shade",
    }
    
    def __init__(self, args, transform, transform_dino, **kwargs):
        """
        Initialize This-is-my VQA dataset.
        
        Args:
            args: Configuration arguments
            transform: Transform for original size images
            transform_dino: Transform for DINO model
            **kwargs: Additional keyword arguments
        """
        self.args = args
        self.transform = transform
        self.transform_dino = transform_dino
        
        # Select appropriate JSON file based on task
        json_filename = 'this-is-my-visual-qa.json'
        if args.task in ['vqa_names_ambiguity', 'base_vlm_vqa_names_ambiguity']:
            json_filename = 'this-is-my-visual-qa-ambiguity.json'
        elif args.task in ['vqa_names_ambiguity_full', 'base_vlm_vqa_names_ambiguity_full']:
            json_filename = 'this-is-my-visual-qa-ambiguity-full.json'
        
        json_path = os.path.join(
            args.data_folder,
            "this-is-my/rgb_dataset/sampled/selected_frames/",
            json_filename
        )
        
        # Construct data path
        data_path = os.path.join(
            args.data_folder,
            "this-is-my/rgb_dataset/sampled/selected_frames/",
            args.split
        )
        
        # Override path for VQA tasks
        if 'vqa' in args.task:
            data_path = "/mnt/lustrefs/data/datasets/zeroshot-seg/personalization/this-is-my/rgb_dataset/sampled/selected_frames/validation_vqa/"
        
        # Load VQA data
        with open(json_path, 'r') as file:
            self.vqa = json.load(file)
        
        # Load dataset files
        if args.specific_cat is not None:
            pattern = os.path.join(data_path, args.specific_cat, "*.png")
            print(f"Loading from: {pattern}")
            self.dataset = glob.glob(pattern)
        else:
            self.dataset = glob.glob(os.path.join(data_path, "**", "*.png"))
        
        # Select appropriate label mapping
        name_tasks = [
            'vqa_names', 'vqa_names_ambiguity', 'base_vlm_vqa_names',
            'base_vlm_vqa_names_ambiguity', 'base_vlm_vqa_names_ambiguity_full',
            'vqa_names_ambiguity_full'
        ]
        
        if args.task in name_tasks:
            self.label_map = self.LABEL_MAP_NAMES
        else:
            self.label_map = self.LABEL_MAP_DEFAULT
    
    def __getitem__(self, index: int) -> Tuple:
        """
        Get dataset item by index.
        
        Args:
            index: Index of item to retrieve
            
        Returns:
            Tuple of (img, img_dino, path, label, question, answer, None, 
                     label_map, open_question, full_answer)
        """
        # Load and convert image to RGB
        img_raw = Image.open(self.dataset[index]).convert("RGB")
        
        # Apply transforms
        img = self.transform(img_raw)
        img_dino = self.transform_dino(img_raw)
        
        # Extract metadata
        path = self.dataset[index]
        filename = path.split('/')[-1]
        label = path.split('/')[-2].split('-')[0]
        
        # Initialize QA variables
        question = None
        answer = None
        full_answer = None
        open_question = None
        
        # Find matching QA pair
        for conv in self.vqa[label]:
            if filename == conv.split('/')[-1]:
                # Format question with options
                question = (
                    f"{self.vqa[label][conv]['question']}"
                    f"A.{self.vqa[label][conv]['options']['A']} or "
                    f"B.{self.vqa[label][conv]['options']['B']}"
                )
                question = question.replace('<sks>', self.label_map[label])
                
                # Format answer
                correct_ans = self.vqa[label][conv]['correct_answer']
                answer = f"{correct_ans}.{self.vqa[label][conv]['options'][correct_ans]}"
                
                # Try to get GPT answer if available
                try:
                    full_answer = self.vqa[label][conv]["gpt_answer"]
                    open_question = self.vqa[label][conv]['question'].replace(
                        '<sks>', self.label_map[label]
                    )
                    full_answer = full_answer.replace('<sks>', self.label_map[label])
                except KeyError:
                    pass
        
        # Convert to numpy for batch processing
        img = (img.permute([1, 2, 0]) * PIXEL_SCALE).int().numpy()
        img_dino = (img_dino.permute([1, 2, 0]) * PIXEL_SCALE).int().numpy()
        
        return (img, img_dino, path, label, question, answer, None,
                self.label_map, open_question, full_answer)
    
    def __len__(self) -> int:
        """Return the size of the dataset."""
        return len(self.dataset)


class YoLLavaDataset(BaseImageDataset):
    """Dataset for YoLLava dataset."""
    
    def __init__(self, args, transform, transform_dino, **kwargs):
        """
        Initialize YoLLava dataset.
        
        Args:
            args: Configuration arguments
            transform: Transform for original size images
            transform_dino: Transform for DINO model
            **kwargs: Additional keyword arguments
        """
        super().__init__(args, transform, transform_dino)
        
        # Construct data path
        data_path = os.path.join(
            args.data_folder, "YoLLaVA", args.split
        )

        if args.split=='train':
        
            # Get all image files
            all_images = glob.glob(os.path.join(data_path, "**", "*.png"), recursive=True)
            
            # Group images by category (subfolder)
            category_images = {}
            for img_path in all_images:
                # Extract category from path (parent directory name)
                category = os.path.basename(os.path.dirname(img_path))
                if category not in category_images:
                    category_images[category] = []
                category_images[category].append(img_path)
            
            # Select n_training_views images per category
            self.dataset = []
            for category, images in category_images.items():
                # Take only n_training_views images from each category
                selected_images = images[:args.n_training_views]
                self.dataset.extend(selected_images)
        if args.split=='test':
            self.dataset = glob.glob(os.path.join(data_path, "**", "*.png"))
        
    
    def __getitem__(self, index: int) -> Tuple:
        """
        Get dataset item by index.
        
        Args:
            index: Index of item to retrieve
            
        Returns:
            Tuple of (img, img_dino, path, label, None, None, None, None, None, None)
        """
        # Load and convert image to RGB
        img_raw = Image.open(self.dataset[index]).convert("RGB")
        
        # Set up augmentation if needed
        self._setup_augmentation(img_raw)
        
        # Apply transforms
        img = self.transform(img_raw)
        img_dino = self.transform_dino(img_raw)
        
        # Extract metadata
        path = self.dataset[index]
        label = path.split('/')[-2]
        
        # Convert to numpy for batch processing
        img = self._convert_to_numpy(img)
        img_dino = self._convert_to_numpy(img_dino)
        
        return img, img_dino, path, label, None, None, None, None, None, None


class YoLLavaDatasetVQA(Dataset):
    """Visual Question Answering dataset for YoLLava."""
    
    def __init__(self, args, transform, transform_dino, **kwargs):
        """
        Initialize YoLLava VQA dataset.
        
        Args:
            args: Configuration arguments
            transform: Transform for original size images
            transform_dino: Transform for DINO model
            **kwargs: Additional keyword arguments
        """
        self.args = args
        self.transform = transform
        self.transform_dino = transform_dino
        
        # Load VQA JSON
        # json_path = os.path.join(
        #     args.data_folder, "yollava/yollava-data/", 'yollava-visual-qa.json'
        # )
        json_path = "/fsx/ad/vlm/clean_datasets_CVPR2026/yollava/yollava-visual-qa.json"
        
        
        with open(json_path, 'r') as file:
            self.vqa = json.load(file)
        
        # Construct data path
        # data_path = os.path.join(args.data_folder, "yollava/yollava-data/", args.split)
        data_path = "/fsx/ad/vlm/clean_datasets_CVPR2026/yollava/train_1view_448"
        
        # if args.split == 'train':
        #     data_path = os.path.join(data_path, 'labelled')
        
        # Load dataset files
        self.dataset = glob.glob(os.path.join(data_path, "**", "*.png"))
        print(self.dataset)
    
    def __getitem__(self, index: int) -> Tuple:
        """
        Get dataset item by index.
        
        Args:
            index: Index of item to retrieve
            
        Returns:
            Tuple of (img, img_dino, path, label, question, answer, 
                     None, None, None, None)
        """
        # Load and convert image to RGB
        img_raw = Image.open(self.dataset[index]).convert("RGB")
        
        # Apply transforms
        img = self.transform(img_raw)
        img_dino = self.transform_dino(img_raw)
        
        # Extract metadata
        path = self.dataset[index]
        filename = path.split('/')[-1]
        label = path.split('/')[-2]
        
        # Initialize QA variables
        question = None
        answer = None
        
        # Find matching QA pair
        for conv in self.vqa[label]:
            if filename == conv.split('/')[-1]:
                # Format question with options
                question = (
                    f"{self.vqa[label][conv]['question']}"
                    f"A.{self.vqa[label][conv]['options']['A']} or "
                    f"B.{self.vqa[label][conv]['options']['B']}"
                )
                question = question.replace('<sks>', label)
                
                # Format answer
                correct_ans = self.vqa[label][conv]['correct_answer']
                answer = f"{correct_ans}.{self.vqa[label][conv]['options'][correct_ans]}"
        
        # Convert to numpy for batch processing
        img = (img.permute([1, 2, 0]) * PIXEL_SCALE).int().numpy()
        img_dino = (img_dino.permute([1, 2, 0]) * PIXEL_SCALE).int().numpy()
        
        return img, img_dino, path, label, question, answer, None, None, None, None
    
    def __len__(self) -> int:
        """Return the size of the dataset."""
        return len(self.dataset)
    
class RAPDataset(BaseImageDataset):
    """Dataset for RAP dataset."""
    
    def __init__(self, args, transform, transform_dino, **kwargs):
        """
        Initialize RAP dataset.
        
        Args:
            args: Configuration arguments
            transform: Transform for original size images
            transform_dino: Transform for DINO model
            **kwargs: Additional keyword arguments
        """
        super().__init__(args, transform, transform_dino)
        
        # Construct data path
        if args.split=='train':
            data_path = "/fsx/ad/vlm/clean_datasets_CVPR2026/rap/multi-concept/train_1view_pekit/"
        elif args.split=='validation':
            data_path = "/fsx/ad/vlm/clean_datasets_CVPR2026/rap/multi-concept/validation/"
        #data_path = os.path.join(
        #    args.data_folder, "yollava", args.split
        #)
        
        # Load dataset files
        #import pdb;pdb.set_trace()
        self.dataset = glob.glob(os.path.join(data_path, "**", "*.jpg"))
    
    def __getitem__(self, index: int) -> Tuple:
        """
        Get dataset item by index.
        
        Args:
            index: Index of item to retrieve
            
        Returns:
            Tuple of (img, img_dino, path, label, None, None, None, None, None, None)
        """
        # Load and convert image to RGB
        img_raw = Image.open(self.dataset[index]).convert("RGB")
        
        # Set up augmentation if needed
        self._setup_augmentation(img_raw)
        
        # Apply transforms
        img = self.transform(img_raw)
        img_dino = self.transform_dino(img_raw)
        
        # Extract metadata
        path = self.dataset[index]
        label = path.split('/')[-2]
        
        # Convert to numpy for batch processing
        img = self._convert_to_numpy(img)
        img_dino = self._convert_to_numpy(img_dino)
        
        return img, img_dino, path, label, None, None, None, None, None, None

class MyVLMDataset(BaseImageDataset):
    """Dataset for MyVLM dataset."""
    
    def __init__(self, args, transform, transform_dino, **kwargs):
        """
        Initialize MyVLM dataset.
        
        Args:
            args: Configuration arguments
            transform: Transform for original size images
            transform_dino: Transform for DINO model
            **kwargs: Additional keyword arguments
        """
        super().__init__(args, transform, transform_dino)
        
        # Construct data path
        data_path = os.path.join(args.data_folder, args.dataset, 'data')
        # Get all image files
        all_images = glob.glob(os.path.join(data_path, "**", "*.jpg"), recursive=True) + \
                     glob.glob(os.path.join(data_path, "**", "*.JPG"), recursive=True)
        
        if args.split=='train':
            # Group images by category (subfolder)
            category_images = {}
            for img_path in all_images:
                # Extract category from path (parent directory name)
                category = os.path.basename(os.path.dirname(img_path))
                if category not in category_images:
                    category_images[category] = []
                category_images[category].append(img_path)
            
            # Select n_training_views images per category
            self.dataset = []
            for category, images in category_images.items():
                # Take only n_training_views images from each category
                selected_images = images[:args.n_training_views]
                self.dataset.extend(selected_images)
        elif args.split=='test':
            self.dataset=all_images
    
    def __getitem__(self, index: int) -> Tuple:
        """
        Get dataset item by index.
        
        Args:
            index: Index of item to retrieve
            
        Returns:
            Tuple of (img, img_dino, path, label, None, None, None, None, None, None)
        """
        # Load and convert image to RGB
        img_raw = Image.open(self.dataset[index]).convert("RGB")
        
        # Set up augmentation if needed
        self._setup_augmentation(img_raw)
        
        # Apply transforms
        img = self.transform(img_raw)
        img_dino = self.transform_dino(img_raw)
        
        # Extract metadata
        path = self.dataset[index]
        label = path.split('/')[-2]
        
        # Convert to numpy for batch processing
        img = self._convert_to_numpy(img)
        img_dino = self._convert_to_numpy(img_dino)
        
        return img, img_dino, path, label, None, None, None, None, None, None


def get_dataloader(args):
    """
    Get dataloader and metadata for specified dataset.
    
    Args:
        args: Configuration arguments containing dataset specifications
        
    Returns:
        Tuple of (dataloader, objects_list, context_pool)
        
    Raises:
        ValueError: If task is not supported for the specified dataset
    """
    # Define base transforms
    base_transform = transforms.Compose([transforms.ToTensor()])
    dino_transform = transforms.Compose([
        transforms.Resize([args.dino_img_size, args.dino_img_size]),
        transforms.ToTensor()
    ])
    
    # DataLoader common parameters
    dataloader_params = {
        'batch_size': args.batch_size,
        'shuffle': args.shuffle,
        'num_workers': args.num_workers,
        'pin_memory': True,
        'drop_last': True,
        'collate_fn': collate_fn
    }
    
    # MyVLM Dataset
    if args.dataset == 'myvlm':
        if args.task in ['vqa', 'tqa', 'base_vlm_vqa']:
            raise ValueError(f"Task '{args.task}' is not supported for MyVLM dataset.")
        
        dataset = MyVLMDataset(
            args,
            transform=base_transform,
            transform_dino=dino_transform,
        )
        dataloader = torch.utils.data.DataLoader(dataset, **dataloader_params)
        return dataloader, OBJECTS_MYVLM, CONTEXT_POOL_MYVLM
    
    # YoLLava Dataset
    elif args.dataset == 'yollava':
        if args.task in ['vqa', 'base_vlm_vqa']:
            dataset = YoLLavaDatasetVQA(
                args,
                transform=base_transform,
                transform_dino=dino_transform,
            )
        else:
            dataset = YoLLavaDataset(
                args,
                transform=base_transform,
                transform_dino=dino_transform,
            )        
        dataloader = torch.utils.data.DataLoader(dataset, **dataloader_params)
        return dataloader, OBJECTS_YOLLAVA, CONTEXT_POOL_YOLLAVA
    
    elif args.dataset == 'rap':
        dataset = RAPDataset(
            args,
            transform=base_transform,
            transform_dino=dino_transform,
        )
        dataloader = torch.utils.data.DataLoader(dataset, **dataloader_params)
        return dataloader, OBJECTS_RAP, CONTEXT_POOL_RAP
    
    # This-is-my Dataset
    elif args.dataset == 'this-is-my':
        vqa_tasks = [
            'vqa', 'base_vlm_vqa', 'base_vlm_vqa_names', 'vqa_names',
            'vqa_names_ambiguity', 'base_vlm_vqa_names_ambiguity',
            'vqa_names_ambiguity_full', 'base_vlm_vqa_names_ambiguity_full'
        ]
        
        if args.task in vqa_tasks:
            dataset = This_is_my_VQA(
                args,
                transform=base_transform,
                transform_dino=dino_transform,
            )
        else:
            dataset = This_is_my_Dataset(
                args,
                transform=base_transform,
                transform_dino=dino_transform,
            )
        
        dataloader = torch.utils.data.DataLoader(dataset, **dataloader_params)
        
        # MODIFICATION: Return multi-concept object list if in multi-concept mode
        if args.multi_concept:
            return dataloader, OBJECTS_THIS_IS_MY_MULTI, CONTEXT_POOL_THIS_IS_MY
        else:
            return dataloader, OBJECTS_THIS_IS_MY, CONTEXT_POOL_THIS_IS_MY
    
    # ADD THIS ELSE BLOCK - it was missing!
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")