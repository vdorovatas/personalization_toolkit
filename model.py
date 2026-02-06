import torch
from transformers import AutoModel, AutoTokenizer
from transformers import GroundingDinoProcessor
from transformers import GroundingDinoForObjectDetection
import numpy as np
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from transformers import AutoProcessor, AutoTokenizer
import torchvision.transforms as T
import math
from transformers import AutoModelForMaskGeneration, AutoProcessor, AutoConfig
import math
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer
import matplotlib.pyplot as plt

class Personalized_InternVL(torch.nn.Module):
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)
    # Initialize class
    def __init__(self, args):
        super(Personalized_InternVL, self).__init__()
        with torch.no_grad():
            self.args=args
            path = args.vlm_model
            device_map = self.split_model(args.vlm_model)
            self.intern_model = AutoModel.from_pretrained(
                path,
                torch_dtype=torch.bfloat16,
                load_in_8bit=False,
                low_cpu_mem_usage=True,
                use_flash_attn=True,
                trust_remote_code=True,
                device_map=device_map).eval()
            
            self.tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)


    def build_transform(self, input_size):
        MEAN, STD = self.IMAGENET_MEAN, self.IMAGENET_STD
        transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=MEAN, std=STD)
        ])
        return transform

    def find_closest_aspect_ratio(self, aspect_ratio, target_ratios, width, height, image_size):
        best_ratio_diff = float('inf')
        best_ratio = (1, 1)
        area = width * height
        for ratio in target_ratios:
            target_aspect_ratio = ratio[0] / ratio[1]
            ratio_diff = abs(aspect_ratio - target_aspect_ratio)
            if ratio_diff < best_ratio_diff:
                best_ratio_diff = ratio_diff
                best_ratio = ratio
            elif ratio_diff == best_ratio_diff:
                if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                    best_ratio = ratio
        return best_ratio

    def dynamic_preprocess(self, image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
        orig_width, orig_height = image.size
        aspect_ratio = orig_width / orig_height

        # calculate the existing image aspect ratio
        target_ratios = set(
            (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
            i * j <= max_num and i * j >= min_num)
        target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

        # find the closest aspect ratio to the target
        target_aspect_ratio = self.find_closest_aspect_ratio(
            aspect_ratio, target_ratios, orig_width, orig_height, image_size)

        # calculate the target width and height
        target_width = image_size * target_aspect_ratio[0]
        target_height = image_size * target_aspect_ratio[1]
        blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

        # resize the image
        resized_img = image.resize((target_width, target_height))
        processed_images = []
        for i in range(blocks):
            box = (
                (i % (target_width // image_size)) * image_size,
                (i // (target_width // image_size)) * image_size,
                ((i % (target_width // image_size)) + 1) * image_size,
                ((i // (target_width // image_size)) + 1) * image_size
            )
            # split the image
            split_img = resized_img.crop(box)
            processed_images.append(split_img)
        assert len(processed_images) == blocks
        if use_thumbnail and len(processed_images) != 1:
            thumbnail_img = image.resize((image_size, image_size))
            processed_images.append(thumbnail_img)
        return processed_images

    def load_image(self, image_file, input_size=448, max_num=12):
        image = Image.open(image_file).convert('RGB')
        transform = self.build_transform(input_size=input_size)
        images = self.dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
        pixel_values = [transform(image) for image in images]
        pixel_values = torch.stack(pixel_values)
        return pixel_values

    def split_model(self, model_name):
        device_map = {}
        world_size = torch.cuda.device_count()
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        num_layers = config.llm_config.num_hidden_layers
        # Since the first GPU will be used for ViT, treat it as half a GPU.
        num_layers_per_gpu = math.ceil(num_layers / (world_size - 0.5))
        num_layers_per_gpu = [num_layers_per_gpu] * world_size
        num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * 0.5)
        layer_cnt = 0
        for i, num_layer in enumerate(num_layers_per_gpu):
            for j in range(num_layer):
                device_map[f'language_model.model.layers.{layer_cnt}'] = i
                layer_cnt += 1
        device_map['vision_model'] = 0
        device_map['mlp1'] = 0
        device_map['language_model.model.tok_embeddings'] = 0
        device_map['language_model.model.embed_tokens'] = 0
        device_map['language_model.output'] = 0
        device_map['language_model.model.norm'] = 0
        device_map['language_model.model.rotary_emb'] = 0
        device_map['language_model.lm_head'] = 0
        device_map[f'language_model.model.layers.{num_layers - 1}'] = 0

        return device_map

    def forward(self,image,question):
        with torch.no_grad():
            # set the max number of tiles in `max_num`
            question=question[0]
            #import pdb;pdb.set_trace()
            pixel_values = self.load_image(image, max_num=1).to(torch.bfloat16).to(self.args.device)
            generation_config = dict(max_new_tokens=50, do_sample=False)

            # single-image single-round conversation (单图单轮对话)
            #question = '<image>\nPlease describe the image shortly.'
            #print(self.intern_model.device)
            #print(pixel_values.device)
            response = self.intern_model.chat(self.tokenizer, pixel_values, question, generation_config)
            response=(f'User: {question}\nAssistant: {response}')
            response=response.split("\nAssistant: ")[-1]
        return response
    
class Extractor(torch.nn.Module):
    # Initialize class
    def __init__(self, args):
        super(Extractor, self).__init__()
        with torch.no_grad():
            self.args = args
            sam_checkpoint = "/shared/home/SSO3984/Pekit/sam_vit_l_0b3195.pth"
            model_type = "vit_l"
            sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
            sam.to(device=args.device)
            self.mask_generator= SamAutomaticMaskGenerator(model=sam, min_mask_region_area=args.min_mask_area,pred_iou_thresh= 0.95)
            # DinoV2 model
            self.dino_transforms=transforms.Compose([
                transforms.Resize(size=518, interpolation=InterpolationMode.BICUBIC, max_size=None, antialias=None),
                transforms.CenterCrop(size=(518, 518)),
                transforms.ToTensor(),
                transforms.Normalize(mean=torch.tensor([0.4850, 0.4560, 0.4060]), std=torch.tensor([0.2290, 0.2240, 0.2250])),])
            self.dino_model=torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14').to(args.device)
            self.g_dino_processor = GroundingDinoProcessor.from_pretrained("IDEA-Research/grounding-dino-base")
            self.g_dino_model = GroundingDinoForObjectDetection.from_pretrained("IDEA-Research/grounding-dino-base").to(args.device)
            

            
    
    def forward_sam(self, image):
        all_masks=[]
        with torch.no_grad():
            for img in image:
                outputs = self.mask_generator.generate(np.uint8(img))
                img_masks=[x['segmentation'] for x in outputs if np.sum(x['segmentation'].astype(int))>self.args.min_mask_area]
                all_masks.append(np.stack(img_masks))
        return all_masks
    
    # Forward pass into DinoV2 model
    def forward_dino(self, image):
        self.dino_model.eval()
        with torch.no_grad():
            pil_image = torch.stack([self.dino_transforms(Image.fromarray(np.uint8(img))).to(self.args.device) for img in image]).to(self.args.device)
            dino_output = self.dino_model(pil_image, is_training=True)
            dino_features=dino_output['x_norm_patchtokens'].reshape([-1,self.args.dino_img_size//self.args.dino_patch_size,
                                                        self.args.dino_img_size//self.args.dino_patch_size, self.args.dino_embedding_size])
        #py_logger.debug("Forwarded data into dino.")  
        
        return dino_features
    
    def forward_grounding_dino(self, image, text):
        # FOR NOW: only batch_size = 1
        # image = torch.stack([Image.fromarray(np.uint8(img)).to(self.args.device) for img in image]).to(self.args.device)
        # text = torch.stack(text)
        image = Image.fromarray(image[0].astype(np.uint8))
        inputs = self.g_dino_processor(images=image, text=text[0], return_tensors="pt").to(self.args.device)
        # inputs = self.g_dino_processor(images=image, text=text[0], return_tensors="pt").to(self.args.device) #HERE replaced with above
        with torch.no_grad():
            outputs = self.g_dino_model(**inputs)
        width, height = image.size


        print(f"\n>>> Raw outputs from G-DINO:")
        print(f"    logits shape: {outputs.logits.shape}")
        print(f"    boxes shape: {outputs.pred_boxes.shape}")
        print(f"    Number of predictions: {outputs.pred_boxes.shape[1]}")


        postprocessed_outputs = self.g_dino_processor.image_processor.post_process_object_detection(outputs,
                                                                        target_sizes=[(height, width)],
                                                                        threshold=0.01)
        results = postprocessed_outputs[0]
        bboxes = results['boxes'].tolist()
        print(f"\n>>> After post_process_object_detection:")
        print(f"    Threshold used: 0.01")
        print(f"    Boxes returned: {len(results['boxes'])}")
        print(f"    Scores: {results['scores'][:10].tolist() if len(results['scores']) > 0 else 'none'}")
        masks = []
        for b in bboxes:
            mask = self.bbox_to_mask(width, height, b)
            masks.append(mask)
        masks = [np.stack(masks)]
 
        return masks

    def bbox_to_mask(self, width, height, bbox):
        x_min, y_min, x_max, y_max = bbox[0], bbox[1], bbox[2], bbox[3]
        x_min, y_min, x_max, y_max = map(int, bbox)
        mask_shape = (width, height)
        bbox_mask = np.zeros(mask_shape, dtype=bool)
        bbox_mask[y_min:y_max, x_min:x_max] = True
        return bbox_mask

#Model Used to extract the training views for each object
class Train_Extractor(torch.nn.Module):
    def __init__(self, args):
        super(Train_Extractor, self).__init__()
        with torch.no_grad():
            self.args = args
            self.device = self.args.device
            
            # Always load DINO and Grounding DINO
            self.dino_transforms=transforms.Compose([
                                transforms.Resize(size=518, interpolation=InterpolationMode.BICUBIC, max_size=None, antialias=None),
                                transforms.CenterCrop(size=(518, 518)),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=torch.tensor([0.4850, 0.4560, 0.4060]), std=torch.tensor([0.2290, 0.2240, 0.2250])),])
            self.dino_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14').to(args.device)
            self.g_dino_processor = GroundingDinoProcessor.from_pretrained("IDEA-Research/grounding-dino-base")
            self.g_dino_model = GroundingDinoForObjectDetection.from_pretrained("IDEA-Research/grounding-dino-base").to(args.device)
            
            # Only load SAM if grounding_sam is enabled
            if args.grounding_sam:
                segmenter_id = "facebook/sam-vit-base"
                self.segmentator = AutoModelForMaskGeneration.from_pretrained(segmenter_id).to(self.device)
                self.processor = AutoProcessor.from_pretrained(segmenter_id)
                print("✓ SAM model loaded")
            else:
                self.segmentator = None
                self.processor = None
                print("✓ SAM model skipped (grounding_sam=False)")

    # Forward pass into DinoV2 model
    def forward_dino(self, image):
        self.dino_model.eval()
        with torch.no_grad():
            # Convert numpy arrays to PIL images and ensure RGB mode
            pil_images = []
            for img in image:
                # Create PIL image from numpy array
                pil_img = Image.fromarray(np.uint8(img))
                
                # Ensure RGB mode (convert if RGBA, L, etc.)
                if pil_img.mode != 'RGB':
                    pil_img = pil_img.convert('RGB')
                
                pil_images.append(pil_img)
            
            # Apply transforms and stack
            pil_image = torch.stack([
                self.dino_transforms(img).to(self.args.device) 
                for img in pil_images
            ]).to(self.args.device)
            
            # Forward through DINO
            dino_output = self.dino_model(pil_image, is_training=True)
            dino_features = dino_output['x_norm_patchtokens'].reshape(
                [-1, 
                self.args.dino_img_size // self.args.dino_patch_size,
                self.args.dino_img_size // self.args.dino_patch_size, 
                self.args.dino_embedding_size]
            )
        
        return dino_features
    
    def bbox_to_mask(self, width, height, bbox):
        x_min, y_min, x_max, y_max = bbox[0], bbox[1], bbox[2], bbox[3]
        x_min, y_min, x_max, y_max = map(int, bbox)
        mask_shape = (width, height)
        bbox_mask = np.zeros(mask_shape, dtype=bool)
        bbox_mask[y_min:y_max, x_min:x_max] = True
        return bbox_mask
    
    def forward_grounding_dino(self, image, text, use_g_sam=True):
        # FOR NOW: only batch_size = 1 
    
        # 1. Get the first image from the batch and ensure it's uint8
        image_np_raw = image[0].astype(np.uint8) 

        # 2. Create the PIL image
        image_pil = Image.fromarray(image_np_raw)

        # 3. Ensure the image is RGB
        if image_pil.mode != 'RGB':
            image_pil = image_pil.convert('RGB')

        # 4. Process with Grounding DINO using PIL image
        inputs = self.g_dino_processor(
            images=[image_pil], 
            text=self.preprocess_caption(text[0]), 
            return_tensors="pt"
        ).to(self.args.device)
        
        with torch.no_grad():
            outputs = self.g_dino_model(**inputs)
            
        # 5. Use the PIL image's size for post-processing
        width, height = image_pil.size

        
        postprocessed_outputs = self.g_dino_processor.image_processor.post_process_object_detection(
            outputs,
            target_sizes=[(height, width)],
            threshold=0.01
        )
        results = postprocessed_outputs[0]
        scores = results['scores'].tolist()
        th = 0.3
        while not scores:
            print("NOT detected: ", text[0])
            print("threshold is: ", th)
            th -= 0.2
            postprocessed_outputs = self.g_dino_processor.image_processor.post_process_object_detection(
                outputs,
                target_sizes=[(height, width)],
                threshold=th
            )
            results = postprocessed_outputs[0]
            scores = results['scores'].tolist()

        bboxes = results['boxes'].tolist()
        max_score_index = scores.index(max(scores))
        highest_score_bbox = bboxes[max_score_index]
        
        if not use_g_sam:
            mask = self.bbox_to_mask(width, height, highest_score_bbox)
            return [mask]
        else:
            # For SAM, pass the PIL image directly
            boxes = [[highest_score_bbox]]
            
            # **THIS IS THE KEY FIX**: Pass PIL image, not numpy array
            inputs = self.processor(
                images=image_pil,  # Changed from init_image to image_pil
                input_boxes=boxes, 
                return_tensors="pt"
            ).to(self.device)

            outputs = self.segmentator(**inputs)
            masks = self.processor.post_process_masks(
                masks=outputs.pred_masks,
                original_sizes=inputs.original_sizes,
                reshaped_input_sizes=inputs.reshaped_input_sizes
            )[0]

            scores = outputs['iou_scores'][0][0].detach().cpu().numpy().tolist()
            best_mask = masks[0][np.argmax(scores)]

            return [best_mask.cpu().numpy()]
    def preprocess_caption(self, caption: str) -> str:
        result = caption.lower().strip()
        if result.endswith("."):
            return result
        return result + "."
            
            
        
        
