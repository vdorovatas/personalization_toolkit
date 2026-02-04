📄 Paper Link: https://arxiv.org/abs/2502.02452

# 📌 Personalization Toolkit (PeKit)

## 📑 Sections
- 📁 📖 Overview  
- 🧠 Method Summary  
- 📂 Dataset: This-is-My-Img  
- ⚙️ Installation  
- 🚀 Evaluation

- 🚀 TODOs  
- 📚 Citation  
- 📌 Contribution Guidelines  

---

## 📖 Overview

**Personalization Toolkit (PeKit)** is a *training-free* approach for personalization of Large Vision-Language Models (LVLMs).  
Instead of fine-tuning or test-time training for each new concept, it uses:

- Pre-trained vision foundation models to extract distinctive features  
- Retrieval-Augmented Generation (RAG) to identify instances in visual inputs  
- Visual prompting to guide LVLM outputs efficiently  

This toolkit is model-agnostic, supports **multi-concept personalization**, and works on both **images and videos**.

## 🧠 Method Summary

### ✅ Key Components

1. **Training-Free View Extraction**  
   Extract object-level embeddings from reference images using pre-trained vision models (e.g., DINOv2, SAM).

2. **Personalized Object Retrieval**  
   Use a retrieval module over stored features to detect personalized concepts in query images.

3. **Personalized Answer Generation**  
   Generate tailored responses via LVLMs by overlaying visual prompts highlighting detected objects.

---

## 📂 Dataset: This-is-My-Img

Google Drive Link: **[ADD LINK HERE]**

### Structure

```text
This-is-My-Img/
├── Single-concept/
│   ├── Reference Images/
│   ├── Validation/
│       ├── positive/
│       ├── Negative (Hard)/
│       ├── Negative (Other)/
│       ├── Fake/
├── Multi-concept/
│   ├── Reference Images/
│   ├── Validation/
```

---

## ⚙️ Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/your-repo/pekit.git
cd pekit
pip install -r requirements.txt
```

(Optional) Install as a package:

```bash
pip install -e .
```

---

## 🚀 Evaluation

### Dataset Folder Structure

Download the datasets and organize them with the following structure:
```
myvlm/
└── data/
    └── [29 concepts]/

YoLLaVA/
├── train/
│   └── [40 concepts]/
└── test/
    └── [40 concepts]/

This-is-My-Img/
├── Single-concept/
│   ├── train/
│   │   └── [14 concepts]/
│   ├── test/
│   │   └── [14 concepts]/
│   └── this-is-my-visual-qa-ambiguity.json
│
└── Multi-concept/
    ├── train/
    │   └── [21 concepts]/
    ├── test/
    │   └── [11 concept pairs]/
    └── VQA/
        └── [VQA files for each multi-concept pair]
```
## Reference View Extraction

Extract reference view features from your dataset using the following command:
```bash
python extraction.py \
  --data_folder ./datasets/ \
  --dataset myvlm \
  --split train \
  --device_ids 0,1,2,3 \
  --n_training_views 5 \
  --variation augment \
  --n_augment 9 \
  --grounding_sam \
  --features_folder ./features/
```

### Arguments

| Argument | Type | Choices | Description |
|----------|------|---------|-------------|
| `--data_folder` | `str` | - | Path to your dataset directory |
| `--dataset` | `str` | `myvlm`, `yollava`, `this-is-my` | Dataset to process |
| `--split` | `str` | `train`, `test` | Data split (must be `train` for reference view extraction) |
| `--variation` | `str` | `normal`, `augment` | Feature extraction mode |
| `--n_augment` | `int` | - | Number of augmented views (only for `variation=augment`) |
| `--grounding_sam` | `flag` | - | Use Grounding SAM for mask extraction (omit to use Grounding DINO) |
| `--multi_concept` | `flag` | - | Process extended concepts (only for `this-is-my` dataset) |
| `--n_training_views` | `int` | - | Number of reference views to extract per concept |
| `--features_folder` | `str` | - | Directory to save extracted feature files |
| `--device_ids` | `str` | - | Comma-separated GPU device IDs (e.g., `0,1,2,3`) |

### Variation Modes

- **`normal`**: Extracts features only from the original reference views
- **`augment`**: Extracts features from both original and augmented reference views

### Example Usage

**Basic extraction with original views only:**
```bash
python extraction.py \
  --data_folder ./datasets/ \
  --dataset yollava \
  --split train \
  --variation normal \
  --n_training_views 3 \
  --features_folder ./features/
```

**Extraction with data augmentation:**
```bash
python extraction.py \
  --data_folder ./datasets/ \
  --dataset this-is-my \
  --split train \
  --variation augment \
  --n_augment 5 \
  --multi_concept \
  --features_folder ./features/
```



## 🚀 TODOs
### 🧑‍💻 Code Release

#### 🔹 Core Pipeline

- [ ] View Extraction  
  - [ ] Implement module for open-vocabulary segmentation (SAM / GroundedDINO)  
  - [ ] Extract patch-level features  
  - [ ] Save embedding vectors in memory  

- [ ] Retrieval System  
  - [ ] Thresholding logic & object matching  
  - [ ] Multiple concept detection  

- [ ] Prompting Integration  
  - [ ] LVLM input formatting  
  - [ ] Prompt templates for VQA & captioning  
  - [ ] Overlay generation for visual cues  
---

### 🧪 Evaluation & Benchmarks

- [ ] Scripts  
  - [ ] Eval suite for VQA  
  - [ ] Eval suite for captioning  
  - [ ] Ablation tests  

- [ ] Metrics  
  - [ ] Accuracy/Precision/recall for personalized retrieval  
  - [ ] Accuracy for Multiple-choice VQA
  - [ ] Accuracy for Open-ended VQA
  - [ ] Captioning Recall

---

## 📚 Citation

```bibtex
@article{seifi2025personalization,
  title={Personalization Toolkit: Training Free Personalization of Large Vision Language Models},
  author={Seifi, Soroush and Dorovatas, Vaggelis and Olmeda Reino, Daniel and Aljundi, Rahaf},
  journal={arXiv preprint arXiv:2502.02452},
  year={2025}
}
```

---