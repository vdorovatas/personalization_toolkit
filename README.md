📄 Paper Link: https://arxiv.org/abs/2502.02452

# 📌 Personalization Toolkit (PeKit)

## 📑 Sections
- 📁 📖 Overview  
- 🧠 Method Summary  
- 📂 Dataset: This-is-My-Img  
- ⚙️ Installation  
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

### 📦 Packaging & Deployment

- [ ] `setup.py` / `pyproject.toml` for pip installs  
- [ ] Example notebooks demonstrating end-to-end usage  

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

## 📌 Contribution Guidelines

- Code style & linting rules  
- Tests required for pull requests  
- Template issues for contributions  
