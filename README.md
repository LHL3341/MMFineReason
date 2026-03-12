# MMFineReason: Closing the Multimodal Reasoning Gap via Open Data-Centric Methods

**[English](README.md)** | **[中文](README_zh.md)**

> [[Paper]](https://arxiv.org/abs/2601.21821) | [[Full Dataset (2.3M)]](https://huggingface.co/datasets/OpenDataArena/MMFineReason-Full-2.3M-Qwen3-VL-235B-Thinking) | [[SFT + RL (1.8M)]](https://huggingface.co/datasets/OpenDataArena/MMFineReason-1.8M) | [[SFT (586K)]](https://huggingface.co/datasets/OpenDataArena/MMFineReason-SFT-586K) | [[SFT (123K)]](https://huggingface.co/datasets/OpenDataArena/MMFineReason-SFT-123K)

This repository implements the complete data construction pipeline described in the MMFineReason paper, for building large-scale, high-quality multimodal reasoning datasets (1.8M samples, 5.1B solution tokens). The pipeline consists of four stages: **Collection → Cleaning → Distillation → Selection**.

## Model Performance

<img src="assets/model_compare.png" alt="Model Comparison" width="700"/>

The figure above compares our models (Qwen3-VL-8B/32B-SFT) against mainstream multimodal models such as GPT-4V, Gemini Ultra, and Qwen-VL on the MMFineReason validation set. The SFT-finetuned Qwen3-VL-32B-Thinking achieves significant improvements in multimodal reasoning, even surpassing some closed-source commercial models — validating the effectiveness of large-scale open data with chain-of-thought annotations.

## Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        MMFineReason Pipeline                           │
│                                                                        │
│  ┌──────────┐    ┌──────────┐    ┌──────────────┐    ┌───────────┐    │
│  │Collection│───▶│ Cleaning │───▶│ Distillation │───▶│ Selection │    │
│  └──────────┘    └──────────┘    └──────────────┘    └───────────┘    │
│                                                                        │
│  HF Dataset       Text/Image      CoT Distill +      Quality +        │
│  Download         Cleaning         Caption Gen        Difficulty       │
│  (pre-processed)  Standardization  4-phase reasoning  Content/Struct   │
│                   Refinement       Visual description  N-gram dedup    │
│                   Suitability                          Consistency     │
│                                                        Difficulty      │
│                                                                        │
│  2.3M raw ──────────────────────────────────────────▶ 1.8M selected   │
└─────────────────────────────────────────────────────────────────────────┘
```

## Project Structure

```
mmfinereason/
├── README.md                          # This file
├── README_zh.md                       # Chinese version
├── requirements.txt                   # Dependencies
├── config.yaml                        # Configuration
├── mfr_pipeline.py                    # Main entry point (4-stage pipeline)
│
├── pipeline/
│   ├── __init__.py
│   ├── collection.py                  # Stage 1: Dataset download
│   ├── cleaning.py                    # Stage 2: Text + image cleaning
│   ├── distillation.py                # Stage 3: CoT distillation & caption
│   ├── selection.py                   # Stage 4: Quality & difficulty filtering
│   └── model_client.py                # Model client (OpenAI/vLLM compatible)
│
├── prompts/                           # Prompt templates
│   ├── distill.txt                    # CoT distillation prompt
│   ├── caption.txt                    # Image captioning prompt
│   ├── clean.txt                      # Data cleaning prompt
│   ├── answer_extraction.txt          # Answer extraction prompt
│   └── verify.txt                     # Correctness verification prompt
│
└── train/                             # Training configs
    └── config/
        └── mfr_8b.yaml               # Qwen3-VL-8B SFT config
```

## Pipeline Stages

### Paper-to-Code Mapping

| Paper Section | Pipeline Stage | Code Module |
|---------------|----------------|-------------|
| Section 3.1 - Data Collection | Stage 1: Collection | `pipeline/collection.py` |
| Section 3.1 - Data Cleaning & Image Cleaning | Stage 2: Cleaning | `pipeline/cleaning.py` |
| Section 3.2 - Data Annotation | Stage 3: Distillation | `pipeline/distillation.py` |
| Section 3.3 - Data Selection | Stage 4: Selection | `pipeline/selection.py` |

### Stage 1: Collection

> Corresponds to Section 3.1: Data Collection

Since the data curation and standardization process is highly fragmented across sources, this codebase directly downloads the pre-processed [MMFineReason-Full](https://huggingface.co/datasets/OpenDataArena/MMFineReason-Full-2.3M-Qwen3-VL-235B-Thinking) dataset from HuggingFace, which already includes cleaned questions, images, and extracted answers.

**Data Sources:**

| Category | Datasets | Samples |
|----------|----------|---------|
| **Mathematics (79.4%)** | MMR1, WaltonColdStart, ViRL39K, Euclid30K, MMK12, Geo170K, Geo3K, mm-openr1, WeMath series | ~1.41M |
| **Science (13.8%)** | VisualWebInstruct, BMMR, TQA, AI2D, Zebra-CoT, ScienceQA | ~244K |
| **Puzzle/Game (4.6%)** | GameQA-140K, Raven, VisualSphinx, PuzzleQA | ~82K |
| **General/OCR (2.2%)** | LLaVA-CoT | ~39K |

**Features:**
- Download MMFineReason-Full from [HuggingFace](https://huggingface.co/datasets/OpenDataArena/MMFineReason-Full-2.3M-Qwen3-VL-235B-Thinking)
- Subset selection (e.g., `--subsets BMMR Euclid30K`)
- Sample limiting for quick testing (e.g., `--max-samples 100`)

**Output:** `collected.parquet`

```bash
# Download specific subsets
python mfr_pipeline.py --step collection --output output/ --subsets BMMR Euclid30K

# Quick test (max 100 samples per subset)
python mfr_pipeline.py --step collection --output output/ --subsets BMMR --max-samples 100
```

---

### Stage 2: Cleaning

> Corresponds to Section 3.1: Data Cleaning + Image Cleaning

Performs comprehensive text and image cleaning on the collected data, ensuring language consistency, text cleanliness, and reasoning suitability.

**Text Cleaning:**
| Operation | Description |
|-----------|-------------|
| **Language Standardization** | Translate non-English text (e.g., Chinese in BMMR, Euclid30K) to English |
| **Noise Removal** | Remove URLs, corrupted characters, formatting artifacts, question numbers, score annotations |
| **Instruction Refinement** | Rewrite shallow reasoning prompts (e.g., "just give the answer" → "provide your answer after careful reasoning") |
| **Task Suitability Filtering** | Filter out coding, drawing, and other non-visual-reasoning tasks |

**Image Cleaning:**
| Operation | Description |
|-----------|-------------|
| Corrupt image removal | Discard unreadable or corrupted images |
| Resizing | Proportionally scale images with longest edge > 2048px |
| Color space normalization | Convert all images to RGB |

**Output:** `cleaned.parquet`

```bash
python mfr_pipeline.py --step cleaning --input output/collected.parquet --output output/
```

---

### Stage 3: Distillation

> Corresponds to Section 3.2: Data Annotation

Uses teacher models to generate high-quality chain-of-thought reasoning and visual descriptions for each sample. The HuggingFace dataset already includes distillation results; this stage skips samples with existing data by default.

**CoT Distillation (Core):**
- **Teacher Model:** `Qwen3-VL-235B-A22B-Thinking` (strongest open-source VLM)
- **Four-phase Reasoning Framework:**
  1. **Comprehensive Information Extraction** — Extract all visual information from the image
  2. **Strategic Problem Setup** — Understand the problem and determine the reasoning type
  3. **Rigorous Solution Execution** — Step-by-step derivation with visual elements as core components
  4. **Solution Validation** — Verify logical consistency of the answer
- **Output Format:** `<think>...</think>` + `<answer>...</answer>`
- **Key Design:** Captions are **not provided** during CoT distillation, ensuring the model reasons from visual information rather than taking text-only shortcuts

**Caption Generation:**
- **Model:** `Qwen3-VL-235B-A22B-Instruct`
- Generates structured dense descriptions: image type classification, global layout, symbolic elements, spatial relations, key visual cues
- Average 609 tokens/caption, 100% coverage

**Output:** `distilled.parquet` (adds `qwen3vl_235b_thinking_response` and `qwen3vl_235b_instruct_caption` fields)

```bash
python mfr_pipeline.py --step distillation --input output/cleaned.parquet --output output/
```

---

### Stage 4: Selection

> Corresponds to Section 3.3: Data Selection

Applies multi-stage filtering to select 1.8M high-quality samples from the 2.3M raw pool, with optional difficulty filtering for efficient training subsets.

#### 4a. Reasoning Quality Filtering → MMFineReason-1.8M

| Filter Step | Description |
|-------------|-------------|
| **Content Filter** | `[ERROR]` keyword check + response word count ≥ 200 + caption word count ≥ 100 |
| **Structure Validation** | Verify `<think>` / `<answer>` tag completeness (skip answer check for specific datasets) |
| **N-gram Deduplication** | Detect templated CoT with 50-gram repeating ≥ 3 times |
| **Correctness Verification** | Extract `<answer>` and compare with ground-truth, discard incorrect reasoning |

#### 4b. Difficulty Filtering → MMFineReason-123K / 586K

| Strategy | Description |
|----------|-------------|
| **Model:** `Qwen3-VL-4B-Thinking` | Generate 4 independent answers per question |
| **pass rate = 0** → MMFineReason-123K | Small model answers all incorrectly (hardest subset, only 7%) |
| **pass rate ≠ 1** → MMFineReason-586K | Small model fails to answer all correctly |

> **"Less is More" Finding:** Only 7% of the data (123K) achieves performance comparable to the full dataset.

**Output:** `selected.parquet`

```bash
# Quality filtering only
python mfr_pipeline.py --step selection --input output/distilled.parquet --output output/

# Quality + difficulty filtering (keep only pass_rate=0, hardest samples)
python mfr_pipeline.py --step selection --input output/distilled.parquet --output output/ \
    --difficulty 0.0
```

---

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Quick Test (BMMR subset, 100 samples)

```bash
# Download + quality filtering, no model API needed
python mfr_pipeline.py --step all --output output/ \
    --subsets BMMR --max-samples 100
```

### 3. Full Pipeline

```bash
# Edit config.yaml to set up model APIs (if distilling new data)
vim config.yaml

# Download full dataset and run all four stages
python mfr_pipeline.py --config config.yaml --step all --output output/
```

### 4. Run Individual Stages

```bash
# Stage 1: Collection — Download dataset from HuggingFace
python mfr_pipeline.py --step collection --output output/ --subsets BMMR Euclid30K

# Stage 2: Cleaning — Text/image cleaning
python mfr_pipeline.py --step cleaning --input output/collected.parquet --output output/

# Stage 3: Distillation — CoT distillation (skipped if HF data already has results)
python mfr_pipeline.py --step distillation --input output/cleaned.parquet --output output/

# Stage 4: Selection — Quality + difficulty filtering
python mfr_pipeline.py --step selection --input output/distilled.parquet --output output/

# Enable difficulty filtering (keep only pass_rate=0, hardest samples)
python mfr_pipeline.py --step selection --input output/distilled.parquet --output output/ \
    --difficulty 0.0
```

## Configuration

### config.yaml

The configuration file is organized by stage, with each stage's model clients and parameters in its own section:

```yaml
# Global
num_proc: 8

# Stage 1: Collection
collection:
  dataset_id: "OpenDataArena/MMFineReason-Full-2.3M-Qwen3-VL-235B-Thinking"
  subsets: [BMMR, Euclid30K]
  cache_dir: null

# Stage 2: Cleaning
cleaning:
  max_image_size: 2048
  llm_client:
    type: vllm
    api_base: "http://localhost:8000/v1"
    model: "Qwen3-30B-A3B-Thinking"

# Stage 3: Distillation
distillation:
  cot:
    client:
      type: vllm
      api_base: "http://localhost:8002/v1"
      model: "Qwen3-VL-235B-A22B-Thinking"
    temperature: 1.0
    top_p: 0.95
    max_tokens: 16384
    thinking: true
  caption:
    client:
      type: vllm
      api_base: "http://localhost:8001/v1"
      model: "Qwen3-VL-235B-A22B-Instruct"
    temperature: 1.0
    top_p: 0.95
    max_tokens: 16384

# Stage 4: Selection
selection:
  content_filter:
    min_response_words: 200
    min_caption_words: 100
    error_keyword: "[ERROR]"
  structure_filter:
    no_answer_check_datasets: [...]
  quality_filter:
    ngram_n: 50
    ngram_freq: 3
    consistency: true          # Whether to verify CoT vs ground-truth consistency
  difficulty_filter:
    threshold: null             # null=disabled, 0.0=hardest only, 0.5=moderately hard
    num_samples: 4
    client:
      type: vllm
      api_base: "http://localhost:8003/v1"
      model: "Qwen3-VL-4B-Thinking"
```

## Data Schema

The HuggingFace dataset includes the following fields:

| Category | Field | Description |
|----------|-------|-------------|
| **Metadata** | `source` | Source dataset |
| | `id` | Unique identifier |
| **Raw Data** | `original_question` | Original question |
| | `original_answer` | Original answer |
| **Input/Output** | `image` | Image |
| | `question` | Question |
| | `answer` | Standardized answer |
| **Augmented** | `qwen3vl_235b_instruct_caption` | Dense image caption |
| | `qwen3vl_235b_thinking_response` | CoT reasoning trace |
| **Metrics** | `qwen3vl_4b_pass_rate` | Difficulty score (0–1) |
| | `is_consistent` | Correctness verification result |
| | `consistency_analysis` | Consistency analysis |

## Training

We use [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) for SFT training:

```bash
# Qwen3-VL-8B SFT
llamafactory-cli train train/config/mfr_8b.yaml
```

Key training settings:
- Base model: `Qwen/Qwen3-VL-8B-Instruct`
- Freeze Vision Tower and Projector, full fine-tuning of LLM
- DeepSpeed ZeRO-2, Flash Attention 2
- Learning Rate: 1e-5, Cosine Scheduler, 10% Warmup

## Citation

```bibtex
@article{lin2026mmfinereason,
  title={MMFineReason: Closing the Multimodal Reasoning Gap via Open Data-Centric Methods},
  author={Lin, Honglin and Liu, Zheng and Zhu, Yun and Qin, Chonghan and Lin, Juekai and Shang, Xiaoran and He, Conghui and Zhang, Wentao and Wu, Lijun},
  journal={arXiv preprint arXiv:2601.21821},
  year={2026}
}
```

## License

This project is for research purposes only. Please comply with the original licenses of all data sources.
