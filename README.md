# LoRA: Low-Rank Adaptation of Large Language Models
### Replication and Extension Study
**DS 642 — Deep Learning | NJIT Spring 2026**

Based on: [Hu et al., 2021 — LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)

---

## Overview

Large pretrained models like GPT-3 (175B parameters) are expensive to fine-tune. Full fine-tuning updates every parameter and requires storing a full model copy per task. LoRA solves this by freezing the pretrained weights and injecting small trainable matrices alongside the attention layers.

The forward pass becomes:

```
h = W₀x + BAx
```

Where **B ∈ ℝᵈˣʳ** and **A ∈ ℝʳˣᵏ**, with rank **r ≪ d**. Only A and B are trained. W₀ never changes. At inference, B and A are merged back into W₀ with zero added latency.

This project replicates the key findings of the paper and extends them with a novel task switching demonstration.

---

## Repository Structure

```
├── lora_experiment_executed.ipynb      # Experiments 1 and 2 — executed on A100 (full dataset)
├── lora_task_switching_clean.ipynb     # Experiment 3 — GPT-2 task switching (novel extension)
├── LoRA_Project_Report.docx            # Full written report
└── README.md
```

---

## Experiments

### Experiment 1 — LoRA vs Full Fine-Tuning on SST-2

**Model:** RoBERTa-base (124.6M parameters)  
**Task:** SST-2 binary sentiment classification (GLUE benchmark)  
**Dataset:** 67,349 training / 872 validation examples  
**Hardware:** Tesla T4 GPU (Google Colab, 10k subset) + NVIDIA A100 (NJIT Wulver HPC, full dataset)  
**File:** `lora_experiment_executed.ipynb`

| Method | Trainable Params | Trainable % | Peak Memory | Train Time | Val Accuracy |
|---|---|---|---|---|---|
| Full Fine-Tuning | 124,647,170 | 100.00% | 2.437 GB | 6.1 min | 94.27% |
| LoRA (r=8) | 887,042 | 0.71% | 1.889 GB | 4.8 min | 92.09% |

**141x fewer trainable parameters. 22.5% less GPU memory. 21% faster training. 2.18% accuracy gap.**

The same experiment was first prototyped on a Tesla T4 GPU on Google Colab using a 10,000-example subset (`train_subset = 10_000`), then scaled to the full 67k dataset on an NVIDIA A100 on the NJIT Wulver HPC cluster (`train_subset = None`). The accuracy gap narrowed from 4.01% on the subset to 2.18% on the full dataset, consistent with the paper's findings.

---

### Experiment 2 — Rank Ablation Study

Replicates **Table 6** from Hu et al. (2021). Sweeps rank r ∈ {1, 2, 4, 8, 16} to verify the paper's claim that weight updates during fine-tuning have a low intrinsic rank.  
**File:** `lora_experiment_executed.ipynb`

| Rank r | Trainable Params | Val Accuracy | Peak Memory |
|---|---|---|---|
| 1 | 629K | 91.40% | 1.298 GB |
| 2 | 665.9K | 91.74% | 1.298 GB |
| 4 | 739.6K | 91.97% | 1.301 GB |
| 8 | 887K | 92.09% | 1.300 GB |
| 16 | 1,182K | 92.66% | 1.310 GB |
| Full FT | 124,647K | 94.27% | 2.437 GB |

Key finding: performance plateaus around r=8 and memory stays nearly flat across all ranks (1.298 GB at r=1 vs 1.310 GB at r=16). LoRA's memory footprint is dominated by the frozen base model, not the adapters.

---

### Experiment 3 — Novel Extension: GPT-2 Task Switching

Demonstrates LoRA's core deployment advantage: one frozen base model serving multiple tasks through interchangeable adapters.  
**File:** `lora_task_switching_clean.ipynb`

**Setup:**
- Base model: GPT-2 small (124M parameters, frozen throughout)
- Adapter 1: Trained on [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) — learns creative fiction writing
- Adapter 2: Trained on [AG News](https://huggingface.co/datasets/ag_news) — learns journalistic news writing
- LoRA config: r=8, alpha=16, target module: c_attn (GPT-2 combined QKV projection)

**Task switching is a single line:**
```python
switch_model.set_adapter('stories')   # creative fiction
switch_model.set_adapter('news')      # news writing
```

**Same prompt, different adapter, completely different output:**

```
Prompt: "The old lighthouse had been abandoned for years, but last night"

Stories Adapter:
  ...a little girl named Mia saw a light flicker in the window. She ran to
  tell her father, who smiled and said the lighthouse keeper had finally come home...

News Adapter:
  ...local authorities confirmed the structure remains decommissioned following
  a Coast Guard inspection. Residents reported unusual activity near the site...
```

**Storage comparison:**

| | Size |
|---|---|
| Full GPT-2 model copy (per task) | 548 MB |
| Stories LoRA adapter | ~3-6 MB |
| News LoRA adapter | ~3-6 MB |
| 10 tasks via Full Fine-Tuning | ~5.48 GB |
| 10 tasks via LoRA | ~548 MB + ~50 MB |

Over 90% storage reduction for multi-task deployment.

---

## How to Run

### Experiment 1 and 2

Open `lora_experiment_executed.ipynb` in Jupyter or Google Colab.

```python
# Install dependencies
pip install transformers peft datasets accelerate evaluate scikit-learn

# For Colab T4 (fast prototyping)
CONFIG['train_subset'] = 10_000

# For full dataset (Wulver A100 or similar)
CONFIG['train_subset'] = None
```

### Experiment 3

Open `lora_task_switching_clean.ipynb` in Google Colab with a T4 GPU.

```python
# Install dependencies
pip install torchao --upgrade
pip install transformers peft datasets accelerate evaluate scikit-learn ipywidgets
```

Run all cells in order. The final cell launches an interactive widget to try different prompts with both adapters.

---

## NJIT Wulver HPC (SLURM)

To run on the Wulver cluster:

```bash
#!/bin/bash -l
#SBATCH --partition=course_gpu
#SBATCH --account=2026-spring-ds-642-bader-nj349
#SBATCH --qos=course
#SBATCH --gres=gpu:a100_10g:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00

module load easybuild
module load GCCcore/13.3.0
module load Python/3.12.3
module load CUDA/12.6.0

pip install transformers peft datasets accelerate evaluate scikit-learn jupyter nbconvert --user -q

jupyter nbconvert --to notebook \
    --execute lora_experiment.ipynb \
    --output lora_experiment_executed.ipynb \
    --ExecutePreprocessor.timeout=7200
```

---

## Key Takeaways

- LoRA reduces trainable parameters by **141x** with only a **2.18%** accuracy gap on the full dataset
- The rank ablation confirms the paper's intrinsic rank hypothesis — performance plateaus at r=8, memory stays flat
- Task switching shows one frozen base model can serve multiple tasks with **~5 MB adapter swaps**
- LoRA introduces **zero inference latency** because adapters are merged into the base weights at deployment

---

## References

- Hu, E. et al. (2021). [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685). arXiv:2106.09685.
- Liu, Y. et al. (2019). [RoBERTa: A Robustly Optimized BERT Pretraining Approach](https://arxiv.org/abs/1907.11692).
- Brown, T. et al. (2020). [Language Models are Few-Shot Learners (GPT-3)](https://arxiv.org/abs/2005.14165).
- Mangrulkar, S. et al. (2022). [PEFT: State-of-the-art Parameter-Efficient Fine-Tuning](https://github.com/huggingface/peft).
- Wang, A. et al. (2019). [GLUE Benchmark](https://arxiv.org/abs/1804.07461).