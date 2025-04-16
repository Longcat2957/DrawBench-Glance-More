
# 🖼️ DrawBench Glance More

This project performs inference over the [DrawBench](https://arxiv.org/abs/2206.13353) prompt benchmark using recent various image generation models. It enables automated image generation, logging, and grid visualization of outputs by prompt category. 

---

## 🚀 Features

- 🔢 Inference using multiple SANA models (`1.6B`, `0.6B`, `4.8B`, etc.)
- 📊 Prompt categorization and controlled sampling
- 🧱 Grid visualization per category
- ✅ Supports all DrawBench categories or filtered categories
- 🔁 Reproducible via manual seed setting
- 🧩 Modular and extensible pipeline
- ⚡ Powered by [uv](https://github.com/astral-sh/uv) for ultra-fast Python dependency management

---

## 🛠️ Quickstart

### 1. Clone the repository

```bash
git clone https://github.com/Longcat2957/DrawBench-Glance-More
cd DrawBench-Glance-More
```

### 2. Sync dependencies with `uv`

```bash
uv sync
```

This will automatically install all required packages listed in `pyproject.toml`.

---

## 📦 Directory Structure

```
.
├── prompts/
│   └── DrawBench.csv               # Prompt file grouped by categories
├── outputs/
│   └── {model_name}/{category}/   # Generated images
│   └── {model_name}/{category}_grid.png
├── models/
│   └── sana.py                     # SANA model loading logic
├── utils/
│   ├── logger.py                   # Logging configuration
│   ├── misc.py                     # Utility: device, dtype
│   └── grid.py                     # Grid image generation
└── main.py                         # Entry-point for prompt-based inference
```

---

## 🧪 Example Usage

### 🔹 Basic usage

```bash
uv run python main.py 
```

### 🔹 Run all categories

```bash
uv run python main.py --all-categories --num 100
```

---

## 🧠 Supported Models

Update `--repo-id` to try different preconfigured SANA models:

| Model Repo ID | Inference Steps | Guidance Scale |
|---------------|------------------|----------------|
| `Efficient-Large-Model/Sana_Sprint_1.6B_1024px_diffusers` | 2 | — |
| `Efficient-Large-Model/Sana_Sprint_0.6B_1024px_diffusers` | 2 | — |
| `Efficient-Large-Model/SANA1.5_4.8B_1024px_diffusers` | 20 | 4.5 |
| `Efficient-Large-Model/SANA1.5_1.6B_1024px_diffusers` | 20 | 4.5 |

---

## 🖼️ Output Example

Each category generates:
- Individual images named with sanitized prompt text and seed
- A grid image with all prompts from the category (unless `--no_grid` is set)

📁 Example:
```
outputs/
└── Sana_Sprint_1.6B/
    ├── Colors/
    │   ├── red_frog_on_leaf_seed42.png
    │   ├── ... 
    └── Colors_grid.png
```

---

## 📚 Prompt File Format

`DrawBench.csv` should follow:

```csv
Category,Prompt
Colors,A red frog on a green leaf
Colors,A blue elephant flying
Positional,A dog under the table
...
```

You may replace or expand this file for your own benchmarks.

---

## 🧑‍💻 Contributing

We welcome contributions to:
- Extend model support
- Add prompt sets (e.g., HRS, TIFA, etc.)
- Improve UI or grid rendering
- Optimize inference pipeline

### Steps

1. Fork this repository
2. Create a feature branch
3. Open a pull request with clear description and sample output if applicable

---

## 📢 Acknowledgments

- The [DrawBench](https://arxiv.org/abs/2206.13353) benchmark
- [SANA](https://huggingface.co/Efficient-Large-Model) model family
- Contributors to the Diffusers ecosystem

---
