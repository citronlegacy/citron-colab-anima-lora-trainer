# Citron Colab — Anima LoRA Trainer

A Google Colab notebook for training LoRA adapters for the [Anima](https://huggingface.co/circlestone-labs/Anima) diffusion model, powered by [kohya-ss/sd-scripts](https://github.com/kohya-ss/sd-scripts).



## Links

| | |
|:--|:--|
| 📦 **GitHub** | <a href="https://github.com/citronlegacy/citron-colab-anima-lora-trainer" target="_blank"><img src="https://img.shields.io/badge/GitHub-citron--colab--anima--lora--trainer-181717?logo=github" alt="GitHub"></a> |
| 🚀 **Open in Colab** | <a href="https://colab.research.google.com/github/citronlegacy/citron-colab-anima-lora-trainer/blob/main/ANIMA_Trainer_v4.ipynb" target="_blank"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab"></a> |
| 🤖 **CivitAI** | <a href="https://civitai.com/user/CitronLegacy/models" target="_blank"><img src="https://img.shields.io/badge/CivitAI-Models-blue.svg" alt="CivitAI"></a> |




---

## Features

- One-click setup: installs `sd-scripts` and downloads all three Anima model components automatically.
- Google Drive support: optional checkbox to mount Drive before downloading models, so outputs survive session resets.
- Automatic step estimator: calculates your total training steps from image count, repeats, and epochs — warns you before you start if you risk a Colab timeout.
- Live streaming output: training progress, loss values, and tqdm bars stream line-by-line to the Colab cell output.
- OOM diagnostics: on failure, prints the most recent kernel messages (`dmesg`) and detects Out-of-Memory kills with a tuning suggestion.
- TOML config generation: training and dataset configurations are written as `.toml` files and stored in `lora_training/configs/` for reproducibility.

---

## Requirements

| Requirement | Notes |
|---|---|
| Google Colab | Free or Pro. Pro recommended for longer runs. |
| GPU runtime | Required. Set Runtime → Change runtime type → T4 GPU or better. |
| ~6 GB VRAM | Default settings (dim=20, res=768) works on free-tier T4.  Reduce to dim=8 / res=512 for faster training. |
| Training images | Flat directory of images + matching `.txt` caption files. |

---

## Quick Start

1. Open [`ANIMA_Trainer_v4.ipynb`](ANIMA_Trainer_v4.ipynb) in Google Colab.
2. **Run cell 2 (Setup)** — installs dependencies and downloads the three Anima model files (~5.6 GB total). Enable `mount_drive` if you want to use Google Drive.
3. *(Optional)* **Run cell 3 (Unzip)** — unzip a dataset archive uploaded to `/content/`.
4. **Run cell 4 (Training Settings)** — set your project name, image directory, and hyperparameters. The cell will estimate your total training steps and warn you if they exceed 1000.
5. **Run cell 5 (Training)** — generates TOML configs and launches training via `accelerate`.

---

## ⚠️ Colab Session Limit

In testing, **1000 training steps takes over 4 hours** on a free-tier Colab T4 GPU. Colab sessions that exceed their runtime limit will disconnect and lose unsaved work.

**Keep total steps under 1000** for reliable single-session runs.

Steps formula:

```
steps_per_epoch = ceil( num_images × repeats / batch_size )
total_steps     = steps_per_epoch × epochs
```

The Training Settings cell calculates this automatically and prints suggested lower values for `epochs` and `repeats` if you are over the limit.

---

## Dataset Format

Use a **flat directory** (no subdirectories):

```
my_dataset/
  image001.png
  image001.txt      ← comma-separated tags
  image002.jpg
  image002.txt
  image003.webp
  image003.txt
```

Caption files must:
- Have the same base name as their image.
- Use the `.txt` extension.
- Contain comma-separated tag-style captions, e.g.:
  ```
  mycharname, 1girl, long blonde hair, blue eyes, smile, high quality
  ```

Supported image formats: `.jpg`, `.jpeg`, `.png`, `.webp`, `.bmp`, `.gif`

---

## Hyperparameter Defaults

| Parameter | Default | Notes |
|---|---|---|
| `network_dim` | 20 | LoRA rank. Lower (e.g. 8) uses less VRAM. |
| `network_alpha` | 20 | Usually equal to `network_dim`. |
| `learning_rate` | 0.0001 | Reduce if you see NaN loss. |
| `max_train_epochs` | 10 | Keep low to stay under 1000 total steps. |
| `resolution` | 768 | Reduce to 512 for OOM errors. |
| `repeats` | 5 | Images × repeats = steps per epoch. |
| `caption_dropout` | 0.1 | Probability of dropping a caption during training. |

---

## Google Drive Integration

Enable `mount_drive` in the Setup cell to mount your Drive at `/content/drive/MyDrive/` before model downloads. Recommended folder structure inside your Drive:

```
MyDrive/
  lora_training/
    datasets/     ← upload your zipped datasets here
    configs/      ← TOML configs are saved here automatically
    output/       ← trained LoRA .safetensors files are saved here
```

Set `image_directory` and `output_directory` in the Training Settings cell to paths under `/content/drive/MyDrive/lora_training/`.

---

## Trained Model Output

After training, the output directory will contain:

- `{project_name}.safetensors` — the final trained LoRA.
- `{project_name}-{epoch}.safetensors` — per-epoch checkpoints (last 4 kept by default).

Load the `.safetensors` file in any Anima-compatible inference pipeline as a LoRA adapter.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| CUDA out of memory | Reduce `network_dim` to 8 and/or `resolution` to 512. |
| NaN loss | Lower `learning_rate`. Ensure PyTorch ≥ 2.5. |
| "No images found" | Check that caption files are `.txt` and image filenames don't end in `.txt`. |
| Colab disconnects mid-run | Reduce total steps below 1000. Use Google Drive to save checkpoints. |
| Models missing on re-run | Re-run the Setup cell (models are in `/content/`, which resets each session unless saved to Drive). |

---

## Notebooks

| Notebook | Description |
|---|---|
| `ANIMA_Trainer_v4.ipynb` | **Current version.** Full pipeline with Drive support, step estimator, and live streaming output. |
| `ANIMA_Trainer_v2.ipynb` | Previous version. Full training pipeline, no step estimator. |
| `ANIMA_Trainer_v1.ipynb` | Initial version. Simpler structure, some duplicated function definitions. |

---

## License

This project is provided as-is for personal and research use. Refer to the individual model and sd-scripts licenses for usage terms:

- [Anima model license](https://huggingface.co/circlestone-labs/Anima)
- [kohya-ss/sd-scripts license](https://github.com/kohya-ss/sd-scripts/blob/main/LICENSE)
