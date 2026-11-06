# ARTI 502 – Deep Learning, Assignment 2
## Training and Testing CNNs on CIFAR-10 with PyTorch

Imam Abdulrahman Bin Faisal University – College of Computer Science & Information Technology – Year 2026, Term 1

**Group #:** 2

| # | Member Name | Academic ID |
|---|-------------|-------------|
| 1 | Sadeem Mohammed Alshashaee | 2230005681 |
| 2 | Mashael Saleh Alamer | 2230005887 |
| 3 | Fajr Abdulaziz Aldawsari | 2230005063 |
| 4 | Alia Ali Alharbi | 2230001846 |

## Repository contents

| File | Description |
|------|-------------|
| `ARTI_502_Assignment2.ipynb` | Full notebook (Tasks 1–6) |
| `requirements.txt` | Python dependencies |
| `task5_results.csv` | Results of all Task 5 experiments |
| `plot_*.png` | Training loss / test loss / test accuracy plots |

## How to run

```bash
pip install -r requirements.txt
jupyter notebook ARTI_502_Assignment2.ipynb   # then: Run All
```

- CIFAR-10 is downloaded automatically into `./data` (about 170 MB) the first time Task 1 runs.
- A GPU is recommended. Run time is roughly 40 minutes on a GPU; the augmented run is the slowest because augmentation runs on the CPU.
- If the DataLoader raises a pickling error on Windows, set `NUM_WORKERS = 0` in the Task 5 cell.

## Notebook structure

| Task | Content |
|------|---------|
| 1 | Download and load CIFAR-10 (50,000 train / 10,000 test images) |
| 2 | CNN definition (Conv2d + MaxPool2d + FC layers), `train_one_epoch` and `test` functions |
| 3 | Inspect the image tensor shape (`[64, 3, 32, 32]`) and visualize images with matplotlib |
| 4 | Data augmentation: ToPILImage, RandomHorizontalFlip, RandomResizedCrop, ColorJitter, RandomRotation, Gaussian noise, ToTensor, Normalize. Noise and random augmentation are applied to training data only; test data is only normalized |
| 5 | Training with loss/accuracy tracking and hyperparameter tuning across architectures, learning rates, batch sizes, optimizers and augmentation |
| 6 | Source code saved and uploaded to GitHub |

## Models

- **BaselineCNN** – 2 conv + 2 max-pool + 3 FC layers
- **WiderCNN** – 3 conv layers (32-64-128 filters) + max-pool + dropout
- **DeepBNCNN** – VGG-style double conv blocks with BatchNorm, max-pool and dropout

## Results (Adam unless noted, 10 epochs unless noted; final-epoch numbers)

| Experiment | Train loss | Test loss | Test acc (%) |
|------------|-----------:|----------:|-------------:|
| Baseline | 1.0818 | 1.1507 | 58.71 |
| Wider | 0.4695 | 0.7638 | 75.35 |
| DeepBN | 0.3873 | 0.5549 | 82.01 |
| Wider, lr=0.01 | 1.4969 | 1.4355 | 48.38 |
| Wider, lr=0.001 | 0.4620 | 0.8216 | 73.86 |
| Wider, lr=0.0001 | 1.1252 | 1.0818 | 61.87 |
| Wider, batch 32 | 0.4861 | 0.8222 | 73.70 |
| Wider, batch 64 | 0.4699 | 0.7995 | 74.21 |
| Wider, batch 128 | 0.5880 | 0.7311 | 74.81 |
| Wider, batch 256 | 0.7158 | 0.7980 | 72.24 |
| Wider, SGD (lr=0.01, momentum 0.9) | 0.5065 | 0.8152 | 73.20 |
| **DeepBN, 20 epochs (best)** | **0.1760** | **0.5496** | **84.89** |
| DeepBN, 20 epochs + normalization | 0.1814 | 0.6366 | 83.41 |
| DeepBN, 20 epochs + normalization + augmentation | 0.5661 | 0.4641 | 84.41 |

**Best configuration:** DeepBN, Adam, lr = 0.001, batch size 64, 20 epochs.

### Observations
- BatchNorm, depth and dropout (DeepBN) gave the largest improvement over the baseline.
- lr = 0.001 was best for Adam; 0.01 was unstable and 0.0001 too slow for 10 epochs.
- Batch sizes 32–128 performed similarly; differences of about 1% are within run-to-run noise (each configuration was run once).
- Augmentation reduced overfitting (lowest test loss, smallest train/test gap) but did not improve accuracy within 20 epochs.

### Limitations
- Each configuration was run once with a fixed seed.
- "Best test accuracy" in the CSV is chosen on the test set (no separate validation split), so the final-epoch numbers above are the ones to report.
