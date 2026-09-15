# Deep Learning from Scratch

PyTorch fundamentals through to semantic segmentation, with every claim measured rather
than assumed. All runs on Colab (free T4).

Background: PhD in satellite remote sensing (Antarctic sea ice), currently a postdoc at
CIMSS, UW–Madison. Strong in NumPy, image processing, geospatial data, classical ML,
and production pipelines. This repository closes the PyTorch gap.

## Week 1 — fundamentals to interpretation

| day | topic | headline result |
|---|---|---|
| [1–2](notebooks/day1_2_tensors_and_training_loop.ipynb) | tensors, autograd, training loop | recovered `y = 2x + 1` from noisy data; measured divergence under a too-large learning rate and the capacity ceiling of a linear model |
| [3](notebooks/day3_mnist_mlp.ipynb) | MLP on MNIST | 97.6% ± 0.15 test, +0.9 pt gap; a fixed pixel permutation costs **nothing** |
| [4](notebooks/day4_cifar10_cnn.ipynb) | CNN on CIFAR-10, from scratch | 76.6% test, **+21.1 pt gap**; augmentation → 81.9% test, +4.3 pt gap |
| [5](notebooks/day5_transfer_learning.ipynb) | transfer learning, ResNet-18 | **93.8% test** — 73% error reduction; a frozen backbone beat the from-scratch model while training 0.046% of the network |
| [6](notebooks/day6_validation_and_interpretation.ipynb) | validation discipline + Grad-CAM | 84.8% ± 0.4% with the test set used **once**; two errors, two opposite diagnoses |
| 7 | repository consolidation | README, structure, reproducible re-runs of Days 1–3 with outputs — no new notebook |

## Week 2 — semantic segmentation

| day | topic | headline result |
|---|---|---|
| [8](notebooks/day8_unet_skip_connections.ipynb) | U-Net and skip connections, LandCover.ai | removing skips changed pixel accuracy by **0.6%** while two entire classes went to **IoU 0.000** |

Days 4 and 6 use the same small CNN; Day 5 uses ResNet-18 at 224×224, so its number is
not comparable to the others. Day 6's lower figure than Day 5 reflects the smaller
architecture, not a regression. Week 2 moves to per-pixel prediction, where accuracy
and IoU diverge sharply.

## Five findings worth reading

**Pixel accuracy conceals the loss of entire classes.** On LandCover.ai, building
occupies 0.23% of pixels and woodland 60.89% — a 265:1 ratio. Ablating U-Net's skip
connections moved pixel accuracy from 0.8992 to 0.8933, a 0.6-point change that looks
like noise. Behind it, building and road IoU both went to exactly **0.000**: the model
never predicted a single pixel of either class across six epochs. Judged on accuracy,
skip connections would look irrelevant.
→ [Day 8, §4b](notebooks/day8_unet_skip_connections.ipynb)

**Skip connections supply localization, not capacity.** In the same ablation, large
classes barely moved (woodland 0.828 → 0.822, background 0.856 → 0.845) while small and
thin ones vanished. That *selectivity* rules out the 11% parameter difference between
the two models as an explanation — a uniform capacity reduction cannot make two specific
classes disappear while leaving the rest intact. Deep layers know *what*; shallow layers
know *where*; skips join them.
→ [Day 8, §4b](notebooks/day8_unet_skip_connections.ipynb)

**A fully-connected network on images uses no spatial structure at all.** Applying one
fixed random permutation to all 784 pixels — identically for every image, rendering them
unreadable to a human — moved MNIST test accuracy from 0.9756 to 0.9765, well inside the
±0.15 standard error. The MLP never used pixel adjacency, so destroying it costs nothing.
A CNN fails this same test. That contrast, not added complexity, is the argument for
CNNs on imagery.
→ [Day 3, §6](notebooks/day3_mnist_mlp.ipynb)

**A hyperparameter ranking that didn't survive a re-run.** Four regularization settings
spanned 1.70 points of validation accuracy. Re-running the apparent winner with the
*same seed* moved it 1.22 points and dropped it from first to third. Run-to-run noise
was comparable to the entire between-configuration spread, so the ranking was mostly
noise. Traced to cuDNN non-deterministic reduction kernels and per-worker DataLoader
RNG. This repository reports that rather than the ranking.
→ [Day 6, §4](notebooks/day6_validation_and_interpretation.ipynb)

**Two errors, two opposite diagnoses.** Grad-CAM showed that cat/dog confusion comes
from the model looking *at the animal* and failing to separate the classes at 32×32 — a
resolution limit. But bird→frog confusion comes from the model looking at *background
vegetation*: it learned "cluttered green → frog". That shortcut works on this test set
and would fail under distribution shift. Accuracy cannot distinguish the two cases, and
they call for opposite interventions.
→ [Day 6, §7](notebooks/day6_validation_and_interpretation.ipynb)

## Method notes

- **Test set used once.** From Day 6 onward: 45k train / 5k validation / 10k test.
  Model selection and early stopping run on validation only; the test set is touched
  once, at the end. Measured selection optimism was +0.34 points — smaller than the
  test set's own standard error, and reported as such rather than assumed away.
- **Error bars on accuracy.** Binomial standard errors are quoted with every headline
  number (±0.4 points at n=10,000), so differences can be judged against noise instead
  of eyeballed.
- **Controlled comparisons.** Identical seed and identical data split across arms;
  augmentation applied to the training transform only, with training accuracy measured
  through a clean loader so the gap compares like with like. In the Day 8 ablation,
  `use_skip` is a constructor flag so the two arms differ by exactly one boolean.
- **Segmentation metrics from an accumulated confusion matrix.** Per-batch IoU averaged
  over batches is wrong when a class is rare — batches containing no buildings
  contribute meaningless zeros. The confusion matrix is accumulated over the full
  loader and IoU computed once.
- **Checkpointing.** Best-by-validation weights are deep-copied to CPU and written to
  disk each time they improve — an in-memory `state_dict` holds GPU references that the
  next optimizer step overwrites.

## What this does not establish

- Hyperparameter rankings are within noise. No multi-seed study was run; `wd=5e-4` was
  carried forward as a reasonable choice, not a demonstrated winner.
- Grad-CAM evidence is 8 samples per confusion pair — a tendency, not a proof. An
  occlusion-sensitivity test would be needed to claim background dependence.
- Grad-CAM resolution is coarse (4×4 feature map upsampled 8×), reliable only at
  "on the subject vs off it" granularity.
- Day 8 trained for a fixed 6 epochs with no early stopping; mIoU was still rising for
  both arms, so those numbers are floors. Tiles were downsampled 512 → 256, which
  penalizes the small and thin classes most.
- Runs are not bitwise reproducible; see Day 6 §4.
- Results so far are on MNIST, CIFAR-10 and aerial RGB. Natural-image and RGB
  benchmarks are optimistic for satellite imagery: no canonical orientation, different
  texture statistics, and multispectral channel counts that ImageNet weights do not
  cover.

## Next

Remainder of Week 2: loss functions for imbalanced segmentation (Dice, class-weighted
cross-entropy), geospatial data pipelines, and **blocked spatial validation** —
neighbouring image patches are not independent, so random train/validation splits leak.

Weeks 3+: U-Net segmentation on Sentinel-1 SAR and AMSR2 passive microwave sea-ice data
(AI4Arctic), using a remote-sensing pretrained backbone (SSL4EO-S12 or Prithvi) rather
than ImageNet.

## Data

LandCover.ai (Days 8+) is licensed **CC-BY-NC-SA-4.0** — non-commercial, share-alike.
Source: <https://landcover.ai.linuxpolska.com/>. Please cite Boguszewski et al., CVPRW
2021 (doi:10.1109/CVPRW53098.2021.00121).

## Setup

```bash
pip install -r requirements.txt
```

Every notebook is self-contained and runs top to bottom on Colab with a T4. Datasets
download automatically on first run.
