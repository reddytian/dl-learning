# Deep Learning from Scratch

PyTorch fundamentals through to semantic segmentation on geospatial data, with every
claim measured rather than assumed.

Background: PhD in satellite remote sensing (Antarctic sea ice), currently a postdoc at
CIMSS, UW–Madison. Strong in NumPy, image processing, geospatial data, classical ML,
and production pipelines. This repository closes the PyTorch gap.

Days 1–8 ran on Colab (free T4); Days 9–12 run locally on Apple Silicon (MPS).

## Week 1 — fundamentals to interpretation

| day | topic | headline result |
|---|---|---|
| [1–2](notebooks/day1_2_tensors_and_training_loop.ipynb) | tensors, autograd, training loop | recovered `y = 2x + 1` from noisy data; measured divergence under a too-large learning rate and the capacity ceiling of a linear model |
| [3](notebooks/day3_mnist_mlp.ipynb) | MLP on MNIST | 97.6% ± 0.15 test, +0.9 pt gap; a fixed pixel permutation costs **nothing** |
| [4](notebooks/day4_cifar10_cnn.ipynb) | CNN on CIFAR-10, from scratch | 76.6% test, **+21.1 pt gap**; augmentation → 81.9% test, +4.3 pt gap |
| [5](notebooks/day5_transfer_learning.ipynb) | transfer learning, ResNet-18 | **93.8% test** — 73% error reduction; a frozen backbone beat the from-scratch model while training 0.046% of the network |
| [6](notebooks/day6_validation_and_interpretation.ipynb) | validation discipline + Grad-CAM | 84.8% ± 0.4% with the test set used **once**; two errors, two opposite diagnoses |
| 7 | repository consolidation | README, structure, reproducible re-runs of Days 1–3 with outputs — no new notebook |

## Week 2 — segmentation and geospatial method

| day | topic | headline result |
|---|---|---|
| [8](notebooks/day8_unet_skip_connections.ipynb) | U-Net and skip connections, LandCover.ai | removing skips changed pixel accuracy by **0.6%** while two entire classes went to **IoU 0.000** |
| [9](notebooks/day9_segmentation_losses.ipynb) | loss functions under class imbalance | plain cross-entropy scores the **highest pixel accuracy of five arms and a building IoU of exactly 0.0000** |
| [10](notebooks/day10_geospatial_pipeline.ipynb) | GeoTIFF → batches → georeferenced output | windowed reads 786× faster than full-scene; unmasked nodata shifts a band mean **19%**; naive tiled inference missed **10.9%** of valid data |
| [11](notebooks/day11_spatial_validation.ipynb) | spatial autocorrelation, blocked validation | random tile splitting inflates validation mIoU by **16.5 points**; blocked splitting by 4.4, with both models equal on held-out ground |
| [12](notebooks/day12_pretrained_encoders.ipynb) | pretrained encoders, multispectral channels | dropping NIR to fit a 3-channel pretrained layer costs **27 points**; ImageNet pretraining itself gave **no measurable benefit** |

Numbers are comparable only within a notebook. Days 4 and 6 share a small CNN; Day 5
uses ResNet-18 at 224×224; Days 8–9 use LandCover.ai at different training-set sizes;
Days 10–12 use synthetic scenes whose absolute scores mean nothing outside themselves.

## Six findings worth reading

**Random tile splitting inflates validation by 16.5 points — and the honest pipeline
looks worse.** On a scene with a measured autocorrelation range of 360 px and 128 px
tiles, every validation tile in a random split had a training tile 128 px away: inside
the range, established before training anything. Against a geographically held-out
region, random-split optimism was +0.165 and blocked-split optimism +0.044, while both
models scored the same on the held-out ground (0.3638 vs 0.3725). The blocked pipeline's
validation score is **11 points lower** and marginally closer to the truth. Comparing
the two protocols on validation score alone selects the leaky one.
→ [Day 11](notebooks/day11_spatial_validation.ipynb)

**Keeping the bands beats keeping the pretrained weights.** Discarding NIR so a 4-band
stack fits ResNet-18's 3-channel `conv1` cost **27 points** of test mIoU — six to ten
standard deviations across three seeds. The three channel-adaptation strategies (RGB
only, re-initialize, inflate) were mutually indistinguishable, and so was the
from-scratch arm: ImageNet pretraining produced no measurable benefit on this data. The
notebook argues which parts of that negative result transfer to real imagery and which
are artifacts of a textureless synthetic scene.
→ [Day 12](notebooks/day12_pretrained_encoders.ipynb)

**Pixel accuracy conceals the loss of entire classes — and will select the worst
model.** On LandCover.ai, building occupies 0.23% of pixels and woodland 60.89%.
Ablating U-Net's skip connections moved pixel accuracy from 0.8992 to 0.8933, a change
that looks like noise; behind it, building and road IoU both went to exactly **0.000**.
A day later, across five loss functions, the arm with the **highest** pixel accuracy
(0.8677) was the only one with a building IoU of zero.
→ [Day 8](notebooks/day8_unet_skip_connections.ipynb), [Day 9](notebooks/day9_segmentation_losses.ipynb)

**Skip connections supply localization, not capacity.** In the same ablation, large
classes barely moved (woodland 0.828 → 0.822, background 0.856 → 0.845) while small and
thin ones vanished. That *selectivity* rules out the 11% parameter difference between
the two models as an explanation — a uniform capacity reduction cannot make two specific
classes disappear while leaving the rest intact. Deep layers know *what*; shallow layers
know *where*; skips join them.
→ [Day 8, §4b](notebooks/day8_unet_skip_connections.ipynb)

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

- **Test set used once.** From Day 6 onward, model selection and early stopping run on
  validation only. Measured selection optimism on CIFAR-10 was +0.34 points — smaller
  than the test set's own standard error, and reported as such rather than assumed away.
- **Spatial splitting from Day 11 onward.** Blocks, not shuffled tiles, with a buffer
  derived from an empirical semivariogram. The nearest-training-tile distribution is
  checked against the range *before* training, which costs seconds and settles whether
  a split leaks.
- **Statistics from training data only.** Per-band normalization constants are computed
  over training tiles. Scene-wide statistics leak test-region information into
  preprocessing even through a correct split.
- **Error bars, and a noise floor.** Binomial standard errors accompany headline
  accuracies. Day 6 measured ~1.2 points of run-to-run variance, and differences below
  that are reported as indistinguishable rather than ranked — including when the
  difference was the one a notebook was built to measure (Day 12).
- **Precision and recall alongside IoU.** IoU alone cannot separate "missed the class"
  from "over-predicted the class". Logging both is what exposed a design flaw in Day 9's
  combined loss (precision 0.202 against recall 0.692) and located it in a single line
  of code.
- **Segmentation metrics from an accumulated confusion matrix.** Per-batch IoU averaged
  over batches is wrong when a class is rare — batches containing no buildings
  contribute meaningless zeros. The matrix is accumulated over the full loader and IoU
  computed once.
- **Invalid pixels excluded twice.** `ignore_index` keeps nodata out of the loss;
  a separate filter keeps it out of the confusion matrix. Doing only the first leaves a
  phantom class in the metrics.
- **Georeferenced output.** Predictions are written back as GeoTIFFs inheriting the
  source CRS and affine transform, with nodata preserved. A prediction that has lost its
  projection is an array, not a product.
- **Checkpointing.** Best-by-validation weights are deep-copied to CPU and written to
  disk each time they improve — an in-memory `state_dict` holds GPU references that the
  next optimizer step overwrites.

## What this does not establish

- **Days 10–12 use synthetic scenes.** They exercise the pipeline, the split protocol
  and the channel-adaptation mechanics; their absolute scores mean nothing. Day 12's
  finding that ImageNet pretraining gave no benefit is **largely an artifact of a
  textureless synthetic scene** — the notebook says so and separates the parts that
  transfer from the parts that do not.
- **Mostly single-seed.** Only Day 12 ran a multi-seed study (three seeds, four arms),
  and it was run precisely because the central comparison fell inside the noise floor.
  Elsewhere, differences under ~1.2 points are not established.
- **Epoch caps throughout.** Days 4, 8, 9 and 12 all stopped on a fixed epoch budget
  with metrics still improving. Those numbers are floors.
- **Grad-CAM evidence is 8 samples per confusion pair** — a tendency, not a proof — at
  4×4 feature-map resolution upsampled 8×, reliable only at "on the subject vs off it"
  granularity.
- **Runs are not bitwise reproducible**; see Day 6 §4 for the diagnosis.
- **No real satellite imagery yet.** Everything so far is MNIST, CIFAR-10, aerial RGB,
  or synthetic multispectral. Natural-image and RGB benchmarks are optimistic for
  overhead data: no canonical orientation, different texture statistics, and
  multispectral channel counts that ImageNet weights do not cover.

## Next

A sea-ice segmentation project on Sentinel-1 SAR and AMSR2 passive microwave data
(AI4Arctic), applying the method established here: blocked spatial validation,
per-class IoU with precision and recall, nodata carried through loss and metrics, and
georeferenced output. A remote-sensing pretrained backbone (SSL4EO-S12 or Prithvi)
rather than ImageNet, for the reasons Day 12 measures.

## Data

LandCover.ai (Days 8–9) is licensed **CC-BY-NC-SA-4.0** — non-commercial, share-alike.
Source: <https://landcover.ai.linuxpolska.com/>. Please cite Boguszewski et al., CVPRW
2021 (doi:10.1109/CVPRW53098.2021.00121).

Days 10–12 generate their own synthetic GeoTIFFs; nothing to download.

## Setup

```bash
pip install -r requirements.txt
```

Days 1–8 run top to bottom on Colab with a T4; datasets download on first run. Days
9–12 are written for a local environment — they select `cuda`, `mps` or `cpu` at
runtime, set `PYTORCH_ENABLE_MPS_FALLBACK` before importing torch, and use
`num_workers=0` and `pin_memory=False` for macOS. They run on Colab unchanged.
