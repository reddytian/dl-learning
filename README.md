# Deep Learning from Scratch — Week 1

PyTorch fundamentals through to model interpretation, with every claim measured rather
than assumed. Six days, five notebooks, all runs on Colab (free T4).

Background: PhD in satellite remote sensing (Antarctic sea ice), currently a postdoc at
CIMSS, UW–Madison. Strong in NumPy, image processing, geospatial data, classical ML,
and production pipelines. This repository closes the PyTorch gap.

## Results

| day | topic | headline result |
|---|---|---|
| [1–2](notebooks/day1_2_tensors_and_training_loop.ipynb) | tensors, autograd, training loop | recovered `y = 2x + 1` from noisy data; measured divergence under a too-large learning rate and the capacity ceiling of a linear model |
| [3](notebooks/day3_mnist_mlp.ipynb) | MLP on MNIST | 97.6% ± 0.15 test, +0.9 pt gap; a fixed pixel permutation costs **nothing** |
| [4](notebooks/day4_cifar10_cnn.ipynb) | CNN on CIFAR-10, from scratch | 76.6% test, **+21.1 pt gap**; augmentation → 81.9% test, +4.3 pt gap |
| [5](notebooks/day5_transfer_learning.ipynb) | transfer learning, ResNet-18 | **93.8% test** — 73% error reduction; a frozen backbone beat the from-scratch model while training 0.046% of the network |
| [6](notebooks/day6_validation_and_interpretation.ipynb) | validation discipline + Grad-CAM | 84.8% ± 0.4% with the test set used **once**; two errors, two opposite diagnoses |

Days 4 and 6 use the same small CNN; Day 5 uses ResNet-18 at 224×224, so its number is
not comparable to the others. Day 6's lower figure than Day 5 reflects the smaller
architecture, not a regression.

## Four findings worth reading

**A fully-connected network on images uses no spatial structure at all.** Applying one
fixed random permutation to all 784 pixels — identically for every image, rendering
them unreadable to a human — moved MNIST test accuracy from 0.9756 to 0.9765, a
difference well inside the ±0.15 standard error. The MLP never used pixel adjacency, so
destroying it costs nothing. A CNN fails this same test, because convolution's premise
is precisely that neighbouring pixels belong together. That contrast, not added
complexity, is the argument for CNNs on imagery.
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

**More trainable parameters, smaller generalization gap.** Fine-tuned ResNet-18 trains
17× more parameters than the from-scratch CNN, yet its gap is 4.5× smaller (+0.047 vs
+0.211). Generalization depends on where in parameter space the weights sit, not how
many there are — pretrained initialization is itself a form of regularization, and a
more effective one here than dropout or weight decay.
→ [Day 5, §6](notebooks/day5_transfer_learning.ipynb)

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
  through a clean loader so the gap compares like with like.
- **Checkpointing.** Best-by-validation weights are deep-copied to CPU and written to
  disk each time they improve — an in-memory `state_dict` holds GPU references that the
  next optimizer step overwrites.

## What this week does not establish

- Hyperparameter rankings are within noise. No multi-seed study was run; `wd=5e-4` was
  carried forward as a reasonable choice, not a demonstrated winner.
- Grad-CAM evidence is 8 samples per confusion pair — a tendency, not a proof. An
  occlusion-sensitivity test would be needed to claim background dependence.
- Grad-CAM resolution is coarse (4×4 feature map upsampled 8×), reliable only at
  "on the subject vs off it" granularity.
- Runs are not bitwise reproducible; see Day 6 §4.
- All results are on MNIST and CIFAR-10. Natural-image benchmarks are optimistic for
  overhead imagery: no canonical orientation, different texture statistics, and
  multispectral channel counts that ImageNet weights do not cover.

## Next

Weeks 2–4: U-Net semantic segmentation on public satellite imagery, using a
remote-sensing pretrained backbone (SSL4EO-S12 or Prithvi rather than ImageNet).
Spatial autocorrelation makes random train/validation splits leaky — neighbouring
patches are not independent — so blocked or geographically separated splitting is
required. The validation discipline from Day 6 gets harder there, not easier.

## Setup

```bash
pip install -r requirements.txt
```

Every notebook is self-contained and runs top to bottom on Colab with a T4. Datasets
download automatically on first run (CIFAR-10 is ~170 MB).
