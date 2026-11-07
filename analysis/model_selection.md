# Model selection for the baseline surrogate

This document records how the surrogate used in the rest of the project was chosen, with every number and
every calculation needed to check the decision. All runs use `src/train_baseline.py`.

**Summary.** Starting from the paper's architecture (3 hidden layers × 256 units, no regularisation,
validation MSE 0.0571), a deeper network with weight decay reaches **0.0493 ± 0.0008** (5 × 256,
weight decay 0.1), about **14 % lower**, far above the run-to-run noise (≈ 0.0009). Dropout and narrower
networks do not help. Two configurations are statistically indistinguishable (A: 5 × 256, wd 0.1 and
B: 6 × 256, wd 0.2); the simpler one, **A**, is the reference surrogate. B is kept for comparison because it
memorises its training data less (train/validation gap ×4.2 vs ×6.3 on average), which may matter for calibration.
One target, the minimum normalised magnetic gradient scale length, stays at R² ≈ 0.76 whatever the setting.
With 20 networks each (section 7), the validation MSE is 0.0508 (paper), **0.0449 (A)** and 0.0457 (B), and A
beats the paper's architecture on 11 of the 12 test metrics.

---

## 1. Protocol

### 1.1 Data and split (identical for every run)

- Data: paper baseline subset, **26,746 shapes** (NFP = 3, DESC or VMEC optimisations, 0.05 % tails trimmed
  per target), built by `src/data.py`.
- Split RAND, data seed 42: **19,258** training shapes, **2,139** validation shapes (10 % of the training part),
  **5,349** test shapes (20 %). The same shapes are in each set for every run of this document.
- Inputs: the 81 Fourier coefficients that vary; targets: the 12 metrics (QI residual as log10).
  Both are z-scored with training-set statistics: $z = (y - \mu_{train}) / \sigma_{train}$.

### 1.2 What is fixed

| element | value |
|---|---|
| loss | mean squared error over the 12 z-scored targets |
| optimiser | AdamW, learning rate $\eta = 10^{-3}$, $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\varepsilon = 10^{-8}$ (PyTorch defaults); with weight decay 0 this is Adam |
| batch size | 256, i.e. $\lceil 19\,258 / 256 \rceil = 76$ steps per epoch |
| learning-rate schedule | halved when the validation loss has not improved for 10 epochs |
| early stopping | stop after 30 epochs without improvement (max 500); keep the best epoch |
| activation | tanh |
| ensemble size during the search | $K = 3$ networks per run, network seeds $42 + 100 + 1000\,m + k$ ($m$ = `--member-seed`, $k = 0, 1, 2$) |

### 1.3 What varies

Depth (`--depth`), width (`--width`), dropout (`--dropout`), weight decay (`--weight-decay`).

### 1.4 Selection criterion

The criterion is the **ensemble validation MSE**: average the predictions of the $K$ networks,
$\bar y = \frac{1}{K}\sum_k \hat y^{(k)}$, then

$$\text{MSE}_{val} = \frac{1}{12\,N_{val}} \sum_{i=1}^{N_{val}} \sum_{j=1}^{12} \big(\bar y_{ij} - y_{ij}\big)^2, \qquad N_{val} = 2\,139 .$$

- **Not the test scores**: the test set is never used to choose; choosing on it would make the final test
  score optimistic.
- **Not the train/validation gap** ($\text{MSE}_{val}/\text{MSE}_{train}$): it describes memorisation, but a
  smaller gap can come with a worse validation error (see dropout below).
- Caveat: the validation set also drives early stopping, so the validation MSE is slightly optimistic. This
  affects all runs alike and does not bias the comparison.
- Caveat: in this average the gradient scale length weighs about 45 % of the total because its error is by far
  the largest ($0.51^2 \approx 0.26$ against 0.01–0.06 for the other targets). Section 6 checks that the gains
  hold on the other 11 targets.

---

## 2. Search, round by round (one run per setting, member seed 0)

Columns: ensemble MSE on training and validation sets, gap = validation / training, and the minimum test R²
over the 12 targets (displayed only, never used to choose). Parameter counts include biases.

### 2.1 Weight decay, first round (scale too small)

| run | train | validation | gap | R² min |
|---|---|---|---|---|
| reference 3 × 256 | 0.0117 | 0.0571 | ×4.9 | 0.736 |
| wd 1e-5 | 0.0117 | 0.0571 | ×4.9 | 0.736 |
| wd 1e-4 | 0.0117 | 0.0571 | ×4.9 | 0.736 |
| wd 1e-3 | 0.0118 | 0.0571 | ×4.8 | 0.736 |

No effect at all. Explanation: AdamW shrinks every weight by a factor $(1 - \eta\lambda)$ at each step. Over
$N$ steps, ignoring the gradients, $\theta_N \approx \theta_0\, e^{-\eta\lambda N}$. With $\eta = 10^{-3}$ and
$N \approx 76 \times 100 = 7\,600$ steps (≈ 100 epochs before early stopping):

| $\lambda$ | $\eta\lambda N$ | total shrink $1 - e^{-\eta\lambda N}$ |
|---|---|---|
| 1e-5 | 0.000076 | 0.01 % |
| 1e-4 | 0.00076 | 0.08 % |
| 1e-3 | 0.0076 | 0.76 % |
| 1e-2 | 0.076 | 7.3 % |
| 5e-2 | 0.38 | 32 % |
| 1e-1 | 0.76 | 53 % |
| 2e-1 | 1.52 | 78 % |
| 5e-1 | 3.8 | 98 % |

The first round was in the "no effect" zone. Values that do something lie between about 1e-2 and 1e-1, which
is also the usual range for AdamW (PyTorch default 1e-2). This is an order of magnitude only: the gradients
pull useful weights back, so the real shrink is smaller.

### 2.2 Dropout

| run | train | validation | gap | R² min |
|---|---|---|---|---|
| dropout 0.05 | 0.0184 | 0.0592 | ×3.2 | 0.755 |
| dropout 0.1 | 0.0268 | 0.0643 | ×2.4 | 0.753 |
| dropout 0.2 | 0.0449 | 0.0732 | ×1.6 | 0.743 |

The gap falls steadily (less memorisation) but the validation error rises: every dropout rate is worse than no
dropout (0.0571). Dropout is not used in the reference model (it will be used for MC Dropout later).

### 2.3 Weight decay, second round (right scale)

| run | train | validation | gap |
|---|---|---|---|
| wd 1e-2 | 0.0122 | 0.0569 | ×4.6 |
| wd 5e-2 | 0.0133 | 0.0562 | ×4.2 |
| wd 1e-1 | 0.0145 | 0.0550 | ×3.8 |
| wd 5e-1 | 0.0349 | 0.0597 | ×1.7 |

A U-shaped curve: the validation error decreases up to 1e-1, then rises at 5e-1, where the training error
triples (the network can no longer fit the data, i.e. underfitting).

### 2.4 Architecture

| run | parameters | train | validation | gap |
|---|---|---|---|---|
| 2 × 128 | 28,556 | 0.0282 | 0.0604 | ×2.1 |
| 3 × 128 | 45,068 | 0.0219 | 0.0576 | ×2.6 |
| 5 × 128 | 78,092 | 0.0168 | 0.0549 | ×3.3 |
| 6 × 128 | 94,604 | 0.0185 | 0.0562 | ×3.0 |
| 3 × 256 (paper) | 155,660 | 0.0117 | 0.0571 | ×4.9 |
| 4 × 256 | 221,452 | 0.0077 | 0.0546 | ×7.1 |
| 5 × 256 | 287,244 | 0.0067 | 0.0531 | ×7.9 |
| 6 × 256 | 353,036 | 0.0055 | 0.0528 | ×9.5 |
| 8 × 256 | 484,620 | 0.0056 | 0.0539 | ×9.6 |
| 3 × 512 | 573,452 | 0.0059 | 0.0563 | ×9.6 |

Depth helps up to 5–6 layers, then stops (8 layers); width helps much less (3 × 512); narrow deep networks
(× 128) are worse than their 256-unit counterparts. Since reducing capacity never helped and adding depth did,
the remaining error is dominated by **bias** (model too simple), not by variance.

### 2.5 Depth combined with weight decay

| run | train | validation | gap |
|---|---|---|---|
| 5 × 256, wd 1e-1 | 0.0077 | 0.0491 | ×6.3 |
| 5 × 256, wd 2e-1 | 0.0132 | 0.0493 | ×3.7 |
| 6 × 256, wd 1e-1 | 0.0070 | 0.0493 | ×7.1 |
| 6 × 256, wd 2e-1 | 0.0110 | 0.0483 | ×4.4 |

The combination gives the largest gain of the study (≈ 0.053 → ≈ 0.049). A plausible reading: a deeper network
has more capacity, hence a stronger tendency to memorise, which weight decay counteracts.

---

## 3. Measuring the run-to-run noise

### 3.1 Why `--member-seed` and not `--seed`

A validation score mixes two things: the effect of the setting, and the luck of the random initialisation and
batch order. To measure the luck alone, the same setting is retrained with other network seeds on **the same
data**.

| option | test shapes | validation shapes | network seeds |
|---|---|---|---|
| default (`--seed 42 --member-seed 0`) | draw with 42 | draw with 43 | 142, 143, 144 |
| `--seed 43` | **other shapes** | **other shapes** | 143, 144, 145 |
| `--member-seed 1` | same | same | **1142, 1143, 1144** |

`--seed` would also change which shapes are scored, mixing the luck of the networks with the difficulty of the
validation shapes, and making the scores incomparable with all previous runs. `--member-seed` changes only the
networks.

### 3.2 Formulas

For $n$ runs of one setting with validation scores $x_1, \dots, x_n$:

- mean: $\bar x = \frac{1}{n}\sum_i x_i$
- standard deviation (run-to-run noise), with $n - 1$ because the mean is estimated from the same data:
  $s = \sqrt{\frac{1}{n-1}\sum_i (x_i - \bar x)^2}$
- uncertainty of the mean: $s / \sqrt{n}$
- uncertainty of the difference between two settings A and B: $\sigma_{\Delta} = \sqrt{s_A^2/n_A + s_B^2/n_B}$
- decision rule: the difference is considered real if $|\bar x_A - \bar x_B| > 2\,\sigma_{\Delta}$
  (a deviation of more than about two standard deviations happens by chance roughly 5 times in 100 for
  Gaussian noise); between 1 and 2, the comparison is inconclusive.

The precision of $s$ itself is limited: its relative error is about $1/\sqrt{2(n-1)}$, i.e. ≈ 50 % for $n = 3$
and ≈ 35 % for $n = 5$. These measurements give the order of magnitude of the noise, which is enough to separate
clear cases from unclear ones.

### 3.3 Three runs per setting (member seeds 0, 1, 2)

**A = 5 × 256, wd 1e-1**: 0.0491, 0.0507, 0.0488.

- $\bar x_A = (0.0491 + 0.0507 + 0.0488)/3 = 0.04953$
- deviations: −0.00043, +0.00117, −0.00073; sum of squares $2.087 \times 10^{-6}$
- $s_A = \sqrt{2.087 \times 10^{-6} / 2} = 0.00102$

**B = 6 × 256, wd 2e-1**: 0.0483, 0.0491, 0.0476.

- $\bar x_B = 0.04833$
- deviations: −0.00003, +0.00077, −0.00073; sum of squares $1.127 \times 10^{-6}$
- $s_B = \sqrt{1.127 \times 10^{-6} / 2} = 0.00075$

**Comparison**

- difference: $0.04953 - 0.04833 = 0.00120$
- $\sigma_\Delta = \sqrt{0.00102^2/3 + 0.00075^2/3} = 0.00073$
- ratio $0.00120 / 0.00073 = 1.64$: **inconclusive** (Welch t-test p ≈ 0.18).

### 3.4 Five runs per setting (member seeds 0 to 4)

**A**: 0.0491, 0.0507, 0.0488, 0.0489, 0.0492.

- $\bar x_A = 0.04934$; deviations −0.00024, +0.00136, −0.00054, −0.00044, −0.00014;
  sum of squares $2.412 \times 10^{-6}$; $s_A = \sqrt{2.412 \times 10^{-6}/4} = 0.00078$

**B**: 0.0483, 0.0491, 0.0476, 0.0478, 0.0499.

- $\bar x_B = 0.04854$; deviations −0.00024, +0.00056, −0.00094, −0.00074, +0.00136;
  sum of squares $3.652 \times 10^{-6}$; $s_B = \sqrt{3.652 \times 10^{-6}/4} = 0.00096$

**Comparison**

- difference: $0.04934 - 0.04854 = 0.00080$
- $\sigma_\Delta = \sqrt{0.00078^2/5 + 0.00096^2/5} = 0.00055$
- ratio $0.00080 / 0.00055 = 1.45$: **inconclusive** (Welch t-test p ≈ 0.19).

With more runs, the difference shrank from 0.00120 to 0.00080 while its uncertainty shrank less: the behaviour
expected when an apparent difference is mostly chance. B is lower than A for 4 seeds out of 5, but the gap is
about 1.6 % of the validation error and reverses for seed 4.

### 3.5 Noise level used in the rest of the document

Pooling the two settings (5 runs each): $s = \sqrt{(4 \times 0.00078^2 + 4 \times 0.00096^2)/8} = 0.00087$.
We round up to **$\sigma \approx 0.0010$** to stay conservative (the larger of the two estimates).

---

## 4. Re-reading the single-run comparisons with this noise

For two settings run **once** each, each score carries its own noise, so
$\sigma_\Delta = \sqrt{\sigma^2 + \sigma^2} = \sqrt{2}\,\sigma \approx 0.0014$, and the threshold is
$2\,\sigma_\Delta \approx 0.0028$.

| comparison | from (validation) | to (validation) | difference | verdict |
|---|---|---|---|---|
| add depth | 3 × 256: 0.0571 | 5 × 256: 0.0531 | 0.0040 | real |
| add weight decay to the deep network | 5 × 256: 0.0531 | 5 × 256, wd 1e-1: 0.0491 | 0.0040 | real |
| paper → best model | 3 × 256: 0.0571 | B, mean of 5 runs: 0.0485 | 0.0086 | clearly real |
| weight decay alone on the paper network | 3 × 256: 0.0571 | 3 × 256, wd 1e-1: 0.0550 | 0.0021 | within noise |
| 5 vs 6 layers (no weight decay) | 5 × 256: 0.0531 | 6 × 256: 0.0528 | 0.0003 | within noise |

For "paper → best model" one side is a mean of 5 runs, so
$\sigma_\Delta = \sqrt{0.0010^2 + 0.0010^2/5} \approx 0.0011$ and the threshold is ≈ 0.0022; the difference is
about 8 times $\sigma_\Delta$.

---

## 5. Decision

- **Reference surrogate: A = 5 × 256, weight decay 0.1, no dropout.** A and B are indistinguishable
  (section 3.4), so the simpler model is chosen: 287,244 parameters instead of 353,036, faster to train, and the
  better-measured of the two ($s_A = 0.00078$).
- **B = 6 × 256, weight decay 0.2 is kept for comparison.** Same accuracy, but it memorises less
  (gap ×4.2 vs ×6.3 on average over the 5 runs: A 6.3, 5.6, 6.1, 6.8, 6.8; B 4.4, 4.2, 4.2, 4.2, 4.0). The uncertainty study will check whether stronger memorisation
  makes the ensemble over-confident.
- **The paper's architecture (3 × 256, no regularisation) is kept as a faithful reproduction.**

**Why A uses weight decay 0.1 and B uses 0.2.** These values come from single runs (section 2.5): for
5 × 256, 0.1 scored 0.0491 and 0.2 scored 0.0493; for 6 × 256, 0.1 scored 0.0493 and 0.2 scored 0.0483. The
numerically better value was taken for each depth, but both differences (0.0002 and 0.0010) are below the
single-run threshold of 0.0028. The choice of weight decay within [0.1, 0.2] is therefore **not demonstrated**,
and A and B differ by two settings at once (depth and weight decay). This does not affect the conclusion that
A and B are equivalent, but it does not allow attributing any difference between them to depth alone.

---

## 6. Check on all 12 targets

Test RMSE (z-scored) and test R² per target for the three final models, each an ensemble of $K = 20$ networks
(section 7). This check uses the test set to verify that the validation criterion is sensible, not to choose
between settings: it was looked at once, after the decision of section 5.

| target | paper RMSE | A RMSE | B RMSE | paper R² | A R² | B R² |
|---|---|---|---|---|---|---|
| aspect ratio | 0.102 | 0.087 | 0.083 | 0.990 | 0.992 | 0.993 |
| aspect ratio / edge iota | 0.240 | 0.223 | 0.221 | 0.936 | 0.945 | 0.946 |
| max elongation | 0.256 | 0.223 | 0.220 | 0.936 | 0.951 | 0.952 |
| axis iota / NFP | 0.202 | 0.187 | 0.191 | 0.954 | 0.961 | 0.959 |
| edge iota / NFP | 0.140 | 0.129 | 0.130 | 0.978 | 0.981 | 0.981 |
| axis mirror ratio | 0.156 | 0.135 | 0.132 | 0.976 | 0.982 | 0.983 |
| edge mirror ratio | 0.135 | 0.119 | 0.118 | 0.982 | 0.986 | 0.986 |
| average triangularity | 0.169 | 0.153 | 0.156 | 0.971 | 0.976 | 0.975 |
| vacuum well | 0.256 | 0.236 | 0.240 | 0.936 | 0.945 | 0.943 |
| **min. magnetic gradient scale length** | **0.492** | **0.490** | **0.493** | **0.757** | **0.759** | **0.756** |
| flux compression in bad curvature | 0.214 | 0.200 | 0.201 | 0.955 | 0.960 | 0.960 |
| log10 qi | 0.201 | 0.185 | 0.190 | 0.960 | 0.966 | 0.964 |

- **A vs paper:** 11 targets improve by 6.5 to 15 % in RMSE (aspect ratio: $(0.102 - 0.087)/0.102 = 15\,\%$;
  edge iota / NFP: $(0.140 - 0.129)/0.140 = 8\,\%$), and the gradient scale length is unchanged
  (0.492 → 0.490). The validation criterion and the test set agree.
- **A vs B:** differences of at most 0.005 in RMSE, in both directions (B better on 5 targets, A on 7):
  consistent with the two models being equivalent.
- **The gradient scale length** stays at R² ≈ 0.76 for every model and every setting of the search (0.73 to 0.76).
  Its error is probably dominated by noise or by information absent from the boundary coefficients (it is a
  **minimum** over the surface, which can jump when the location of the minimum moves), not by the model.
- **Gap with the paper's reported accuracy** (R² > 0.97 for all metrics): without the gradient scale length, the
  lowest R² is 0.936 (paper reproduction) and 0.945 (A); 3 targets of A remain below 0.96 (aspect ratio /
  edge iota, max elongation, vacuum well).

---

## 7. Final models

Commands (20 networks each, member seed 0, same split and validation set as the search):

```
uv run python src/train_baseline.py --n-members 20 --depth 5 --width 256 --weight-decay 1e-1 --tag _A_final
uv run python src/train_baseline.py --n-members 20 --depth 6 --width 256 --weight-decay 2e-1 --tag _B_final
uv run python src/train_baseline.py --n-members 20 --tag _paper
```

**Reproducibility.** Networks 0 to 9 of `_paper` stop at exactly the same epochs (114, 99, 104, 91, 99, 91, 113,
96, 109, 119) with the same validation MSE as the first 10-network run of the paper's architecture: same data
seed, same network seeds and same code give the same numbers on CPU.

### 7.1 Results

| model | params | epochs | train | val. (K = 20) | val. 1 network | gap | test R² min | test RMSE max |
|---|---|---|---|---|---|---|---|---|
| paper | 155,660 | 81–119 | 0.0109 | 0.0508 | 0.0748 | ×4.7 | 0.757 | 0.492 |
| **A** | 287,244 | 98–185 | 0.0071 | **0.0449** | 0.0619 | ×6.3 | 0.759 | 0.490 |
| B | 353,036 | 169–228 | 0.0109 | 0.0457 | 0.0577 | ×4.2 | 0.756 | 0.493 |

Paper = 3 × 256, A = 5 × 256 with weight decay 0.1, B = 6 × 256 with weight decay 0.2. All MSE are z-scored and
averaged over the 12 targets; "val. 1 network" is the mean over the 20 networks of each network's own validation
MSE; epochs per network; gap = validation / train ensemble MSE, e.g. A: $0.0449 / 0.0071 = 6.3$. Total training time on a laptop CPU:
about 19 min (paper), 57 min (A) and 62 min (B). Regularised networks train longer because weight decay slows
down memorisation, so the validation minimum comes later.

- **A vs paper:** $0.0508 - 0.0449 = 0.0059$, i.e. 12 % lower. At $K = 3$ the run-to-run noise was
  $\sigma \approx 0.0010$ and it can only be smaller with 20 networks: the gain is real.
- **A vs B:** $0.0457 - 0.0449 = 0.0008$, now in favour of A (it was in favour of B at $K = 3$, section 3.4).
  The sign flips with the networks drawn: A and B remain equivalent, as concluded in section 5.

### 7.2 How the error depends on the number of networks

For an ensemble of $K$ networks trained independently, the expected MSE splits into a part the average cannot
remove and a part that shrinks with $K$:

$$\text{MSE}_K = a + \frac{b}{K}$$

- $a$: bias of the architecture and noise in the data (what all networks get wrong together);
- $b$: variance between networks (where they disagree); averaging $K$ networks divides it by $K$.

**Fitting $a$ and $b$ from two points.** $K = 1$ is the mean validation MSE of a single network, $K = 20$ the
ensemble:

$$a + b = \text{MSE}_1, \qquad a + \frac{b}{20} = \text{MSE}_{20}
\quad\Rightarrow\quad b = \frac{\text{MSE}_1 - \text{MSE}_{20}}{1 - 1/20}, \qquad a = \text{MSE}_1 - b.$$

- **A:** $b = (0.0619 - 0.0449)/0.95 = 0.0179$; $a = 0.0619 - 0.0179 = 0.0440$.
- **B:** $b = (0.0577 - 0.0457)/0.95 = 0.0127$; $a = 0.0577 - 0.0127 = 0.0450$.
- **paper:** $b = (0.0748 - 0.0508)/0.95 = 0.0253$; $a = 0.0748 - 0.0253 = 0.0495$.

**Check against the $K = 3$ runs of the search** (prediction $a + b/3$ vs measurement):

| model | prediction $a + b/3$ | measured at $K = 3$ |
|---|---|---|
| A | $0.0440 + 0.0179/3 = 0.0500$ | $0.0493 \pm 0.0008$ (5 seeds) |
| B | $0.0450 + 0.0127/3 = 0.0492$ | $0.0485 \pm 0.0010$ (5 seeds) |
| paper | $0.0495 + 0.0253/3 = 0.0579$ | $0.0571$ (1 run) |

All three predictions fall within about one noise level of the measurements: the model $a + b/K$ describes the
data well.

**What it shows.**

- Beyond 20 networks the error can hardly decrease: for A, $b/20 = 0.0009$ is all that is left to gain, and
  $\text{MSE}_K$ never goes below $a = 0.0440$. The remaining error is bias (or noise in the data), not variance,
  as concluded in section 2.4.
- The regularised models have a lower floor $a$ (0.044–0.045 vs 0.0495): this is where their gain over the paper
  comes from.
- B's networks are individually better than A's (0.0577 vs 0.0619) but **disagree less** ($b$ = 0.0127 vs 0.0179),
  so averaging helps B less and both ensembles end at the same level. Since the ensemble's uncertainty is
  precisely the disagreement between networks, B should produce **narrower** uncertainty intervals than A. Whether
  they are too narrow (over-confident) is the first question of the uncertainty study.

The first 10 networks of each run form the $K = 10$ ensembles compared with the paper; all 20 are used to study
how calibration depends on $K$.

---

## 8. Limitations

- Noise was measured on 2 settings with 5 runs each; its own uncertainty is about ±35 %.
- About 25 settings were compared on the same validation set: the best ones benefit slightly from the luck of
  that set. The test set, untouched, gives the honest final score.
- The search covered depth, width, dropout and weight decay only; learning rate, batch size and activation were
  not tuned, and the paper's own tuned values are not published.
- All results are on the RAND split (in-distribution).
