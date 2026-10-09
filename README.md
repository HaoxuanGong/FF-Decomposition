# Disentangling Objective Design and Gradient Locality in Supervised Forward--Forward Learning

This repository contains only the code required to reproduce the paper's MLP,
CNN, and MLP-Mixer experiments. The schedulers run one job at a time on one
CUDA GPU, record the complete protocol and source hashes, and resume only
artifacts that pass their integrity checks.

## Experiment matrix

| Architecture | Datasets | Methods | Selection | Runs |
|---|---|---|---|---:|
| MLP | MNIST, FashionMNIST, CIFAR-10, CIFAR-100 | 8 FF-derived variants and 3 CE references | thresholds 1/2/4, then constant Adam, cosine Adam, or step-decayed SGD | 144 threshold candidates, 360 optimizer candidates, 120 final tests |
| CNN | MNIST, FashionMNIST, CIFAR-10, CIFAR-100 | 3 CE references | cosine Adam or step-decayed SGD | 72 candidates, 36 final tests |
| MLP-Mixer | CIFAR-10, CIFAR-100, PathMNIST, TinyImageNet | 3 CE references | fixed D5-W256 AdamW protocol | 36 tests |

The exhaustive FC-FF variants are omitted on CIFAR-100, matching the paper.
The command-line method identifiers map to the manuscript notation as follows:

| Identifier | Paper notation |
|---|---|
| `ff` | Vanilla FF (Local, Multi-Head) |
| `ff-matched-ge` | FF (Global, Multi-Head) |
| `ff-ge` | FF (Global, Terminal) |
| `nn-ff-ge` | NN-FF (Global, Terminal) |
| `fc-ff` | FC-FF (Local, Multi-Head) |
| `fc-ff-matched-ge` | FC-FF (Global, Multi-Head) |
| `fc-ff-ge` | FC-FF (Global, Terminal) |
| `fc-nn-ff-ge` | FC-NN-FF (Global, Terminal) |
| `local-bp` | CE (Local, Multi-Head) |
| `ce-matched-ge` | CE (Global, Multi-Head) |
| `bp` | CE (Global, Terminal) |

## Setup

Run from the repository root on a Linux CUDA host:

```bash
conda env create -f environment.yml
conda activate ff-decomposition
python prepare_data.py --data-dir "$PWD/data"
```

The last command prepares the four torchvision datasets used by the MLP and
CNN studies. The Mixer scheduler prepares PathMNIST and TinyImageNet when they
are first needed.

## MLP experiments

Select the goodness threshold with constant Adam:

```bash
T="$PWD/runs/mlp-thresholds"
python mlp_threshold_sweep.py plan \
  --run-dir "$T" --project-root "$PWD" --python "$(command -v python)"
nohup python mlp_threshold_sweep.py launch --run-dir "$T" \
  > "$T/scheduler.log" 2>&1 &
echo $! > "$T/scheduler.pid"
python mlp_threshold_sweep.py status --run-dir "$T"
```

When the state is `training_complete`, freeze the validation-selected values:

```bash
python mlp_threshold_sweep.py select --run-dir "$T"
```

Run and finalize the optimizer sweep:

```bash
M="$PWD/runs/mlp"
python mlp_optimizer_sweep.py plan \
  --run-dir "$M" --project-root "$PWD" --python "$(command -v python)" \
  --thresholds-json "$T/selected_thresholds.json"
nohup python mlp_optimizer_sweep.py launch --run-dir "$M" \
  > "$M/scheduler.log" 2>&1 &
echo $! > "$M/scheduler.pid"
python mlp_optimizer_sweep.py status --run-dir "$M"
python mlp_optimizer_sweep.py finalize \
  --run-dir "$M" --data-dir "$PWD/data" --device cuda
python mlp_optimizer_sweep.py aggregate --run-dir "$M"
```

The paper-ready result table is `runs/mlp/summary.csv`.

## CNN experiments

```bash
C="$PWD/runs/cnn"
mkdir -p "$C"
nohup python cnn_optimizer_sweep.py \
  --run-dir "$C" --project-root "$PWD" --python "$(command -v python)" \
  > "$C/scheduler.log" 2>&1 &
echo $! > "$C/scheduler.pid"
cat "$C/status.json"
```

The scheduler performs validation-based profile selection and the final test
evaluations. Its paper-ready table is `runs/cnn/summary.csv`.

## MLP-Mixer experiments

```bash
X="$PWD/runs/mixer"
mkdir -p "$X"
nohup python mixer_benchmark.py \
  --run-dir "$X" --project-root "$PWD" --python "$(command -v python)" \
  > "$X/scheduler.log" 2>&1 &
echo $! > "$X/scheduler.pid"
cat "$X/status.json"
```

The paper-ready table is `runs/mixer/summary.csv`. Re-running any scheduler
with the same run directory resumes only results whose configuration, source,
checkpoint, and data-split evidence still match.
