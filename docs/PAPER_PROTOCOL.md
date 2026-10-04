# Paper protocol

This document fixes the code-to-paper mapping for **Disentangling Objective
Design and Gradient Locality in Supervised Forward--Forward Learning**. The
machine-readable version is [`configs/paper_protocol.json`](../configs/paper_protocol.json).

## MLP decomposition matrix

The paper contains eleven MLP variants in three supervision families.

| CLI ID | Paper label | Supervision | Gradient routing |
|---|---|---|---|
| `ff` | Vanilla FF (Local, Multi-Head) | thresholded goodness at every layer | local |
| `ff-matched-ge` | FF (Global, Multi-Head) | thresholded goodness at every layer | global |
| `ff-ge` | FF (Global, Terminal) | thresholded goodness at the final layer | global |
| `nn-ff-ge` | NN-FF (Global, Terminal) | final thresholded goodness, without hidden-layer normalization | global |
| `fc-ff` | FC-FF (Local, Multi-Head) | all-class goodness at every layer | local |
| `fc-ff-matched-ge` | FC-FF (Global, Multi-Head) | all-class goodness at every layer | global |
| `fc-ff-ge` | FC-FF (Global, Terminal) | final all-class goodness | global |
| `fc-nn-ff-ge` | FC-NN-FF (Global, Terminal) | final all-class goodness, without hidden-layer normalization | global |
| `local-bp` | CE (Local, Multi-Head) | cross-entropy at every layer | local |
| `ce-matched-ge` | CE (Global, Multi-Head) | cross-entropy at every layer | global |
| `bp` | CE (Global, Terminal) | final cross-entropy | global |

The extra IDs `ff-matched-local` and `ce-matched-local` are compatibility
controls used by implementation tests. They are outside the eleven-row paper
matrix and are not scheduled by the paper sweep.

Full-comparison methods are not run on CIFAR-100. Exhaustively scoring each
class requires 100 class-conditioned evaluations for every example, which
makes this family impractical for the intended experiment.

## MLP selection sequence

1. Use seeds 424, 425, and 426. Each seed defines its own stratified
   5,000-example validation holdout.
2. For thresholded FF methods, select from `1`, `2`, and `4` using validation
   performance under constant Adam, then freeze the threshold. Exact ties
   prefer the smaller threshold. `MLPThresholdSweepScheduler.py` writes the
   selected mapping and complete validation evidence without evaluating test.
3. Compare constant Adam, cosine-annealed Adam, and step-decayed SGD. Select
   one profile independently for each method and dataset using mean best
   validation accuracy across the three seeds.
4. Candidate runs save validation-selected checkpoints and defer the test
   split. Restore and evaluate only the selected profile on the official test
   set.

The three profiles are:

- Adam, learning rate `1e-3`, without a scheduler.
- Adam, learning rate `1e-3`, with cosine annealing (`T_max=200`,
  `eta_min=0`).
- SGD, learning rate `0.1`, momentum `0.9`, no Nesterov momentum, with StepLR
  (`step_size=30`, `gamma=0.1`).

All MLP runs use a maximum of 200 epochs, patience 15, batch size 128, strict
validation-accuracy improvements, no augmentation, no dropout, and no weight
decay. The terminal and multi-head MLP cross-entropy classifiers are
bias-free.

## CNN references

The CNN reference uses four convolutional blocks with 64, 128, 256, and 512
channels. Each block contains a bias-free 3x3 convolution, batch normalization,
ReLU, and 2x2 max pooling. The three methods are `local-bp`, `ce-matched-ge`,
and `bp`, corresponding to local multi-head, global multi-head, and global
terminal cross-entropy.

The paper scheduler compares cosine-annealed Adam with step-decayed SGD using
the same learning rates and scheduler parameters as the MLP profiles. It uses
the same four datasets, seeds, validation policy, epoch limit, patience, and
training batch size as the MLP benchmark. The terminal CNN classifier is
bias-free; the four auxiliary multi-head classifiers include bias terms.

## MLP-Mixer references

The paper MLP-Mixer is D5-W256: depth 5, embedding dimension 256, token hidden
dimension 256, and channel hidden dimension 1024. Patch sizes are 4 for
CIFAR-10/100, 7 for PathMNIST, and 8 for TinyImageNet. The active paper matrix
contains the same three cross-entropy methods as the CNN reference on
CIFAR-10, CIFAR-100, PathMNIST, and TinyImageNet.

The profile is fixed to AdamW with learning rate `3e-4`, weight decay `0.05`,
and cosine annealing over at most 200 epochs. The model uses dropout `0.1`,
automatic mixed precision, patience 15, batch size 128, and one local update
per block and minibatch. PathMNIST uses its official validation split; the
other datasets use a seed-specific stratified 10% training holdout.

CIFAR-10, CIFAR-100, and TinyImageNet training uses random cropping with
padding 4 and random horizontal flipping. PathMNIST training uses random
horizontal and vertical flips. Validation and test transforms contain no
random augmentation. The terminal Mixer classifier is bias-free, while the
multi-head classifiers include bias terms.

The general Mixer runner also supports COIL-100 and additional MedMNIST
datasets. Those datasets are extensions and are not part of the four-dataset
paper table.

## Interpretation of local/global rows

The local methods reproduce layer- or block-local training with independent
optimizers. Their global multi-head counterparts jointly optimize the sum or
mean of the head losses with one optimizer. The comparison therefore measures
the complete local-versus-global training formulations used in the paper; it
is not a literal single-line detach-only intervention. The run manifests
record optimizer topology, loss placement, inference aggregation, source
hashes, and dataset split identities so these differences remain auditable.
The paper-facing `ff` and `ff-matched-ge` rows also use their independently
validation-selected thresholds. Their numerical gap must therefore be
interpreted as a comparison of the two complete formulations, rather than as
an estimate attributable only to removing stop-gradients.

## Result provenance

The MLP and CNN optimizer-sweep schedulers separate validation-only candidates
from final test evaluation. The fixed-profile Mixer restores its best
validation checkpoint and evaluates the test split once within each run.
Existing outputs are reused only after the available configuration,
checkpoint, source, dataset metadata, and split-identity checks pass. Tiny
ImageNet additionally receives a content fingerprint.
`paper_results/reported_accuracy.csv` is a transcription of the aggregate
paper-table snapshot supplied during release preparation and is clearly
separated from newly generated run directories. It must be checked against the
final manuscript before a tagged release.

Profile ties use a fixed order rather than test accuracy: MLP prefers constant
Adam, then cosine Adam, then step-decayed SGD; CNN prefers cosine Adam over
step-decayed SGD. Test metrics never break a selection tie.
