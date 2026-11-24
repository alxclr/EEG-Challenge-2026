# EEG-Challenge-2026

Sleep-onset latency prediction from streaming Muse EEG, with a SleepFM-inspired CNN encoder, a GRU and an MLP head. This is my Track 3 (Sleep Onset) entry for the [EEG/EMG Foundation Challenge 2026](https://neural-interfaces26.github.io/).

The full write-up (challenge, architecture, training and results) is in [paper/best_model.pdf](paper/best_model.pdf).

## Task

Each recording is an overnight Muse headband session: 4 EEG channels at 128 Hz, cut into 5 s windows. For every window, the model predicts the time left until sleep onset (the first N2 epoch), capped at 600 s. It sees the recording only up to the current window.

Submissions are ranked by **W-bMAE**. The windows are grouped by distance to onset, the MAE is averaged in each group, and the groups are weighted by these values:

| Time to onset | 0–40 s | 40–90 s | 90–300 s | 300–600 s |
|---|---|---|---|---|
| Weight | 10 | 5 | 3 | 1 |

The score is computed per recording, then averaged across recordings.

## Model

For each 5 s window:

- **Tokenizer.** A small CNN, shared by all 4 channels, turns each z-scored channel (640 samples) into 64 features. It has 5 blocks of Conv1d (kernel 5, stride 2), BatchNorm, ELU and LayerNorm, followed by global average pooling and a linear layer. Output: 4 × 64 = 256 features.
- **Window statistics.** For each channel: the mean, the log standard deviation and log power in 11 frequency bands, from 0.5 to 64.5 Hz. Output: 4 × 13 = 52 features. Each statistic is taken relative to its running mean over the recording so far, so the model sees changes within a person rather than their absolute level.
- **GRU (32 units).** Takes the 256 + 52 features plus the elapsed time (309 inputs). Its state carries the memory of the recording from one window to the next.
- **MLP head (32 → 64 → 1).** Turns the GRU state into a time to onset between 0 and 600 s.

Inference runs in real time, one window at a time. The architecture diagram is in the [write-up](paper/best_model.pdf).

## Training

| | |
|---|---|
| Loss | L1 weighted per recording, matching W-bMAE |
| Dropout | 0.3, plus 15% of windows dropped |
| Augmentation | per-recording channel gain, random channel swap |
| Optimiser | AdamW, weight decay 0.05, gradient clipping at 1.0 |
| Learning rate | cosine decay from 1e-3 to 2e-5 |
| Batch | 8 whole recordings |
| Epochs | 40 |
| Compute | Google Colab T4 GPU, about 6 min per run |

## Results

Local validation on 88 recordings from held-out participants (not seen in training):

| Time to onset | 0–40 s | 40–90 s | 90–300 s | 300–600 s |
|---|---|---|---|---|
| MAE (s) | 12.4 | 31.0 | 78.6 | 263.6 |
| Bias (s) | +6.2 | −8.7 | −45.3 | −262.9 |

**W-bMAE: 37.6 s** (median 36.3 s across recordings).

Close to onset the model is precise: about 12 s of error in the last 40 s, where the metric weighs most. Far from onset it predicts sleep too early, a trade-off the low weight of that group allows.

For reference, the organisers' baselines on the public split score between 60.1 s and 72.9 s (constant 35 s, EEGNet, CTNet, fine-tuned REVE).

## Repository contents

The files use the same paths as the [official competition repository](https://github.com/neural-interfaces26/2026-competition), so you can copy them straight into it.

| File | What it is |
|---|---|
| [`tracks/sleep_onset/solvers/sleep_onset_gru.py`](tracks/sleep_onset/solvers/sleep_onset_gru.py) | The model and its training loop, as a Benchopt solver (`SleepOnset-GRU`) |
| [`tracks/sleep_onset/sleep_onset_gru.yml`](tracks/sleep_onset/sleep_onset_gru.yml) | Benchopt config: train, then score on the validation participants |
| [`tracks/sleep_onset/objective.py`](tracks/sleep_onset/objective.py) | Official objective with extra diagnostics: median and std of W-bMAE across recordings, and MAE, bias and window count per time-to-onset group. The ranked W-bMAE is unchanged. |
| [`tracks/sleep_onset/datasets/interaxon2026muse.py`](tracks/sleep_onset/datasets/interaxon2026muse.py) | Official Muse dataset with two added options: `eval_split` (`test` or `val`) and `train_split` (`train`, or `train+val` to train a final model on all training participants) |
| [`benchmark_utils/`](benchmark_utils/) | Shared competition utilities, plus `sleepedf.py`, a loader that turns PhysioNet Sleep-EDF into Muse-like windows (optional extra data, not used by this model) |
| [`paper/best_model.pdf`](paper/best_model.pdf) | The write-up |

## How to run it

### 1. Get both repositories

```bash
git clone https://github.com/neural-interfaces26/2026-competition.git
git clone https://github.com/alxclr/EEG-Challenge-2026.git
cd 2026-competition
```

### 2. Install the dependencies

Use Python 3.10 or newer. Install PyTorch for your machine first (see [pytorch.org](https://pytorch.org/get-started/locally/)), then:

```bash
pip install -r requirements.txt
```

This installs Benchopt (1.10 or newer), the NeuralBench data stack and the other packages the competition uses.

### 3. Copy this repository's files in

```bash
cp -R ../EEG-Challenge-2026/tracks ../EEG-Challenge-2026/benchmark_utils .
```

This overwrites the official `objective.py`, `interaxon2026muse.py` and `benchmark_utils/` with the versions described above.

### 4. Check that the setup works (no download)

```bash
benchopt run tracks/sleep_onset --config tracks/sleep_onset/starter.yml
```

### 5. Download and prepare the Muse data

```bash
benchopt prepare tracks/sleep_onset --config tracks/sleep_onset/sleep_onset_gru.yml
```

### 6. Train and evaluate

```bash
benchopt run tracks/sleep_onset --config tracks/sleep_onset/sleep_onset_gru.yml
```

This trains on the training participants, scores the 88 validation recordings and prints the W-bMAE with its breakdown per time-to-onset group. A GPU is strongly recommended.

Useful variants:

```bash
# More epochs
benchopt run tracks/sleep_onset --config tracks/sleep_onset/sleep_onset_gru.yml \
  -s "SleepOnset-GRU[n_epochs=60]"

# Score on the public warm-up test set instead of validation
benchopt run tracks/sleep_onset --config tracks/sleep_onset/sleep_onset_gru.yml \
  -d "Interaxon2026Muse[eval_split=test]"

# Final model: train on all training participants (train + val)
benchopt run tracks/sleep_onset --config tracks/sleep_onset/sleep_onset_gru.yml \
  -d "Interaxon2026Muse[train_split=train+val,eval_split=test]"
```

### 7. Package a submission

The training run writes `submission.py` and `weights.pt` to `tracks/sleep_onset/outputs/SleepOnset-GRU/`. Zip the two files, not the folder:

```bash
cd tracks/sleep_onset/outputs/SleepOnset-GRU
zip -j ../SleepOnset-GRU.zip submission.py weights.pt
cd -
```

To replay the packaged model on the data without retraining:

```bash
COMPET_SUBMISSION_DIR="$PWD/tracks/sleep_onset/outputs/SleepOnset-GRU" \
  benchopt run tracks/sleep_onset -d Interaxon2026Muse \
  -s "$PWD/tracks/sleep_onset/outputs/SleepOnset-GRU/submission.py"
```

See the competition's [Submission Guide](https://github.com/neural-interfaces26/2026-competition/blob/main/codabench/pages/participate.md) for the full contract.

## Next steps

- Reduce the systematic underestimate beyond 300 s before onset.
- Encourage predictions to decrease smoothly over time.
- Tune the tokenizer width (16 or 32 features per channel instead of 64). The tokens account for about half of the parameters.

## Acknowledgements

The model design is inspired by SleepFM. The competition code, data pipeline and metric are by the EEG/EMG Foundation Challenge 2026 organisers.
