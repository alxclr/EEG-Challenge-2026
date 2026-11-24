"""Sleep-onset GRU: per-channel CNN tokenizer, GRU memory and MLP head.

The evaluation streams each recording one window at a time, forward in time,
on a fresh copy of the model (``reset_state`` first): a model may remember
the earlier windows of the recording it is predicting. This one does::

    window t --> [viewer] --> z_t --+
    elapsed time t * 5 s -----------+--> [GRU] --> h_t --> [MLP head] --> latency_t
                          h_{t-1} --+       |
                                            +--> kept for window t+1

- viewer (``Viewer``): a small CNN shared by every channel (inspired by
  SleepFM's tokenizer, 1 x 640 -> 64) on the z-scored signal, plus per
  channel the mean, log standard deviation and 11 log band powers, taken
  relative to their running mean over the recording so far;
- thinker (``Thinker``): a one-layer GRU over the windows seen so far in the
  recording, given the elapsed time too (the windows start at the
  recording's start);
- head (``Head``): an MLP from the GRU state to a latency in ``[0, 1]``
  (seconds over ``CAP_S``), mapped back to seconds in ``predict``;
- ``SleepOnsetModel`` chains the three and keeps the state between windows.

Training reads every train window once, in recording order (not through the
bin-balanced sampler, which draws single windows), and runs the GRU over
whole recordings from their start, as at evaluation. The loss is an L1
weighted per recording like the W-bMAE ranking metric.
"""

import time

import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from benchmark_utils.base_solver import CompetSolver

CAP_S = 600.0
#bins and severity weights as the objective's W-bMAE
BIN_EDGES = (40.0, 90.0, 300.0)
BIN_WEIGHTS = (10.0, 5.0, 3.0, 1.0)
SFREQ = 128.0
BANDS = ((0.5, 1.0), (1.0, 2.0), (2.0, 4.0), (4.0, 6.0), (6.0, 8.0),
         (8.0, 10.0), (10.0, 13.0), (13.0, 16.0), (16.0, 30.0),
         (30.0, 45.0), (45.0, 64.5))


class TokenizerBlock(nn.Module):
    """Conv1d(k=5, stride 2) -> BatchNorm -> ELU -> LayerNorm: halves T."""

    def __init__(self, n_in, n_out):
        super().__init__()
        self.conv = nn.Conv1d(n_in, n_out, kernel_size=5, stride=2, padding=2)
        self.bn = nn.BatchNorm1d(n_out)
        self.act = nn.ELU()
        self.ln = nn.LayerNorm(n_out)

    def forward(self, X):
        X = self.act(self.bn(self.conv(X)))
        return self.ln(X.transpose(1, 2)).transpose(1, 2)


class Tokenizer(nn.Module):
    """(N, 1, T) -> (N, widths[-1]): one embedding per channel window."""

    def __init__(self, widths=(4, 8, 16, 32, 64)):
        super().__init__()
        n_ins = (1,) + widths[:-1]
        self.blocks = nn.Sequential(
            *[TokenizerBlock(i, o) for i, o in zip(n_ins, widths)],
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )
        self.out = nn.Linear(widths[-1], widths[-1])

    def forward(self, X):
        return self.out(self.blocks(X))


class Viewer(nn.Module):
    """(N, C, T) windows -> (N, C * dim + 2 * C) features, one per window."""

    def __init__(self, n_chans, dim=64):
        super().__init__()
        self.tokenizer = Tokenizer(widths=(4, 8, 16, 32, dim))
        self.n_bands = len(BANDS)
        self.n_stats = (2 + self.n_bands) * n_chans
        self.stats_norm = nn.BatchNorm1d(self.n_stats)
        self.n_features = n_chans * dim + self.n_stats

    def forward(self, X):
        N, C, T = X.shape
        # Log band powers of the raw (unfiltered) window, 0.5 to 64 Hz.
        P = torch.fft.rfft(X - X.mean(dim=-1, keepdim=True), dim=-1)
        P = P.real ** 2 + P.imag ** 2                              # (N, C, F)
        f = torch.fft.rfftfreq(T, d=1.0 / SFREQ).to(X.device)
        bp = torch.stack([P[..., (f >= lo) & (f < hi)].sum(-1)
                          for lo, hi in BANDS], dim=-1)           # (N, C, B)
        bp = (bp + 1e-3).log().reshape(N, -1)
        mean = X.mean(dim=-1)                                     # (N, C)
        std = X.std(dim=-1, unbiased=False).clamp_min(1e-6)       # (N, C)
        X = ((X - mean[..., None]) / std[..., None]).clamp(-15.0, 15.0)
        # Every channel through the same tokenizer, then side by side.
        tokens = self.tokenizer(X.reshape(N * C, 1, T)).reshape(N, -1)
        stats = torch.cat([mean, std.log(), bp], dim=1)
        return tokens, stats


class Thinker(nn.GRU):
    """(R, L, n_in) window features in time order -> (R, L, hidden) states.

    A one-layer GRU: the memory of the model. Its state after window t sums
    up the recording so far and is carried to window t+1 (``h``).
    """

    def __init__(self, n_in, hidden=32):
        super().__init__(n_in, hidden, batch_first=True)


class Head(nn.Sequential):
    """(..., hidden) GRU states -> (...) latencies in ``[0, 1]`` (over CAP_S).

    An MLP read out at every window, with no memory of its own.
    """

    def __init__(self, hidden=32, width=64):
        super().__init__(
            nn.Linear(hidden, width),
            nn.ReLU(),
            nn.Linear(width, 1),
        )

    def forward(self, h):
        return torch.sigmoid(super().forward(h)).squeeze(-1)


class SleepOnsetModel(nn.Module):
    """Windows of one recording, in time order -> one latency per window:
    Viewer (each window) -> Thinker (memory) -> Head (latency)."""

    def __init__(self, n_chans, window_s, dim=64, hidden=32, dropout=0.3):
        super().__init__()
        self.window_s = window_s
        self.viewer = Viewer(n_chans, dim=dim)
        self.drop = nn.Dropout(dropout)
        # +1 input: the elapsed time at the window's start, over CAP_S. The
        # names gru/head keep the keys of the weights.pt trained before.
        self.gru = Thinker(self.viewer.n_features + 1, hidden)
        self.head = Head(hidden)
        self.reset_state()

    def reset_state(self):
        """Forget the recording: called before each one at evaluation."""
        self.h = None   # GRU state after the windows seen so far
        self.ssum = None  # running sum of the raw stats features
        self.step = 0   # number of windows seen so far

    def forward(self, X, lengths):
        """Whole recordings, from their start, for training.

        ``X`` holds their windows back to back ``(sum(lengths), C, T)``;
        returns the latencies in ``[0, 1]`` padded to ``(R, max(lengths))``.
        """
        # The viewer sees real windows only, so padding never reaches its
        # BatchNorms.
        tokens, stats = self.viewer(X)
        # Causal per-recording normalisation: stats minus their running mean
        # over the windows seen so far (removes participant/electrode level).
        rel = []
        for s in stats.split(lengths):
            cm = s.cumsum(0) / torch.arange(1, len(s) + 1,
                                            device=s.device)[:, None]
            rel.append(s - cm)
        stats = self.viewer.stats_norm(torch.cat(rel))
        z = torch.cat([tokens, stats], dim=1)
        if self.training:
            # Whole-window dropout: a window's features replaced by zeros
            # (artifact / lost-window robustness, forces use of history).
            z = z * (torch.rand(len(z), 1, device=z.device) > 0.15).float()
        z = self.drop(z).split(lengths)
        elapsed = [torch.arange(n, device=X.device)[:, None]
                   * (self.window_s / CAP_S) for n in lengths]
        seqs = pad_sequence([torch.cat([zi, ti], dim=1)
                             for zi, ti in zip(z, elapsed)],
                            batch_first=True)                     # (R, L, F)
        # Causal: the padding after a recording's end leaves its outputs be.
        out, _ = self.gru(seqs)                                   # (R, L, H)
        return self.head(out)                                     # (R, L)

    @torch.no_grad()
    def predict(self, X):
        """The next window(s) of the current recording, in time order."""
        self.eval()
        preds = []
        for x in X:                     # streamed evaluation: B = 1
            tok, st = self.viewer(x[None])
            self.ssum = st if self.ssum is None else self.ssum + st
            rel = st - self.ssum / (self.step + 1)
            z = torch.cat([tok, self.viewer.stats_norm(rel)], dim=1)  # (1, F)
            t = z.new_full((1, 1), self.step * self.window_s / CAP_S)
            out, self.h = self.gru(torch.cat([z, t], dim=1)[:, None], self.h)
            preds.append(self.head(out[:, -1]))
            self.step += 1
        return torch.cat(preds) * CAP_S  # (B,) seconds, in [0, CAP_S]


class Solver(CompetSolver):

    name = "SleepOnset-GRU"

    # Override from the command line: -s "SleepOnset-GRU[n_epochs=100]".
    parameters = {"n_epochs": [40]}

    lr = 1e-3
    records_per_batch = 8

    def load_model(self, meta):
        model = SleepOnsetModel(
            n_chans=meta["n_chans"],
            window_s=meta["n_times"] / float(meta["sfreq"]),
        ).to(self.device)
        # Inference-only runs (the platform's) load the shipped weights; a
        # training run starts from scratch (see torch_linear.py).
        weights = meta["submission_dir"] / "weights.pt"
        if weights.exists() and self.train_loader is None:
            model.load_state_dict(
                torch.load(weights, map_location=self.device))
        return model

    def read_recordings(self, train_loader):
        """Every train window once, grouped by recording, in time order."""
        # The train loader's sampler draws single windows, bin-balanced; a
        # plain pass over its dataset reads them all, in storage order.
        loader = DataLoader(train_loader.dataset, batch_size=256,
                            collate_fn=train_loader.collate_fn)
        Xs, ys, records, onsets = [], [], [], []
        for X, y, info in loader:
            Xs.append(X)
            ys.append(y.float())
            records.append(torch.as_tensor(info["record_id"]))
            onsets.append(torch.as_tensor(info["onset"], dtype=torch.float64))
        X, y = torch.cat(Xs), torch.cat(ys)
        records, onsets = torch.cat(records), torch.cat(onsets)
        # Sort by recording, then by onset within each recording.
        order = torch.argsort(onsets, stable=True)
        order = order[torch.argsort(records[order], stable=True)]
        X, y, records = X[order], y[order], records[order]
        lengths = torch.unique_consecutive(records, return_counts=True)[1]
        return list(X.split(lengths.tolist())), list(y.split(lengths.tolist()))

    @staticmethod
    def augment(X, lengths):
        """Per-recording, per-channel gain and offset (participant/electrode
        variability), plus channel dropout: a channel replaced by another."""
        R, (N, C, T) = len(lengths), X.shape
        rec = torch.repeat_interleave(
            torch.arange(R, device=X.device),
            torch.as_tensor(lengths, device=X.device))
        gain = torch.exp(0.5 * torch.randn(R, C, 1, device=X.device))[rec]
        X = X * gain
        drop = torch.rand(R, C, device=X.device) < 0.2
        src = torch.randint(0, C, (R, C), device=X.device)
        Xc = X.clone()
        for r in range(R):
            for c in torch.nonzero(drop[r]).flatten().tolist():
                if src[r, c] != c:
                    m = rec == r
                    Xc[m, c] = X[m, int(src[r, c])]
        return Xc

    def fit(self, model, train_loader):
        start = time.time()
        recs_X, recs_y = self.read_recordings(train_loader)
        print(f"[SleepOnset-GRU] read {len(recs_X)} recordings "
              f"({sum(len(y) for y in recs_y)} windows) "
              f"in {time.time() - start:.1f} s")

        opt = torch.optim.AdamW(model.parameters(), lr=self.lr,
                                weight_decay=5e-2)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=self.n_epochs, eta_min=self.lr * 0.02)
        edges = torch.tensor(BIN_EDGES, device=self.device)
        bin_weights = torch.tensor(BIN_WEIGHTS, device=self.device)
        model.train()
        for epoch in range(self.n_epochs):
            start = time.time()
            total, n = 0.0, 0
            perm = torch.randperm(len(recs_X)).split(self.records_per_batch)
            for idx in perm:
                # The recordings may sit on the CPU (newer loaders collate
                # there): bring each batch onto the model's device.
                X = torch.cat([recs_X[i] for i in idx]).to(self.device)
                lengths = [len(recs_y[i]) for i in idx]
                y = pad_sequence([recs_y[i] for i in idx],
                                 batch_first=True).to(self.device)
                mask = pad_sequence([torch.ones_like(recs_y[i]) for i in idx],
                                    batch_first=True).to(self.device)
                if len(X) < 2:
                    continue  # BatchNorm needs more than one window
                # Metric-matched weights: per recording, each non-empty bin
                # has total weight BIN_WEIGHTS[b] / sum of non-empty bins'
                # weights, shared by the bin's windows (W-bMAE per recording).
                b = torch.bucketize(y, edges, right=True)
                onehot = nn.functional.one_hot(b, 4).float() * mask[..., None]
                cnt = onehot.sum(1, keepdim=True)                # (R, 1, 4)
                denom = ((cnt > 0).float() * bin_weights).sum(-1)  # (R, 1)
                w = (onehot * (bin_weights / cnt.clamp_min(1))).sum(-1)
                w = w / denom.clamp_min(1e-6) * mask
                X = self.augment(X, lengths)
                err = (model(X, lengths) - y / CAP_S).abs()
                loss = (w * err).sum() / mask.new_tensor(len(idx))
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                total += loss.item() * len(idx)
                n += len(idx)
            sched.step()
            print(f"[SleepOnset-GRU] epoch {epoch + 1}/{self.n_epochs} | "
                  f"train error (weighted MAE): "
                  f"{total / max(n, 1) * CAP_S:.1f} s of latency | "
                  f"took {time.time() - start:.1f} s")
        model.reset_state()

    def save_model(self, model, path):
        torch.save(model.state_dict(), path / "weights.pt")
