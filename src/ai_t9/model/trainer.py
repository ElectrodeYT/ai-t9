"""PyTorch training for the DualEncoder model.

This module is intentionally isolated from the inference path — PyTorch is an
optional dependency (pip install ai-t9[train]).  The output is a .npz file
that the pure-NumPy DualEncoder can load without any ML framework.

Training objective: frequency-weighted negative sampling
  For each (context_words, target_word) pair drawn from the corpus:
    - Score the positive (target) word against the context embedding
    - Sample `neg_samples` negatives using a frequency-weighted distribution
      (f^0.75, Word2Vec-style) via torch.multinomial
    - Apply binary cross-entropy (positive=1, negatives=0)

torch.compile() (PyTorch 2.0+) handles graph capture, kernel fusion, and mixed
precision automatically — no manual CUDA graph wiring required.  Falls back
transparently on CPU or older PyTorch versions.
"""

from __future__ import annotations

import io
import math
import shutil
import time
from pathlib import Path

import numpy as np

from .vocab import Vocabulary
from .dual_encoder import DualEncoder


def _require_torch():
    try:
        import torch
        return torch
    except ImportError:
        raise ImportError(
            "PyTorch is required for training: pip install ai-t9[train]  "
            "or  pip install torch"
        )


def _resolve_device(torch, preference: str) -> "torch.device":
    """Pick the best available compute device.

    preference values:
      "auto"  – CUDA if available, then MPS (Apple Silicon), then CPU
      "cuda"  – CUDA; raises if not available
      "mps"   – MPS; raises if not available
      "cpu"   – always CPU
    """
    if preference == "cpu":
        return torch.device("cpu")
    if preference == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but no CUDA device found.")
        return torch.device("cuda")
    if preference == "mps":
        if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            raise RuntimeError("MPS requested but not available.")
        return torch.device("mps")
    # auto
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Corpus iterator helpers
# ---------------------------------------------------------------------------

def _brown_sentence_ids(vocab: Vocabulary) -> list[list[int]]:
    """Return all Brown corpus sentences as lists of word IDs.

    UNK tokens (ID 0) are excluded so they never appear as training
    targets or pollute context windows.
    """
    import nltk
    try:
        nltk.data.find("corpora/brown")
    except LookupError:
        nltk.download("brown", quiet=True)
    from nltk.corpus import brown
    sentences = []
    unk = vocab.UNK_ID
    for sent in brown.sents():
        ids = [vocab.word_to_id(w.lower()) for w in sent if w.isalpha()]
        ids = [wid for wid in ids if wid != unk]
        if len(ids) >= 2:
            sentences.append(ids)
    return sentences


def _corpus_file_sentence_ids(path: Path, vocab: Vocabulary) -> list[list[int]]:
    """Read a plain-text file (one sentence per line) and convert to word IDs.

    UNK tokens (ID 0) are excluded so they never appear as training
    targets or pollute context windows.
    """
    sentences = []
    unk = vocab.UNK_ID
    with path.open(encoding="utf-8", errors="ignore") as f:
        for line in f:
            words = line.strip().lower().split()
            ids = [vocab.word_to_id(w) for w in words if w.isalpha()]
            ids = [wid for wid in ids if wid != unk]
            if len(ids) >= 2:
                sentences.append(ids)
    return sentences


def _precompute_pairs(
    sentences: list[list[int]],
    context_window: int,
    verbose: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Pre-compute all (context, target) training pairs as flat NumPy arrays.

    Called once before training.  The resulting arrays are shuffled per epoch
    via NumPy index permutation (fast, releases the GIL).

    Pairs where the target is UNK (ID 0) are skipped — they teach nothing
    useful and waste gradient signal.  (UNK tokens should already be stripped
    from sentences by the corpus loaders, but this is a safety net.)

    Returns:
        ctx_arr: int64 (n_pairs, context_window) — zero-padded on the left
        pos_arr: int64 (n_pairs,)
    """
    _UNK = 0
    n_pairs = sum(max(0, len(s) - 1) for s in sentences)
    ctx_arr = np.zeros((n_pairs, context_window), dtype=np.int64)
    pos_arr = np.empty(n_pairs, dtype=np.int64)
    if verbose:
        print("Precomputing training pairs...")
        progress_interval = max(1, n_pairs // 100)  # update every ~1%
    idx = 0
    for sent in sentences:
        for t in range(1, len(sent)):
            if sent[t] == _UNK:
                continue  # skip UNK targets
            ctx_start = max(0, t - context_window)
            src = sent[ctx_start:t]
            ctx_arr[idx, context_window - len(src):] = src
            pos_arr[idx] = sent[t]
            idx += 1
            if verbose and idx % progress_interval == 0:
                frac = idx / n_pairs
                bar_w = 20
                filled = int(bar_w * frac)
                bar = "\u2588" * filled + "\u2591" * (bar_w - filled)
                print(f"\r  |{bar}| {idx}/{n_pairs} pairs", end="", flush=True)
    if verbose:
        print()  # newline after progress bar
    # Truncate to actual count (idx < n_pairs when UNK targets were skipped)
    return ctx_arr[:idx], pos_arr[:idx]


def save_pairs(
    sentences: list[list[int]],
    context_window: int,
    vocab_size: int,
    path: str | Path,
    verbose: bool = False,
) -> int:
    """Precompute training pairs from sentences and persist them to a .npz file.

    The file stores arrays as int32 (half the size of int64) and embeds
    ``context_window`` and ``vocab_size`` as metadata so ``load_pairs()``
    can detect stale files built from a different vocab or window setting.

    Returns the number of pairs written.

    Typical workflow::

        # Step 1 — CPU-heavy, run once locally or on a cheap instance:
        sentences = _corpus_file_sentence_ids(corpus_path, vocab)
        save_pairs(sentences, context_window=3, vocab_size=vocab.size,
                   path="data/pairs.npz", verbose=True)

        # Step 2 — GPU-heavy, run on Modal / cloud GPU:
        trainer = DualEncoderTrainer(vocab, ...)
        trainer.train_from_pairs_file("data/pairs.npz", epochs=5)
    """
    ctx_arr, pos_arr = _precompute_pairs(sentences, context_window, verbose=verbose)
    # Build the .npz entirely in memory first, then write sequentially.
    # np.savez seeks to random offsets when given a file path, which is
    # incompatible with S3 CloudBucketMounts (Mountpoint only supports
    # sequential writes).  Writing to BytesIO avoids all seeks.
    buf = io.BytesIO()
    np.savez(
        buf,
        ctx=ctx_arr.astype(np.int32),
        pos=pos_arr.astype(np.int32),
        vocab_size=np.array(vocab_size, dtype=np.int64),
        context_window=np.array(context_window, dtype=np.int64),
    )
    buf.seek(0)
    path = Path(path)
    if not str(path).endswith(".npz"):
        path = Path(str(path) + ".npz")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        shutil.copyfileobj(buf, f)
    n = len(pos_arr)
    if verbose:
        print(f"  Saved {n:,} pairs \u2192 {path}  ({path.stat().st_size / 1e6:.1f} MB)")
    return n


def load_pairs(
    path: str | Path,
    context_window: int | None = None,
    vocab_size: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Load precomputed training pairs from a .npz file.

    Validates ``context_window`` and ``vocab_size`` against embedded metadata
    when provided, raising ``ValueError`` if they don't match — catching the
    common mistake of reusing pairs built from a different vocab or window.

    Returns ``(ctx_arr, pos_arr)`` as int64 arrays ready for training.
    """
    data = np.load(path)
    if context_window is not None:
        saved_cw = int(data["context_window"])
        if saved_cw != context_window:
            raise ValueError(
                f"Pairs file context_window={saved_cw} does not match "
                f"requested context_window={context_window}. "
                "Regenerate the pairs file with matching settings."
            )
    if vocab_size is not None:
        saved_vs = int(data["vocab_size"])
        if saved_vs != vocab_size:
            raise ValueError(
                f"Pairs file vocab_size={saved_vs} does not match "
                f"current vocab_size={vocab_size}. "
                "Regenerate the pairs file with the matching vocab."
            )
    return data["ctx"].astype(np.int64), data["pos"].astype(np.int64)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class DualEncoderTrainer:
    """Train a DualEncoder from a text corpus using frequency-weighted negative sampling.

    Uses a standard nn.Module with torch.compile() for efficient GPU training.
    The compiled model handles graph capture, kernel fusion, and mixed precision
    automatically — compatible with Modal and other GPU cloud providers.

    Usage::

        vocab   = Vocabulary.build_from_nltk()
        trainer = DualEncoderTrainer(vocab, embed_dim=64)
        trainer.train_from_nltk(epochs=3)
        trainer.save_numpy("model.npz")

    The resulting .npz can be loaded with ``DualEncoder.load(path, vocab)``
    without PyTorch.
    """

    def __init__(
        self,
        vocab: Vocabulary,
        embed_dim: int = 64,
        context_window: int = 3,
        neg_samples: int = 20,
        lr: float = 0.005,
        batch_size: int = 2048,
        seed: int = 42,
        device: str = "auto",
        debug: bool = False,
    ) -> None:
        self._vocab = vocab
        self._embed_dim = embed_dim
        self._context_window = context_window
        self._neg_samples = neg_samples
        self._lr = lr
        self._batch_size = batch_size
        self._seed = seed
        self._device_pref = device
        self._debug = debug
        self._model = None  # set after first train call

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train_from_nltk(
        self,
        epochs: int = 3,
        verbose: bool = True,
    ) -> None:
        """Train on the NLTK Brown corpus (auto-downloaded)."""
        torch = _require_torch()
        t0 = time.monotonic()
        if verbose:
            print("Loading Brown corpus…")
        sentences = _brown_sentence_ids(self._vocab)
        if verbose:
            print(f"  {len(sentences):,} sentences loaded  ({time.monotonic()-t0:.2f}s)")
        self._train(sentences, epochs=epochs, torch=torch, verbose=verbose)

    def train_from_file(
        self,
        corpus_path: str | Path,
        epochs: int = 3,
        verbose: bool = True,
    ) -> None:
        """Train on a plain-text file (one sentence per line)."""
        self.train_from_files([Path(corpus_path)], epochs=epochs, verbose=verbose)

    def train_from_files(
        self,
        paths: list[str | Path],
        epochs: int = 3,
        verbose: bool = True,
    ) -> None:
        """Train on multiple plain-text corpus files (combined into one pass).

        Files are loaded in the order given; their sentences are concatenated
        before training begins.  The per-epoch shuffle then mixes them together.
        """
        torch = _require_torch()
        t0 = time.monotonic()
        sentences: list[list[int]] = []
        if verbose:
            print("Loading corpus...")
        total_sentences = 0
        for i, path in enumerate(paths):
            path = Path(path)
            file_sents = _corpus_file_sentence_ids(path, self._vocab)
            total_sentences += len(file_sents)
            sentences.extend(file_sents)
            if verbose:
                elapsed = time.monotonic() - t0
                print(f"\r  Loaded {i+1}/{len(paths)} files ({total_sentences:,} sentences)  ({elapsed:.2f}s)", end="", flush=True)
        if verbose:
            print()  # newline
            print(f"  Total: {len(sentences):,} sentences")
        self._train(sentences, epochs=epochs, torch=torch, verbose=verbose)

    def save_numpy(self, path: str | Path) -> None:
        """Export trained embeddings to .npz (NumPy format) for inference."""
        if self._model is None:
            raise RuntimeError("No trained model — call train_from_nltk() first.")
        ctx = self._model.ctx_embed.weight.detach().cpu().numpy()
        wrd = self._model.wrd_embed.weight.detach().cpu().numpy()
        encoder = DualEncoder(ctx, wrd, self._vocab)
        encoder.save(path)
        print(f"Saved model to {path}  ({Path(path).stat().st_size / 1e6:.1f} MB)")

    def get_encoder(self) -> DualEncoder:
        """Return a DualEncoder with the current trained weights (no file I/O)."""
        if self._model is None:
            raise RuntimeError("No trained model — call train_from_nltk() first.")
        ctx = self._model.ctx_embed.weight.detach().cpu().numpy()
        wrd = self._model.wrd_embed.weight.detach().cpu().numpy()
        return DualEncoder(ctx, wrd, self._vocab)

    def train_from_pairs_file(
        self,
        pairs_path: str | Path,
        epochs: int = 3,
        verbose: bool = True,
    ) -> None:
        """Train directly from a precomputed pairs .npz file (skips corpus loading).

        The file must have been produced by ``save_pairs()`` with the same
        ``context_window`` and ``vocab_size``; a ``ValueError`` is raised if
        they don't match, preventing silent training on stale data.
        """
        torch = _require_torch()
        if verbose:
            print(f"Loading precomputed pairs from {pairs_path}…")
        ctx_np, pos_np = load_pairs(
            pairs_path,
            context_window=self._context_window,
            vocab_size=self._vocab.size,
        )
        if verbose:
            print(f"  {len(pos_np):,} pairs loaded")
        self._train_from_arrays(ctx_np, pos_np, epochs=epochs, torch=torch, verbose=verbose)

    # ------------------------------------------------------------------
    # Internal training loop
    # ------------------------------------------------------------------

    def _train(
        self,
        sentences: list[list[int]],
        epochs: int,
        torch,
        verbose: bool,
    ) -> None:
        ctx_np, pos_np = _precompute_pairs(sentences, self._context_window, verbose=verbose)
        self._train_from_arrays(ctx_np, pos_np, epochs=epochs, torch=torch, verbose=verbose)

    def _train_from_arrays(
        self,
        ctx_np: np.ndarray,
        pos_np: np.ndarray,
        epochs: int,
        torch,
        verbose: bool,
    ) -> None:
        """Core training loop operating on pre-loaded numpy pair arrays.

        Shared by both the corpus-loading path (_train) and the precomputed
        pairs path (train_from_pairs_file).
        """
        try:
            from tqdm import tqdm as _tqdm
        except ImportError:
            _tqdm = None

        device = _resolve_device(torch, self._device_pref)
        is_cuda = str(device).startswith("cuda")
        torch.manual_seed(self._seed)

        # TF32 on Ampere+ GPUs — faster matmul at negligible precision cost.
        if is_cuda and torch.cuda.is_available() and torch.cuda.get_device_properties(device).major >= 8:
            torch.set_float32_matmul_precision("high")

        vocab_size = self._vocab.size
        embed_dim = self._embed_dim
        neg_samples = self._neg_samples

        # Build frequency-weighted negative sampling distribution (f^0.75).
        # Follows Word2Vec: sampling proportional to f(w)^0.75 produces better
        # embeddings than uniform by giving common words more exposure as negatives.
        logfreqs = np.array(self._vocab.logfreq_array(), dtype=np.float32)
        raw_freqs = np.exp(logfreqs)
        neg_w = np.power(raw_freqs, 0.75)
        neg_w[0] = 0.0  # never sample UNK as a negative
        neg_w /= neg_w.sum()
        neg_weights = torch.from_numpy(neg_w).to(device)

        # ---- Model ----------------------------------------------------------
        # Defined here so nn can be closed over without a module-level import.
        nn = torch.nn

        class _DualEncoderModel(nn.Module):
            def __init__(self_):
                super().__init__()
                self_.ctx_embed = nn.Embedding(vocab_size, embed_dim)
                self_.wrd_embed = nn.Embedding(vocab_size, embed_dim)
                nn.init.xavier_uniform_(self_.ctx_embed.weight)
                nn.init.xavier_uniform_(self_.wrd_embed.weight)
                self_.register_buffer("_neg_weights", neg_weights)

            def forward(self_, ctx_ids, pos_ids):
                F = torch.nn.functional
                batch = ctx_ids.size(0)

                # Context vector: mean-pool then L2-normalise → (batch, dim)
                ctx_vecs = F.normalize(self_.ctx_embed(ctx_ids).mean(1), dim=-1)

                # Positive scores → (batch, 1)
                pos_vecs = F.normalize(self_.wrd_embed(pos_ids), dim=-1)
                pos_scores = (ctx_vecs * pos_vecs).sum(-1, keepdim=True)

                # Negative IDs: frequency-weighted sampling entirely on device
                neg_ids = torch.multinomial(
                    self_._neg_weights.expand(batch, -1),
                    neg_samples,
                    replacement=True,
                )
                neg_vecs = F.normalize(self_.wrd_embed(neg_ids), dim=-1)        # (batch, neg, dim)
                neg_scores = torch.bmm(neg_vecs, ctx_vecs.unsqueeze(-1)).squeeze(-1)  # (batch, neg)

                logits = torch.cat([pos_scores, neg_scores], dim=1)             # (batch, 1+neg)
                labels = torch.zeros_like(logits)
                labels[:, 0] = 1.0
                return logits, labels

        model = _DualEncoderModel().to(device)
        self._model = model  # raw module — save_numpy/get_encoder read weights from here

        # torch.compile() handles graph capture, kernel fusion, and precision
        # choices automatically. Falls back transparently on CPU or PyTorch < 2.0.
        compiled = model
        if hasattr(torch, "compile"):
            try:
                compiled = torch.compile(model)
            except Exception:
                pass

        optimizer = torch.optim.Adam(model.parameters(), lr=self._lr)
        bce = nn.BCEWithLogitsLoss()

        # ---- Move pairs to device ------------------------------------------
        n_pairs = len(pos_np)
        n_batches = math.ceil(n_pairs / self._batch_size)
        ctx_dev = torch.from_numpy(ctx_np).to(device)
        pos_dev = torch.from_numpy(pos_np).to(device)
        del ctx_np, pos_np  # free CPU copies

        if verbose:
            if is_cuda:
                vram_alloc_mb = torch.cuda.memory_allocated(device) / 1e6
                vram_total_mb = torch.cuda.get_device_properties(device).total_memory / 1e6
                vram_str = f"VRAM: {vram_alloc_mb:.0f} / {vram_total_mb:.0f} MB"
            else:
                vram_str = f"data: {(ctx_dev.nbytes + pos_dev.nbytes) / 1e6:.0f} MB"
            compiled_str = "on" if compiled is not model else "off"
            print(
                f"Device: {device}  |  training pairs: {n_pairs:,}  |  "
                f"vocab: {vocab_size}  |  embed_dim: {embed_dim}  |  "
                f"neg_samples: {neg_samples}  |  "
                f"batch: {self._batch_size}  |  torch.compile: {compiled_str}  |  "
                f"{vram_str}"
            )

        # ---- Training epochs ------------------------------------------------
        t_train = time.monotonic()

        for epoch in range(1, epochs + 1):
            t_epoch = time.monotonic()
            total_loss = 0.0
            perm = torch.randperm(n_pairs, device=device)

            batch_range: object = range(n_batches)
            if _tqdm is not None and verbose:
                batch_range = _tqdm(
                    batch_range,
                    desc=f"Epoch {epoch}/{epochs}",
                    unit="batch",
                    leave=False,
                )

            for b in batch_range:
                start = b * self._batch_size
                end = min(start + self._batch_size, n_pairs)
                idx = perm[start:end]

                optimizer.zero_grad(set_to_none=True)
                logits, labels = compiled(ctx_dev[idx], pos_dev[idx])
                loss = bce(logits, labels)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                if _tqdm is not None and verbose:
                    batch_range.set_postfix(loss=f"{total_loss / (b + 1):.4f}")

            elapsed = time.monotonic() - t_epoch
            avg_loss = total_loss / max(n_batches, 1)
            pairs_per_sec = n_pairs / elapsed if elapsed > 0 else 0
            if verbose:
                print(
                    f"  Epoch {epoch}/{epochs}  loss={avg_loss:.4f}  "
                    f"time={elapsed:.1f}s  ({pairs_per_sec:,.0f} pairs/s)"
                )

        if verbose:
            total_time = time.monotonic() - t_train
            m, s = divmod(int(total_time), 60)
            h, m = divmod(m, 60)
            time_str = f"{h}h{m:02d}m{s:02d}s" if h else (f"{m}m{s:02d}s" if m else f"{s}s")
            print(f"  Training complete in {time_str}")
