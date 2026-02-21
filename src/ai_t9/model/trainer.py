"""PyTorch training for the DualEncoder model.

This module is intentionally isolated from the inference path — PyTorch is an
optional dependency (pip install ai-t9[train]).  The output is a .npz file
that the pure-NumPy DualEncoder can load without any ML framework.

Training objective: frequency-weighted negative sampling
  For each (context_words, target_word) pair drawn from the corpus:
    - Score the positive (target) word against the context embedding
    - Sample `neg_samples` negatives *on the GPU* using a frequency-weighted
      distribution (f^0.75, as in Word2Vec), via torch.multinomial
    - Apply binary cross-entropy (positive=1, negatives=0)

GPU utilisation design
-----------------------
1. Pre-computed pairs on device — all (context, target) pairs are
   materialised as flat NumPy arrays once, then transferred to the compute
   device (GPU).  Per-epoch shuffling uses torch.randperm on-device, and
   batches are simple O(1) tensor slices + index_select — the CPU does
   virtually no work during the training loop.

2. On-GPU negative sampling — torch.randint generates all negative IDs
   directly on the GPU; no candidate data crosses the PCIe bus per batch.

3. Blocking CUDA sync — cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync)
   is called before context creation so the CPU sleeps on a mutex instead
   of the default spin-wait, which would otherwise pin a core at 100 %.

4. Automatic mixed precision (AMP) — on CUDA, forward passes run under
   torch.amp.autocast, casting torch.bmm to float16.  GradScaler is
   intentionally omitted because this model produces no float16 gradients
   (only bmm is autocasted; everything else stays float32).

5. GPU-side loss accumulation — training loss is accumulated in a device
   tensor and only transferred to CPU for progress-bar redraws (~4×/s),
   avoiding per-batch CPU↔GPU synchronisation.

6. TF32 matmul (Ampere+ GPUs) — float32 matrix multiplications use
   TensorFloat-32 for ~3× speed at effectively no accuracy cost.

7. CUDA Graphs — the entire forward+backward+optimizer step is captured
   into a single replayable CUDA graph after a short warmup.  On each
   batch the CPU only performs two index_selects (data gather into static
   buffers) and one graph.replay() call, reducing per-batch dispatch
   overhead from ~40 Python→C++ transitions to ~3.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import math
import re
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


def _find_cudart() -> ctypes.CDLL | None:
    """Find the CUDA runtime library that is actually loaded in this process.

    PyTorch pip wheels install cudart via the ``nvidia-cuda-runtime-cuXX``
    package (e.g. ``nvidia/cuda_runtime/lib/libcudart.so.12``), NOT inside
    ``torch/lib/``.  The only reliable way to find it is to ensure PyTorch
    has loaded CUDA (``import torch`` triggers this) and then inspect
    ``/proc/self/maps`` for the mapped library.

    Search order:
      1. Import torch (triggers lazy-loading of its CUDA libraries)
      2. ``/proc/self/maps`` — the authoritative answer on Linux
      3. ``nvidia.cuda_runtime`` package path (direct lookup)
      4. ``ctypes.util.find_library`` (system-wide fallback)
    """
    # Ensure torch (+ its bundled CUDA libs) are loaded into the process.
    try:
        import torch as _torch  # noqa: F811
    except ImportError:
        pass

    # 1) Already mapped into the process (most reliable on Linux)
    try:
        with open("/proc/self/maps") as f:
            for line in f:
                m = re.search(r"(/\S*libcudart\S*\.so\S*)", line)
                if m:
                    try:
                        return ctypes.CDLL(m.group(1))
                    except OSError:
                        pass
    except OSError:
        pass

    # 2) nvidia-cuda-runtime package (pip install nvidia-cuda-runtime-cu12)
    try:
        import nvidia.cuda_runtime  # type: ignore[import-untyped]
        lib_dir = Path(nvidia.cuda_runtime.__file__).parent / "lib"
        for candidate in sorted(lib_dir.glob("libcudart*.so*"), reverse=True):
            try:
                return ctypes.CDLL(str(candidate))
            except OSError:
                continue
    except Exception:
        pass

    # 3) System library
    name = ctypes.util.find_library("cudart")
    if name:
        try:
            return ctypes.CDLL(name)
        except OSError:
            pass

    return None


def _set_cuda_blocking_sync() -> bool:
    """Switch CUDA synchronisation from spin-wait to blocking (mutex).

    By default CUDA uses ``cudaDeviceScheduleSpin`` when the device context
    count is low, which busy-loops a CPU core at 100 % while waiting for
    GPU operations.  ``cudaDeviceScheduleBlockingSync`` (flag 0x04) makes
    the CPU sleep on a mutex instead, freeing the core for useful work.

    Must be called **before** the CUDA primary context is created (i.e.
    before any ``.to('cuda')``, and ideally before ``torch.cuda.is_available()``
    since some driver versions create a context there).

    Returns True if the flag was accepted, False otherwise.
    """
    cudart = _find_cudart()
    if cudart is None:
        return False
    try:
        cudart.cudaSetDeviceFlags.argtypes = [ctypes.c_uint]
        cudart.cudaSetDeviceFlags.restype = ctypes.c_int
        # cudaDeviceScheduleBlockingSync = 0x04
        ret = cudart.cudaSetDeviceFlags(0x04)
        return ret == 0   # cudaSuccess
    except Exception:
        return False


# ---------------------------------------------------------------------------
# PyTorch model (training-time only)
# ---------------------------------------------------------------------------

class _TorchDualEncoder:
    """Two nn.Embedding tables with on-GPU negative sampling.

    Uses frequency-weighted negative sampling (Word2Vec-style f^0.75
    distribution) for better embedding quality.  The sampling weight table
    is pre-computed once and stored on the GPU device.
    """

    def __init__(self, vocab_size: int, embed_dim: int, neg_samples: int,
                 torch, device, neg_weights: "torch.Tensor | None" = None):
        self._vocab_size = vocab_size
        self._neg_samples = neg_samples
        nn = torch.nn
        self.ctx_embed = nn.Embedding(vocab_size, embed_dim).to(device)
        self.wrd_embed = nn.Embedding(vocab_size, embed_dim).to(device)
        nn.init.xavier_uniform_(self.ctx_embed.weight)
        nn.init.xavier_uniform_(self.wrd_embed.weight)

        # Frequency-weighted negative sampling distribution.
        # If no weights are provided, fall back to uniform sampling.
        if neg_weights is not None:
            self._neg_weights = neg_weights.to(device)
        else:
            self._neg_weights = None

    def parameters(self):
        return list(self.ctx_embed.parameters()) + list(self.wrd_embed.parameters())

    def score_with_negatives(self, ctx_ids, pos_ids, torch):
        """Compute scores for positives and GPU-generated negatives.

        Args:
            ctx_ids: LongTensor (batch, ctx_len) — context word IDs
            pos_ids: LongTensor (batch,)         — positive target word IDs

        Returns:
            logits: FloatTensor (batch, 1 + neg_samples)
            labels: FloatTensor (batch, 1 + neg_samples) — 1.0 at index 0 only
        """
        F = torch.nn.functional
        device = ctx_ids.device
        batch = ctx_ids.size(0)

        # Context vector: mean-pool then L2-normalise  → (batch, dim)
        ctx_vecs = F.normalize(self.ctx_embed(ctx_ids).mean(dim=1), dim=-1)

        # Positive scores  → (batch, 1)
        pos_vecs   = F.normalize(self.wrd_embed(pos_ids), dim=-1)
        pos_scores = (ctx_vecs * pos_vecs).sum(-1, keepdim=True)

        # Negative IDs: frequency-weighted sampling (f^0.75) when available,
        # uniform fallback otherwise.  Both run entirely on the GPU.
        if self._neg_weights is not None:
            neg_ids = torch.multinomial(
                self._neg_weights.expand(batch, -1),
                self._neg_samples,
                replacement=True,
            )
        else:
            neg_ids = torch.randint(1, self._vocab_size, (batch, self._neg_samples), device=device)
        neg_vecs  = F.normalize(self.wrd_embed(neg_ids), dim=-1)          # (batch, neg, dim)
        neg_scores = torch.bmm(neg_vecs, ctx_vecs.unsqueeze(-1)).squeeze(-1)  # (batch, neg)

        logits = torch.cat([pos_scores, neg_scores], dim=1)               # (batch, 1+neg)
        labels = torch.zeros_like(logits)
        labels[:, 0] = 1.0
        return logits, labels


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
    # Truncate to actual count (idx < n_pairs when UNK targets were skipped)
    return ctx_arr[:idx], pos_arr[:idx]


def _format_eta(seconds: float) -> str:
    """Format seconds into a compact human-readable duration."""
    if seconds < 0:
        return "?"
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds}s"
    m, s = divmod(seconds, 60)
    if m < 60:
        return f"{m}m{s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m{s:02d}s"


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class DualEncoderTrainer:
    """Train a DualEncoder from a text corpus using candidate-only neg sampling.

    Usage::

        vocab   = Vocabulary.build_from_nltk()
        trainer = DualEncoderTrainer(vocab, embed_dim=64)
        trainer.train_from_nltk(epochs=3)
        trainer.save_numpy("model.npz")

    The resulting .npz can be loaded with ``DualEncoder.load(path, vocab)``
    without PyTorch.

    Set ``debug=True`` to print a timestamped breakdown of every setup phase
    and the first few batches, which is useful for identifying startup bottlenecks.
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
        self._model: _TorchDualEncoder | None = None

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
        for path in paths:
            path = Path(path)
            file_sents = _corpus_file_sentence_ids(path, self._vocab)
            if verbose:
                print(f"  {path.name}: {len(file_sents):,} sentences  "
                      f"({time.monotonic()-t0:.2f}s)")
            sentences.extend(file_sents)
        if verbose:
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

    # ------------------------------------------------------------------
    # Debug helpers
    # ------------------------------------------------------------------

    def _phase(self, label: str, t_ref: float) -> None:
        """Print a timestamped phase marker (only when debug=True)."""
        if self._debug:
            print(f"  [+{time.monotonic() - t_ref:.3f}s] {label}")

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
        t_ref = time.monotonic()

        # Switch CUDA from spin-wait to blocking sync.  This MUST happen
        # before _resolve_device() because torch.cuda.is_available() may
        # create the CUDA primary context on some driver versions, after
        # which cudaSetDeviceFlags returns cudaErrorSetOnActiveProcess.
        if self._device_pref in ("auto", "cuda"):
            ok = _set_cuda_blocking_sync()
            self._phase(
                f"CUDA blocking sync {'set' if ok else 'FAILED (context may already exist)'}",
                t_ref,
            )

        # ---- Phase 1: device resolution --------------------------------------
        device = _resolve_device(torch, self._device_pref)
        is_cuda = str(device).startswith("cuda")
        self._phase("device resolved", t_ref)

        torch.manual_seed(self._seed)

        # ---- Phase 2: model creation + .to(device) --------------------------
        # NOTE: On CUDA the first .to(device) call triggers driver initialisation
        # which can block for 10–30 s.  This is a one-time cost per process.
        vocab_size = self._vocab.size

        # Build frequency-weighted negative sampling distribution (f^0.75).
        # This follows the Word2Vec insight: sampling negatives proportional
        # to f(w)^0.75 makes the model work harder on common words and
        # produces significantly better embeddings than uniform sampling.
        logfreqs = np.array(self._vocab.logfreq_array(), dtype=np.float32)
        raw_freqs = np.exp(logfreqs)
        neg_w = np.power(raw_freqs, 0.75)
        neg_w[0] = 0.0  # never sample UNK as a negative
        neg_w /= neg_w.sum()
        neg_weights = torch.from_numpy(neg_w).to(device)
        self._phase("negative sampling weights computed", t_ref)

        model = _TorchDualEncoder(
            vocab_size, self._embed_dim, self._neg_samples, torch, device,
            neg_weights=neg_weights,
        )
        self._model = model
        self._phase(f"model created and moved to {device}", t_ref)

        # ---- Phase 3: optimizer + loss fn -----------------------------------
        # capturable=True lets Adam.step() be recorded inside a CUDA graph.
        optimizer = torch.optim.Adam(
            model.parameters(), lr=self._lr, capturable=is_cuda)
        bce = torch.nn.BCEWithLogitsLoss()
        self._phase("optimizer ready", t_ref)

        # ---- Phase 4: pre-compute pairs + move to device ---------------------
        # All training data lives on the compute device (GPU).  Per-epoch
        # shuffling and per-batch slicing happen entirely on-device via
        # torch.randperm + index_select, so the CPU does virtually no work
        # during the training loop — just dispatching CUDA kernels.
        ctx_np, pos_np = _precompute_pairs(sentences, self._context_window)
        n_pairs = len(pos_np)
        n_batches = math.ceil(n_pairs / self._batch_size)
        self._phase(f"pairs pre-computed ({n_pairs:,} pairs)", t_ref)

        ctx_dev = torch.from_numpy(ctx_np).to(device)
        pos_dev = torch.from_numpy(pos_np).to(device)
        del ctx_np, pos_np          # free CPU copies
        self._phase("training data moved to device", t_ref)

        # AMP autocast (CUDA only).  Under autocast only torch.bmm is cast to
        # float16; everything else stays float32.  GradScaler is omitted
        # because no float16 gradients are produced.
        use_amp = is_cuda

        # TF32 matmul on Ampere+ GPUs.
        if is_cuda and hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")

        # ---- Phase 4b: debug timing probe (before graph setup) ---------------
        # Run a single eager forward+backward BEFORE CUDA graph warmup so that
        # AccumulateGrad nodes baked into the graph during capture don't
        # conflict with the eager backward here.  Graph setup resets the model
        # weights and optimizer state afterwards, so this step has no effect
        # on the trained result.
        if self._debug:
            debug_idx = torch.arange(
                min(self._batch_size, n_pairs), dtype=torch.int64, device=device)
            ctx_t = ctx_dev[debug_idx]
            pos_t = pos_dev[debug_idx]
            self._phase(f"  first batch on device ({len(debug_idx)} pairs)", t_ref)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type="cuda", enabled=use_amp):
                logits, labels = model.score_with_negatives(ctx_t, pos_t, torch)
                loss = bce(logits, labels)
            self._phase("  first forward pass done", t_ref)
            loss.backward()
            self._phase("  first backward pass done", t_ref)
            optimizer.step()
            self._phase("  first optimizer step done", t_ref)
            del ctx_t, pos_t, logits, labels, loss, debug_idx
            # Reset model and optimizer so this probe step has zero effect on
            # actual training (the graph setup below also resets, but we do it
            # here too so the probe is invisible when graphs are not used).
            optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
                torch.nn.init.xavier_uniform_(model.ctx_embed.weight)
                torch.nn.init.xavier_uniform_(model.wrd_embed.weight)
            for state in optimizer.state.values():
                for v in state.values():
                    if isinstance(v, torch.Tensor):
                        v.zero_()

        # ---- Phase 5: CUDA Graph setup (optional) ----------------------------
        # Each batch iteration dispatches ~40 individual CUDA kernels through
        # Python → PyTorch dispatcher → CUDA driver.  At ~300 batches/sec
        # that's ~12 000 dispatch calls/sec — the CPU cost of those Python→C++
        # transitions is the dominant source of the remaining ~90 % CPU usage.
        #
        # CUDA Graphs capture the entire forward+backward+optimizer step into
        # a single replayable GPU-side graph.  On each batch the CPU work is
        # reduced to 2 index_selects (data gather) + 1 graph.replay(), cutting
        # per-batch dispatch overhead by roughly an order of magnitude.
        use_graph = is_cuda and n_pairs >= self._batch_size
        graph = None
        static_ctx = None
        static_pos = None
        static_loss = None

        if use_graph:
            batch_sz = self._batch_size
            static_ctx = torch.zeros(
                batch_sz, self._context_window, dtype=torch.int64, device=device)
            static_pos = torch.zeros(
                batch_sz, dtype=torch.int64, device=device)

            # Warmup — CUDA graphs require several eager runs on the same
            # static tensors to let PyTorch's caching allocator settle.
            # Uses a side stream; graph capture below happens on the SAME
            # stream so AccumulateGrad nodes match.
            graph_stream = torch.cuda.Stream()
            graph_stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(graph_stream):
                for _ in range(3):
                    optimizer.zero_grad(set_to_none=True)
                    with torch.amp.autocast(device_type="cuda", enabled=use_amp):
                        logits, labels = model.score_with_negatives(
                            static_ctx, static_pos, torch)
                        loss = bce(logits, labels)
                    loss.backward()
                    optimizer.step()
                    # Drop all references to the autograd graph immediately after
                    # each backward.  If any of these tensors survived into the
                    # next iteration their AccumulateGrad nodes — registered on
                    # graph_stream during this backward — would still be alive
                    # when the next forward builds a fresh graph, causing PyTorch
                    # to warn about a stream mismatch on the AccumulateGrad nodes.
                    del logits, labels, loss
            torch.cuda.current_stream().wait_stream(graph_stream)
            self._phase("CUDA graph warmup done", t_ref)

            # Capture the full training step into a graph on the SAME side
            # stream used for warmup, so all AccumulateGrad nodes match.
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.stream(graph_stream):
                optimizer.zero_grad(set_to_none=True)
                with torch.cuda.graph(graph):
                    with torch.amp.autocast(device_type="cuda", enabled=use_amp):
                        s_logits, s_labels = model.score_with_negatives(
                            static_ctx, static_pos, torch)
                        static_loss = bce(s_logits, s_labels)
                    static_loss.backward()
                    optimizer.step()
            torch.cuda.current_stream().wait_stream(graph_stream)
            self._phase("CUDA graph captured", t_ref)

            # Reset model weights and optimizer moments so the warmup's
            # garbage updates don't affect real training.  We must keep the
            # SAME tensor objects (addresses) because the graph references them.
            with torch.no_grad():
                torch.nn.init.xavier_uniform_(model.ctx_embed.weight)
                torch.nn.init.xavier_uniform_(model.wrd_embed.weight)
            for state in optimizer.state.values():
                for v in state.values():
                    if isinstance(v, torch.Tensor) and v.is_floating_point():
                        v.zero_()
                    elif isinstance(v, torch.Tensor):  # step counter
                        v.zero_()
            self._phase("model + optimizer reset after warmup", t_ref)

            # Truncate training pairs to a multiple of batch_size so there is
            # never a partial last batch.  A partial batch cannot use the CUDA
            # graph (wrong shape), so it falls back to an eager backward — but
            # the captured graph keeps AccumulateGrad nodes for the embedding
            # weights alive with an internal stream reference, and any eager
            # backward on those same weights triggers a stream-mismatch warning
            # and can cause CUDA assert failures.  Dropping at most
            # (batch_size - 1) pairs per epoch is negligible at this scale.
            n_pairs = (n_pairs // self._batch_size) * self._batch_size
            n_batches = n_pairs // self._batch_size

        if verbose:
            vram_mb = (ctx_dev.nbytes + pos_dev.nbytes) / 1e6
            print(
                f"Device: {device}  |  training pairs: {n_pairs:,}  |  "
                f"vocab: {vocab_size}  |  embed_dim: {self._embed_dim}  |  "
                f"neg_samples: {self._neg_samples} (on GPU)  |  "
                f"batch: {self._batch_size}  |  AMP: {'on' if use_amp else 'off'}  |  "
                f"CUDA graph: {'on' if graph else 'off'}  |  "
                f"data VRAM: {vram_mb:.0f} MB"
            )

        # GPU-side loss accumulator.
        loss_accum = torch.zeros((), device=device)

        # ---- Training epochs ------------------------------------------------
        total_train_batches = n_batches * epochs
        train_batches_done = 0
        t_train = time.monotonic()
        last_progress_t = 0.0
        _PROGRESS_INTERVAL = 0.25       # seconds between progress-bar redraws

        for epoch in range(1, epochs + 1):
            t_epoch = time.monotonic()
            loss_accum.zero_()

            # On-device shuffle via randperm — no CPU work at all.
            perm = torch.randperm(n_pairs, device=device)
            self._phase(f"epoch {epoch} shuffle done", t_ref)

            for b in range(n_batches):
                start = b * self._batch_size
                end = min(start + self._batch_size, n_pairs)
                idx = perm[start:end]
                batch_len = end - start

                # ---- graph replay or eager fallback -------------------------
                if graph is not None and batch_len == self._batch_size:
                    # Graph path: gather into static buffers, replay graph.
                    # Only 3 CUDA API calls instead of ~40.
                    torch.index_select(ctx_dev, 0, idx, out=static_ctx)
                    torch.index_select(pos_dev, 0, idx, out=static_pos)
                    graph.replay()
                    loss_accum += static_loss.detach()
                else:
                    # Eager fallback (non-CUDA or no graph — never reached when
                    # use_graph is True because n_pairs is truncated to a
                    # multiple of batch_size above).
                    ctx_t = ctx_dev[idx]
                    pos_t = pos_dev[idx]
                    optimizer.zero_grad(set_to_none=True)
                    with torch.amp.autocast(device_type="cuda", enabled=use_amp):
                        logits, labels = model.score_with_negatives(ctx_t, pos_t, torch)
                        loss = bce(logits, labels)
                    loss.backward()
                    optimizer.step()
                    loss_accum += loss.detach()

                train_batches_done += 1

                # ---- inline progress bar ------------------------------------
                now = time.monotonic()
                if verbose and (
                    now - last_progress_t >= _PROGRESS_INTERVAL
                    or b == n_batches - 1
                ):
                    last_progress_t = now
                    done = b + 1
                    frac = done / n_batches
                    bar_w = 20
                    filled = int(bar_w * frac)
                    bar = "\u2588" * filled + "\u2591" * (bar_w - filled)
                    avg_loss = loss_accum.item() / done  # only sync point

                    epoch_elapsed = now - t_epoch
                    epoch_eta = epoch_elapsed / done * (n_batches - done)

                    train_elapsed = now - t_train
                    total_eta = (
                        train_elapsed / train_batches_done
                        * (total_train_batches - train_batches_done)
                    )

                    print(
                        f"\r  Epoch {epoch}/{epochs}  |{bar}| "
                        f"{done}/{n_batches}  "
                        f"loss={avg_loss:.4f}  "
                        f"ETA: {_format_eta(epoch_eta)}  "
                        f"[total: {_format_eta(total_eta)}]"
                        "\033[K",
                        end="", flush=True,
                    )

            elapsed = time.monotonic() - t_epoch
            if verbose:
                total_loss = loss_accum.item()
                avg_loss = total_loss / max(n_batches, 1)
                pairs_per_sec = n_pairs / elapsed if elapsed > 0 else 0
                print(
                    f"\r  Epoch {epoch}/{epochs}  loss={avg_loss:.4f}  "
                    f"time={elapsed:.1f}s  ({pairs_per_sec:,.0f} pairs/s)"
                    "\033[K"
                )

        if verbose:
            total_time = time.monotonic() - t_train
            print(f"  Training complete in {_format_eta(total_time)}")
