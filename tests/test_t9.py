"""Tests for the ai_t9 library."""

from __future__ import annotations

import math
from collections import Counter

import numpy as np
import pytest

from ai_t9.t9_map import (
    T9_MAP,
    word_to_digits,
    candidates_from_digits,
    is_valid_digit_sequence,
)
from ai_t9.model.vocab import Vocabulary
from ai_t9.dictionary import T9Dictionary
from ai_t9.model.dual_encoder import DualEncoder
from ai_t9.ngram import BigramScorer
from ai_t9.predictor import T9Predictor, _normalise
from ai_t9.session import T9Session


# ---------------------------------------------------------------------------
# Fixtures: tiny vocabulary for fast tests (no NLTK needed)
# ---------------------------------------------------------------------------

SAMPLE_WORDS = [
    # digit seq → candidates
    "home",   # 4663
    "gone",   # 4663
    "good",   # 4663
    "hood",   # 4663
    "i",      # 4
    "go",     # 46
    "in",     # 46
    "me",     # 63
    "of",     # 63
    "the",    # 843
    "tie",    # 843
    "and",    # 263
    "any",    # 269
    "a",      # 2
    "be",     # 23
    "hi",     # 44
]

SAMPLE_COUNTS = [
    1000,  # home
    200,   # gone
    800,   # good
    50,    # hood
    5000,  # i
    600,   # go
    500,   # in
    900,   # me
    3000,  # of
    8000,  # the
    100,   # tie
    4000,  # and
    300,   # any
    6000,  # a
    700,   # be
    400,   # hi
]


@pytest.fixture
def tiny_vocab() -> Vocabulary:
    return Vocabulary(SAMPLE_WORDS, SAMPLE_COUNTS)


@pytest.fixture
def tiny_dict(tiny_vocab: Vocabulary) -> T9Dictionary:
    return T9Dictionary(tiny_vocab)


@pytest.fixture
def tiny_predictor(tiny_vocab: Vocabulary, tiny_dict: T9Dictionary) -> T9Predictor:
    """Predictor with no model or ngram — pure frequency ranking."""
    return T9Predictor(tiny_dict, model=None, ngram=None)


@pytest.fixture
def tiny_encoder(tiny_vocab: Vocabulary) -> DualEncoder:
    return DualEncoder.random_init(tiny_vocab, embed_dim=8, seed=0)


# ===========================================================================
# T9 mapping tests
# ===========================================================================

class TestT9Map:
    def test_keypad_coverage(self):
        """All digits 2-9 are present in T9_MAP."""
        assert set(T9_MAP.keys()) == set("23456789")

    def test_all_26_letters_covered(self):
        all_letters = "".join(T9_MAP.values())
        assert sorted(all_letters) == list("abcdefghijklmnopqrstuvwxyz")

    def test_word_to_digits_simple(self):
        assert word_to_digits("home") == "4663"
        assert word_to_digits("good") == "4663"
        assert word_to_digits("the") == "843"
        assert word_to_digits("i") == "4"

    def test_word_to_digits_uppercase_normalised(self):
        assert word_to_digits("HOME") == "4663"
        assert word_to_digits("Hello") == "43556"

    def test_word_to_digits_non_alpha_returns_none(self):
        assert word_to_digits("hello!") is None
        assert word_to_digits("3abc") is None
        assert word_to_digits("") is None

    def test_candidates_from_digits_length(self):
        # "2" → 3 letters
        cands = candidates_from_digits("2")
        assert len(cands) == 3
        assert set(cands) == {"a", "b", "c"}

    def test_candidates_from_digits_two_digits(self):
        cands = candidates_from_digits("23")
        assert len(cands) == 3 * 3  # abc × def = 9
        assert "ad" in cands
        assert "cf" in cands

    def test_candidates_from_digits_invalid_raises(self):
        with pytest.raises(ValueError):
            candidates_from_digits("1")  # 1 not on T9

    def test_is_valid_digit_sequence(self):
        assert is_valid_digit_sequence("4663")
        assert not is_valid_digit_sequence("")
        assert not is_valid_digit_sequence("4061")  # 0 and 1 not valid


# ===========================================================================
# Vocabulary tests
# ===========================================================================

class TestVocabulary:
    def test_size(self, tiny_vocab: Vocabulary):
        # +1 for <unk> inserted at index 0
        assert tiny_vocab.size == len(SAMPLE_WORDS) + 1

    def test_unk_at_index_zero(self, tiny_vocab: Vocabulary):
        assert tiny_vocab.UNK_ID == 0
        assert tiny_vocab.id_to_word(0) == "<unk>"

    def test_word_roundtrip(self, tiny_vocab: Vocabulary):
        for word in SAMPLE_WORDS:
            wid = tiny_vocab.word_to_id(word)
            assert tiny_vocab.id_to_word(wid) == word

    def test_unknown_word_maps_to_unk(self, tiny_vocab: Vocabulary):
        assert tiny_vocab.word_to_id("zzzzzzz") == 0

    def test_logfreq_ordering(self, tiny_vocab: Vocabulary):
        """Higher-count words should have higher log-freq."""
        the_lf = tiny_vocab.logfreq(tiny_vocab.word_to_id("the"))
        hood_lf = tiny_vocab.logfreq(tiny_vocab.word_to_id("hood"))
        assert the_lf > hood_lf

    def test_logfreq_non_positive(self, tiny_vocab: Vocabulary):
        """All log-freqs should be ≤ 0 (log of probability)."""
        for wid in range(tiny_vocab.size):
            assert tiny_vocab.logfreq(wid) <= 0.0

    def test_save_load_roundtrip(self, tiny_vocab: Vocabulary, tmp_path):
        path = tmp_path / "vocab.json"
        tiny_vocab.save(path)
        loaded = Vocabulary.load(path)
        assert loaded.size == tiny_vocab.size
        for word in SAMPLE_WORDS:
            assert loaded.word_to_id(word) == tiny_vocab.word_to_id(word)

    def test_build_from_counts(self):
        counter = Counter({"cat": 10, "dog": 5, "the": 100, "a": 50})
        vocab = Vocabulary.build_from_counts(counter, max_words=3)
        # Should keep top 3: the, a, cat
        assert "the" in vocab
        assert "a" in vocab
        assert "cat" in vocab
        assert "dog" not in vocab  # cut off at max_words=3


# ===========================================================================
# T9Dictionary tests
# ===========================================================================

class TestT9Dictionary:
    def test_lookup_returns_matching_words(self, tiny_dict: T9Dictionary):
        hits = tiny_dict.lookup("4663")
        words = [w for w, _ in hits]
        assert "home" in words
        assert "good" in words
        assert "gone" in words
        assert "hood" in words

    def test_lookup_sorted_by_frequency(self, tiny_dict: T9Dictionary):
        hits = tiny_dict.lookup("4663")
        words = [w for w, _ in hits]
        # "the" has highest count in sample; "home" > "gone"
        assert words.index("home") < words.index("gone")  # home (1000) > gone (200)

    def test_lookup_no_match_returns_empty(self, tiny_dict: T9Dictionary):
        # "111" is invalid but even a valid seq with no words should return []
        # "22222" (five a/b/c combos) almost certainly has no word in tiny vocab
        result = tiny_dict.lookup("22222")
        assert result == []

    def test_lookup_single_letter(self, tiny_dict: T9Dictionary):
        hits = tiny_dict.lookup("4")  # → i, h (but "i" is in vocab, "h" not)
        words = [w for w, _ in hits]
        assert "i" in words

    def test_save_load_roundtrip(self, tiny_dict: T9Dictionary, tiny_vocab: Vocabulary, tmp_path):
        path = tmp_path / "dict.json"
        tiny_dict.save(path)
        loaded = T9Dictionary.load(path, tiny_vocab)
        assert loaded.lookup("4663") == tiny_dict.lookup("4663")

    def test_word_ids_are_valid(self, tiny_dict: T9Dictionary, tiny_vocab: Vocabulary):
        for seq in tiny_dict.digit_sequences():
            for word, wid in tiny_dict.lookup(seq):
                assert tiny_vocab.id_to_word(wid) == word


# ===========================================================================
# DualEncoder tests
# ===========================================================================

class TestDualEncoder:
    def test_encode_context_shape(self, tiny_encoder: DualEncoder, tiny_vocab: Vocabulary):
        ctx_ids = tiny_vocab.words_to_ids(["the", "good"])
        vec = tiny_encoder.encode_context(ctx_ids)
        assert vec.shape == (tiny_encoder.embed_dim,)

    def test_encode_empty_context_returns_zeros(self, tiny_encoder: DualEncoder):
        vec = tiny_encoder.encode_context([])
        assert vec.shape == (tiny_encoder.embed_dim,)
        np.testing.assert_array_equal(vec, 0)

    def test_score_candidates_shape(self, tiny_encoder: DualEncoder, tiny_vocab: Vocabulary):
        ctx_ids = tiny_vocab.words_to_ids(["the"])
        cand_ids = tiny_vocab.words_to_ids(["home", "gone", "good"])
        scores = tiny_encoder.score_candidates(ctx_ids, cand_ids)
        assert scores.shape == (3,)

    def test_score_candidates_empty(self, tiny_encoder: DualEncoder):
        scores = tiny_encoder.score_candidates([0], [])
        assert len(scores) == 0

    def test_save_load_roundtrip(self, tiny_encoder: DualEncoder, tiny_vocab: Vocabulary, tmp_path):
        path = tmp_path / "model.npz"
        tiny_encoder.save(path)
        loaded = DualEncoder.load(path, tiny_vocab)
        ctx_ids = [1, 2]
        cand_ids = [1, 2, 3]
        np.testing.assert_allclose(
            tiny_encoder.score_candidates(ctx_ids, cand_ids),
            loaded.score_candidates(ctx_ids, cand_ids),
        )

    def test_quantize_preserves_shape(self, tiny_encoder: DualEncoder, tiny_vocab: Vocabulary):
        q = tiny_encoder.quantize_int8()
        assert q.embed_dim == tiny_encoder.embed_dim
        ctx_ids = [1]
        cand_ids = [1, 2, 3]
        scores = q.score_candidates(ctx_ids, cand_ids)
        assert scores.shape == (3,)


# ===========================================================================
# BigramScorer tests
# ===========================================================================

class TestBigramScorer:
    @pytest.fixture
    def trained_bigram(self, tiny_vocab: Vocabulary) -> BigramScorer:
        scorer = BigramScorer(tiny_vocab)
        # Simulate a corpus: "the good home"
        ids = tiny_vocab.words_to_ids(["the", "good", "home", "the", "good", "gone"])
        scorer.train_on_ids(ids)
        return scorer

    def test_log_prob_negative(self, trained_bigram: BigramScorer, tiny_vocab: Vocabulary):
        the_id = tiny_vocab.word_to_id("the")
        good_id = tiny_vocab.word_to_id("good")
        lp = trained_bigram.log_prob(the_id, good_id)
        assert lp < 0

    def test_seen_bigram_higher_than_unseen(self, trained_bigram: BigramScorer, tiny_vocab: Vocabulary):
        the_id = tiny_vocab.word_to_id("the")
        good_id = tiny_vocab.word_to_id("good")
        hood_id = tiny_vocab.word_to_id("hood")
        lp_seen = trained_bigram.log_prob(the_id, good_id)
        lp_unseen = trained_bigram.log_prob(the_id, hood_id)
        assert lp_seen > lp_unseen

    def test_score_candidates_length(self, trained_bigram: BigramScorer, tiny_vocab: Vocabulary):
        the_id = tiny_vocab.word_to_id("the")
        cand_ids = tiny_vocab.words_to_ids(["home", "gone", "good", "hood"])
        scores = trained_bigram.score_candidates(the_id, cand_ids)
        assert len(scores) == 4

    def test_save_load_roundtrip(self, trained_bigram: BigramScorer, tiny_vocab: Vocabulary, tmp_path):
        path = tmp_path / "bigram.json"
        trained_bigram.save(path)
        loaded = BigramScorer.load(path, tiny_vocab)
        the_id = tiny_vocab.word_to_id("the")
        good_id = tiny_vocab.word_to_id("good")
        assert math.isclose(
            trained_bigram.log_prob(the_id, good_id),
            loaded.log_prob(the_id, good_id),
        )


# ===========================================================================
# T9Predictor tests
# ===========================================================================

class TestT9Predictor:
    def test_predict_returns_matching_words(self, tiny_predictor: T9Predictor):
        results = tiny_predictor.predict("4663")
        assert set(results).issubset({"home", "good", "gone", "hood"})
        assert len(results) <= 5

    def test_predict_top_k(self, tiny_predictor: T9Predictor):
        results = tiny_predictor.predict("4663", top_k=2)
        assert len(results) == 2

    def test_predict_no_match(self, tiny_predictor: T9Predictor):
        results = tiny_predictor.predict("22222")
        assert results == []

    def test_predict_invalid_seq_raises(self, tiny_predictor: T9Predictor):
        with pytest.raises(ValueError):
            tiny_predictor.predict("1234")  # 1 is not T9

    def test_predict_freq_ordering(self, tiny_predictor: T9Predictor):
        """Without context, higher-frequency word should rank first."""
        results = tiny_predictor.predict("4663")
        # "home" (1000) should beat "gone" (200) and "hood" (50)
        assert results.index("home") < results.index("gone")
        assert results.index("home") < results.index("hood")

    def test_predict_with_model(
        self, tiny_dict: T9Dictionary, tiny_encoder: DualEncoder, tiny_vocab: Vocabulary
    ):
        predictor = T9Predictor(tiny_dict, model=tiny_encoder)
        results = predictor.predict("4663", context=["the"])
        assert len(results) >= 1
        assert all(w in {"home", "good", "gone", "hood"} for w in results)

    def test_predict_return_details(self, tiny_predictor: T9Predictor):
        from ai_t9.predictor import RankedCandidate
        results = tiny_predictor.predict("4663", return_details=True)
        assert all(isinstance(r, RankedCandidate) for r in results)
        # Check scores are monotonically decreasing
        finals = [r.final_score for r in results]
        assert finals == sorted(finals, reverse=True)

    def test_weights_normalised(self, tiny_predictor: T9Predictor):
        w = tiny_predictor.weights
        assert abs(sum(w.values()) - 1.0) < 1e-6

    def test_from_files_roundtrip(
        self,
        tiny_dict: T9Dictionary,
        tiny_vocab: Vocabulary,
        tiny_encoder: DualEncoder,
        tmp_path,
    ):
        vocab_p = tmp_path / "vocab.json"
        dict_p = tmp_path / "dict.json"
        model_p = tmp_path / "model.npz"
        tiny_vocab.save(vocab_p)
        tiny_dict.save(dict_p)
        tiny_encoder.save(model_p)

        loaded = T9Predictor.from_files(vocab_p, dict_p, model_path=model_p)
        assert loaded.has_model
        results = loaded.predict("4663")
        assert len(results) >= 1


# ===========================================================================
# T9Session tests
# ===========================================================================

class TestT9Session:
    def test_dial_returns_words(self, tiny_predictor: T9Predictor):
        session = T9Session(tiny_predictor)
        results = session.dial("4663")
        assert len(results) >= 1

    def test_confirm_updates_context(self, tiny_predictor: T9Predictor):
        session = T9Session(tiny_predictor)
        session.confirm("hello")
        assert session.context == ["hello"]

    def test_add_context(self, tiny_predictor: T9Predictor):
        session = T9Session(tiny_predictor)
        session.add_context("i", "am", "going")
        assert session.context == ["i", "am", "going"]

    def test_context_window_max_size(self, tiny_predictor: T9Predictor):
        session = T9Session(tiny_predictor, context_window=3)
        for word in ["a", "b", "c", "d", "e"]:
            session.confirm(word)
        assert len(session.context) == 3
        assert session.context == ["c", "d", "e"]

    def test_reset_clears_context(self, tiny_predictor: T9Predictor):
        session = T9Session(tiny_predictor)
        session.add_context("hello", "world")
        session.reset()
        assert session.context == []

    def test_full_flow(self, tiny_predictor: T9Predictor):
        """Simulate typing 'i go home'."""
        session = T9Session(tiny_predictor)
        r1 = session.dial("4")   # "i"
        session.confirm("i")
        r2 = session.dial("46")  # "go", "in"
        session.confirm("go")
        r3 = session.dial("4663")  # "home", "good", "gone", "hood"
        assert "home" in r3 or len(r3) > 0  # something is returned


# ===========================================================================
# Normalise helper
# ===========================================================================

class TestNormalise:
    def test_min_max(self):
        arr = np.array([1.0, 2.0, 3.0, 4.0])
        out = _normalise(arr)
        assert out[0] == pytest.approx(0.0)
        assert out[-1] == pytest.approx(1.0)

    def test_uniform_returns_zeros(self):
        arr = np.array([5.0, 5.0, 5.0])
        out = _normalise(arr)
        np.testing.assert_array_equal(out, 0)
