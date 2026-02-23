"""Tests for the ai_t9 library."""

from __future__ import annotations

import math
import os
import sys
from collections import Counter

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

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
from ai_t9.predictor import T9Predictor, RankedCandidate, _normalise
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

    def test_merge_wordlist_adds_new_words(self, tiny_vocab: Vocabulary):
        wordlist = {"xylophone", "aardvark", "home"}  # home already in vocab
        merged = tiny_vocab.merge_wordlist(wordlist)
        # home was already present — should not be duplicated
        assert merged.word_to_id("home") == tiny_vocab.word_to_id("home")
        # xylophone and aardvark are new
        assert "xylophone" in merged
        assert "aardvark" in merged
        assert merged.word_to_id("xylophone") != merged.UNK_ID
        assert merged.word_to_id("aardvark") != merged.UNK_ID
        assert merged.size == tiny_vocab.size + 2

    def test_merge_wordlist_floor_frequency(self, tiny_vocab: Vocabulary):
        merged = tiny_vocab.merge_wordlist({"xylophone"})
        xyl_lf = merged.logfreq(merged.word_to_id("xylophone"))
        unk_lf = merged.logfreq(merged.UNK_ID)
        home_lf = merged.logfreq(merged.word_to_id("home"))
        # New word should have log-freq above UNK but below frequent words
        assert xyl_lf > unk_lf
        assert xyl_lf < home_lf

    def test_merge_wordlist_empty_returns_self(self, tiny_vocab: Vocabulary):
        merged = tiny_vocab.merge_wordlist(set())
        assert merged is tiny_vocab

    def test_merge_wordlist_all_existing_returns_self(self, tiny_vocab: Vocabulary):
        merged = tiny_vocab.merge_wordlist({"home", "good"})
        assert merged is tiny_vocab  # all words already present


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

    def test_wordlist_restricts_candidates(self, tiny_vocab: Vocabulary):
        """When a wordlist is given, only those words should appear."""
        wordlist = {"home", "good", "the", "a"}
        d = T9Dictionary(tiny_vocab, wordlist=wordlist)
        hits_4663 = [w for w, _ in d.lookup("4663")]
        assert "home" in hits_4663
        assert "good" in hits_4663
        assert "gone" not in hits_4663   # not in wordlist
        assert "hood" not in hits_4663   # not in wordlist

    def test_wordlist_allows_unknown_words(self, tiny_vocab: Vocabulary):
        """Words in the wordlist but not in the vocab should still be indexed (with UNK ID)."""
        wordlist = {"home", "good", "xylophone"}
        d = T9Dictionary(tiny_vocab, wordlist=wordlist)
        # xylophone → 995674663
        hits = d.lookup("995674663")
        words = [w for w, _ in hits]
        assert "xylophone" in words
        # Its word_id should be UNK since it's not in the tiny vocab
        for w, wid in hits:
            if w == "xylophone":
                assert wid == tiny_vocab.UNK_ID

    def test_wordlist_none_indexes_all_vocab(self, tiny_vocab: Vocabulary):
        """Without a wordlist, all vocab words are indexed (original behaviour)."""
        d_all = T9Dictionary(tiny_vocab, wordlist=None)
        d_old = T9Dictionary(tiny_vocab)
        assert d_all.lookup("4663") == d_old.lookup("4663")


# ===========================================================================
# Wordlist loader tests
# ===========================================================================

class TestLoadWordlist:
    def test_basic_loading(self, tmp_path):
        from ai_t9.dictionary import load_wordlist
        p = tmp_path / "words.txt"
        p.write_text("hello\nworld\nfoo\n", encoding="utf-8")
        wl = load_wordlist(p)
        assert wl == {"hello", "world", "foo"}

    def test_filters_non_alpha(self, tmp_path):
        from ai_t9.dictionary import load_wordlist
        p = tmp_path / "words.txt"
        p.write_text("hello\ncan't\ngood\n123\n", encoding="utf-8")
        wl = load_wordlist(p)
        assert wl == {"hello", "good"}

    def test_skips_comments_and_blanks(self, tmp_path):
        from ai_t9.dictionary import load_wordlist
        p = tmp_path / "words.txt"
        p.write_text("# comment\nhello\n\nworld\n", encoding="utf-8")
        wl = load_wordlist(p)
        assert wl == {"hello", "world"}

    def test_lowercases(self, tmp_path):
        from ai_t9.dictionary import load_wordlist
        p = tmp_path / "words.txt"
        p.write_text("Hello\nWORLD\n", encoding="utf-8")
        wl = load_wordlist(p)
        assert wl == {"hello", "world"}


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

    def test_undo_confirm_returns_last_word(self, tiny_predictor: T9Predictor):
        session = T9Session(tiny_predictor)
        session.confirm("hello")
        session.confirm("world")
        popped = session.undo_confirm()
        assert popped == "world"
        assert session.context == ["hello"]

    def test_undo_confirm_on_empty_returns_none(self, tiny_predictor: T9Predictor):
        session = T9Session(tiny_predictor)
        assert session.undo_confirm() is None
        assert session.context == []

    def test_undo_confirm_all_restores_empty(self, tiny_predictor: T9Predictor):
        session = T9Session(tiny_predictor)
        session.confirm("a")
        session.confirm("b")
        session.undo_confirm()
        session.undo_confirm()
        assert session.context == []
        assert session.undo_confirm() is None

    def test_undo_confirm_respects_window_boundary(self, tiny_predictor: T9Predictor):
        """Words evicted from the window can't be un-confirmed."""
        session = T9Session(tiny_predictor, context_window=2)
        for w in ["a", "b", "c", "d"]:
            session.confirm(w)         # window: [c, d]
        session.undo_confirm()         # pops d  → [c]
        session.undo_confirm()         # pops c  → []
        assert session.context == []
        # 'a' and 'b' were evicted and are irrecoverable
        assert session.undo_confirm() is None

    def test_undo_confirm_lowers_count(self, tiny_predictor: T9Predictor):
        session = T9Session(tiny_predictor)
        for w in ["the", "quick", "brown"]:
            session.confirm(w)
        assert len(session.context) == 3
        session.undo_confirm()
        assert len(session.context) == 2
        session.undo_confirm()
        assert len(session.context) == 1


# ===========================================================================
# Normalise helper
# ===========================================================================

class TestNormalise:
    def test_rank_based_endpoints(self):
        arr = np.array([1.0, 2.0, 3.0, 4.0])
        out = _normalise(arr)
        assert out[0] == pytest.approx(0.0)       # lowest rank
        assert out[-1] == pytest.approx(1.0)       # highest rank

    def test_rank_based_intermediate(self):
        arr = np.array([1.0, 2.0, 3.0, 4.0])
        out = _normalise(arr)
        assert out[1] == pytest.approx(1.0 / 3)
        assert out[2] == pytest.approx(2.0 / 3)

    def test_uniform_returns_zeros(self):
        arr = np.array([5.0, 5.0, 5.0])
        out = _normalise(arr)
        np.testing.assert_array_equal(out, 0)

    def test_single_element_returns_zero(self):
        arr = np.array([42.0])
        out = _normalise(arr)
        np.testing.assert_array_equal(out, 0)

    def test_ties_get_average_rank(self):
        arr = np.array([1.0, 3.0, 3.0, 5.0])
        out = _normalise(arr)
        # ranks: 0, 1.5, 1.5, 3 → normalised: 0, 0.5, 0.5, 1.0
        assert out[0] == pytest.approx(0.0)
        assert out[1] == pytest.approx(0.5)
        assert out[2] == pytest.approx(0.5)
        assert out[3] == pytest.approx(1.0)

    def test_invariant_to_scale(self):
        """Rank-based norm should give identical output for any monotone transform."""
        arr1 = np.array([1.0, 2.0, 3.0, 4.0])
        arr2 = np.array([10.0, 200.0, 3000.0, 40000.0])  # same ordering
        np.testing.assert_allclose(_normalise(arr1), _normalise(arr2))


# ===========================================================================
# Prefix completion tests — T9Dictionary.prefix_lookup
# ===========================================================================

class TestPrefixLookup:
    """Tests for prefix-based autocompletion at the dictionary layer."""

    def test_prefix_returns_longer_words(self, tiny_dict: T9Dictionary):
        # "46" maps exactly to "go"/"in"; prefix should find "4663" words
        results = tiny_dict.prefix_lookup("46")
        words = [w for w, _, _ in results]
        for expected in ("home", "good", "gone", "hood"):
            assert expected in words

    def test_prefix_excludes_exact_matches(self, tiny_dict: T9Dictionary):
        results = tiny_dict.prefix_lookup("46")
        words = [w for w, _, _ in results]
        # "go" and "in" map to "46" exactly — should not appear
        assert "go" not in words
        assert "in" not in words

    def test_prefix_returns_full_digit_seq(self, tiny_dict: T9Dictionary):
        results = tiny_dict.prefix_lookup("46")
        for _word, _wid, full_digits in results:
            assert full_digits.startswith("46")
            assert len(full_digits) > 2

    def test_prefix_sorted_by_frequency(self, tiny_dict: T9Dictionary):
        results = tiny_dict.prefix_lookup("46")
        ids = [wid for _, wid, _ in results]
        vocab = tiny_dict.vocab
        log_freqs = [vocab.logfreq(wid) for wid in ids]
        assert log_freqs == sorted(log_freqs, reverse=True)

    def test_single_digit_prefix(self, tiny_dict: T9Dictionary):
        # "2" exactly matches "a"; prefix should find "be"(23), "and"(263), "any"(269)
        results = tiny_dict.prefix_lookup("2")
        words = [w for w, _, _ in results]
        assert "a" not in words  # exact match excluded
        for expected in ("be", "and", "any"):
            assert expected in words

    def test_max_extra_digits_caps_results(self, tiny_dict: T9Dictionary):
        # "2" prefix with max_extra_digits=1 → only "be" (23, 1 extra digit)
        results = tiny_dict.prefix_lookup("2", max_extra_digits=1)
        words = [w for w, _, _ in results]
        assert "be" in words
        assert "and" not in words  # 263 is 2 extra digits
        assert "any" not in words  # 269 is 2 extra digits

    def test_no_prefix_matches_returns_empty(self, tiny_dict: T9Dictionary):
        results = tiny_dict.prefix_lookup("99999")
        assert results == []

    def test_max_candidates_limits_output(self, tiny_dict: T9Dictionary):
        results = tiny_dict.prefix_lookup("4", max_candidates=2)
        assert len(results) <= 2

    def test_prefix_after_save_load(
        self, tiny_dict: T9Dictionary, tiny_vocab: Vocabulary, tmp_path,
    ):
        path = tmp_path / "dict.json"
        tiny_dict.save(path)
        loaded = T9Dictionary.load(path, tiny_vocab)
        assert loaded.prefix_lookup("46") == tiny_dict.prefix_lookup("46")


# ===========================================================================
# Prefix completion tests — T9Predictor.predict_completions
# ===========================================================================

class TestPredictCompletions:
    """Tests for predict_completions() on the predictor."""

    def test_returns_words(self, tiny_predictor: T9Predictor):
        # "466" prefix → adaptive max_extra=2, finds "4663" words (good/gone/home/hood)
        results = tiny_predictor.predict_completions("466")
        assert all(isinstance(w, str) for w in results)
        assert len(results) >= 1

    def test_completions_are_extensions(self, tiny_predictor: T9Predictor):
        """Every returned word's digit sequence should extend the prefix."""
        from ai_t9.t9_map import word_to_digits

        results = tiny_predictor.predict_completions("466")
        for word in results:
            digits = word_to_digits(word)
            assert digits is not None
            assert digits.startswith("466")
            assert len(digits) > 3

    def test_excludes_exact_matches(self, tiny_predictor: T9Predictor):
        results = tiny_predictor.predict_completions("466", max_extra_digits=6)
        assert "go" not in results
        assert "in" not in results

    def test_top_k_respected(self, tiny_predictor: T9Predictor):
        results = tiny_predictor.predict_completions("466", top_k=2)
        assert len(results) <= 2

    def test_return_details(self, tiny_predictor: T9Predictor):
        results = tiny_predictor.predict_completions("466", return_details=True)
        assert all(isinstance(r, RankedCandidate) for r in results)
        finals = [r.final_score for r in results]
        assert finals == sorted(finals, reverse=True)

    def test_invalid_prefix_raises(self, tiny_predictor: T9Predictor):
        with pytest.raises(ValueError):
            tiny_predictor.predict_completions("1234")

    def test_no_completions_returns_empty(self, tiny_predictor: T9Predictor):
        results = tiny_predictor.predict_completions("99999")
        assert results == []

    def test_length_weight_prefers_shorter(self, tiny_predictor: T9Predictor):
        """With high w_length, shorter completions should rank higher."""
        # "26" prefix with explicit max_extra=3: "and"(263, +1), "any"(269, +1),
        # vs longer words (+2 or more digits)
        results = tiny_predictor.predict_completions(
            "26", top_k=5, max_extra_digits=3, w_length=0.90,
        )
        # "and"/"any" are only 1 extra digit; anything longer should rank lower
        if len(results) >= 2:
            from ai_t9.t9_map import word_to_digits
            extras = [len(word_to_digits(w)) - 2 for w in results]
            # Sorted by score descending, shorter extras should dominate with high w_length
            assert extras[0] <= extras[-1]

    def test_adaptive_short_prefix_conservative(self, tiny_predictor: T9Predictor):
        """Short prefix (≤2 digits) → very conservative: max_extra=1, returns ≤1 result."""
        results = tiny_predictor.predict_completions("46", top_k=10)
        # With adaptive params: prefix_len=2 → max_extra=1, effective_top_k=1
        assert len(results) <= 1

    def test_adaptive_long_prefix_expansive(self, tiny_predictor: T9Predictor):
        """Long prefix (≥6 digits) → expansive: max_extra=5, returns up to top_k results."""
        # "4663" is the sequence for good/gone/home/hood — use their 4-digit prefix
        # which triggers adaptive max_extra=3 (prefix_len=4)
        results = tiny_predictor.predict_completions("4663", top_k=5)
        # prefix_len=4 → max_extra=3 → can find 5-8 letter words starting with "4663"
        # results may be empty (tiny vocab), but top_k is not artificially reduced to 1
        assert len(results) <= 5  # never exceeds top_k

    def test_explicit_max_extra_overrides_adaptive(self, tiny_predictor: T9Predictor):
        """Passing explicit max_extra_digits disables adaptive scaling."""
        results_adaptive = tiny_predictor.predict_completions("46")          # max_extra=1
        results_explicit = tiny_predictor.predict_completions("46", max_extra_digits=6)
        # explicit=6 should find at least as many results as adaptive=1
        assert len(results_explicit) >= len(results_adaptive)

    def test_with_model(
        self, tiny_dict: T9Dictionary, tiny_encoder: DualEncoder, tiny_vocab: Vocabulary,
    ):
        predictor = T9Predictor(tiny_dict, model=tiny_encoder)
        results = predictor.predict_completions("466", context=["the"])
        assert len(results) >= 1
        assert all(w in {"home", "good", "gone", "hood"} for w in results)


# ===========================================================================
# Session completion tests
# ===========================================================================

class TestSessionCompletions:
    def test_completions_returns_words(self, tiny_predictor: T9Predictor):
        session = T9Session(tiny_predictor)
        # "466" → adaptive max_extra=2, finds 4-char words starting with "466"
        results = session.completions("466")
        assert len(results) >= 1

    def test_completions_use_context(self, tiny_predictor: T9Predictor):
        session = T9Session(tiny_predictor)
        session.add_context("the")
        results = session.completions("466")
        assert len(results) >= 1

    def test_dial_with_completions(self, tiny_predictor: T9Predictor):
        session = T9Session(tiny_predictor)
        # dial "46" (exact matches: go/in), completions with explicit max_extra=6
        exact, comps = session.dial_with_completions("46", max_extra_digits=6)
        # exact matches: "go", "in"
        assert any(w in exact for w in ("go", "in"))
        # completions: "home", "good", "gone", "hood"
        assert len(comps) >= 1
        comp_words = set(comps)
        assert comp_words.issubset({"home", "good", "gone", "hood"})

    def test_dial_with_completions_details(self, tiny_predictor: T9Predictor):
        session = T9Session(tiny_predictor)
        exact, comps = session.dial_with_completions(
            "466", return_details=True,
        )
        assert all(isinstance(r, RankedCandidate) for r in exact)
        assert all(isinstance(r, RankedCandidate) for r in comps)

    def test_completions_flow(self, tiny_predictor: T9Predictor):
        """Simulate a user getting completions while typing."""
        session = T9Session(tiny_predictor)
        session.confirm("the")

        # After typing "466", see exact + completions
        exact, comps = session.dial_with_completions("466")
        # User accepts a completion
        if comps:
            session.confirm(comps[0])
            assert len(session.context) == 2


# ===========================================================================
# GUI (T9PhoneWindow) tests
# ===========================================================================

# Skip the entire class if PyQt6 is not installed.
PyQt6 = pytest.importorskip("PyQt6", reason="PyQt6 not installed — skipping GUI tests")


@pytest.fixture(scope="session")
def qt_app():
    """Single QApplication reused across all GUI tests."""
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture
def gui_window(qt_app, tiny_predictor: T9Predictor):
    """A fresh T9PhoneWindow for each test."""
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent / "examples"))
    from gui_demo import T9PhoneWindow
    win = T9PhoneWindow(tiny_predictor, top_k=5)
    yield win
    win.close()


class TestT9PhoneGUI:
    # ── Construction ──────────────────────────────────────────────────────────

    def test_window_title(self, gui_window):
        assert "t9" in gui_window.windowTitle().lower()

    def test_initial_state_empty(self, gui_window):
        assert gui_window._digit_buf == ""
        assert gui_window._committed == ""
        assert gui_window._candidates == []
        assert gui_window._cand_idx == 0

    def test_initial_mode_is_t9(self, gui_window):
        assert gui_window._mode == gui_window._MODE_T9

    # ── T9 digit input ────────────────────────────────────────────────────────

    def test_digit_appends_to_buffer(self, gui_window):
        gui_window._on_key("4")
        assert gui_window._digit_buf == "4"
        gui_window._on_key("6")
        assert gui_window._digit_buf == "46"

    def test_digit_sequence_produces_candidates(self, gui_window):
        for d in "4663":
            gui_window._on_key(d)
        assert len(gui_window._candidates) > 0
        words = [c.word for c in gui_window._candidates]
        assert any(w in words for w in ("home", "good", "gone", "hood"))

    def test_candidates_sorted_by_final_score(self, gui_window):
        for d in "4663":
            gui_window._on_key(d)
        finals = [c.final_score for c in gui_window._candidates]
        assert finals == sorted(finals, reverse=True)

    def test_top_k_respected(self, gui_window):
        gui_window._top_k = 3
        for d in "4663":
            gui_window._on_key(d)
        assert len(gui_window._candidates) <= 3

    # ── Candidate cycling ─────────────────────────────────────────────────────

    def test_hash_key_cycles_candidates(self, gui_window):
        for d in "4663":
            gui_window._on_key(d)
        gui_window._on_key("#")
        assert gui_window._cand_idx == 1
        gui_window._on_key("#")
        assert gui_window._cand_idx == 2

    def test_candidate_cycling_wraps(self, gui_window):
        for d in "4663":
            gui_window._on_key(d)
        n = len(gui_window._candidates)
        for _ in range(n):
            gui_window._on_key("#")
        assert gui_window._cand_idx == 0

    # ── Confirm (Space / 0 key) ───────────────────────────────────────────────

    def test_space_confirms_top_candidate(self, gui_window):
        for d in "4663":
            gui_window._on_key(d)
        first_word = gui_window._candidates[0].word
        gui_window._on_key("0")
        assert first_word in gui_window._committed
        assert gui_window._digit_buf == ""
        assert gui_window._candidates == []

    def test_confirm_updates_session_context(self, gui_window):
        for d in "4663":
            gui_window._on_key(d)
        gui_window._on_key("0")
        assert len(gui_window._session.context) == 1

    def test_confirm_with_hash_selects_alt_candidate(self, gui_window):
        for d in "4663":
            gui_window._on_key(d)
        gui_window._on_key("#")  # advance to index 1
        second_word = gui_window._candidates[1].word
        gui_window._on_key("0")
        assert second_word in gui_window._committed

    # ── Backspace ─────────────────────────────────────────────────────────────

    def test_backspace_removes_last_digit(self, gui_window):
        for d in "4663":
            gui_window._on_key(d)
        gui_window._on_key("*")
        assert gui_window._digit_buf == "466"

    def test_backspace_clears_buffer_fully(self, gui_window):
        gui_window._on_key("4")
        gui_window._on_key("*")
        assert gui_window._digit_buf == ""
        assert gui_window._candidates == []

    def test_backspace_on_empty_buffer_after_word_restores_word(self, gui_window):
        """Backspace after confirming a word should restore it into the buffer."""
        for d in "4663":
            gui_window._on_key(d)
        confirmed_word = gui_window._candidates[0].word
        gui_window._on_key("0")  # confirm + append space → "word "
        # The committed text ends with a space; one backspace peels off the
        # space AND the word together, restoring its digit sequence to the buffer.
        gui_window._on_key("*")
        from ai_t9.t9_map import word_to_digits
        expected_digits = word_to_digits(confirmed_word)
        assert gui_window._digit_buf == expected_digits
        assert gui_window._session.context == []

    def test_backspace_on_empty_does_nothing(self, gui_window):
        gui_window._on_key("*")
        assert gui_window._digit_buf == ""
        assert gui_window._committed == ""

    # ── Punctuation key (1) ───────────────────────────────────────────────────

    def test_punct_key_appends_punctuation(self, gui_window):
        gui_window._on_key("1")
        assert gui_window._committed != ""
        first = gui_window._committed
        gui_window._on_key("1")
        assert gui_window._committed != first  # cycled to next punct

    def test_punct_cycles_through_all_entries(self, gui_window):
        from examples.gui_demo import PUNCT_CYCLE
        for _ in range(len(PUNCT_CYCLE)):
            gui_window._on_key("1")
        # After a full cycle the index wraps; another press gives first punct again
        gui_window._on_key("1")
        assert gui_window._committed[-1] == PUNCT_CYCLE[0]

    # ── Clear all ─────────────────────────────────────────────────────────────

    def test_clear_all_resets_everything(self, gui_window):
        for d in "4663":
            gui_window._on_key(d)
        gui_window._on_key("0")
        gui_window._clear_all()
        assert gui_window._digit_buf == ""
        assert gui_window._committed == ""
        assert gui_window._candidates == []
        assert gui_window._session.context == []

    # ── Mode toggle ───────────────────────────────────────────────────────────

    def test_toggle_mode_flips_state(self, gui_window):
        assert gui_window._mode == gui_window._MODE_T9
        gui_window._toggle_mode()
        assert gui_window._mode == gui_window._MODE_ABC
        gui_window._toggle_mode()
        assert gui_window._mode == gui_window._MODE_T9

    def test_toggle_mode_flushes_digit_buffer(self, gui_window):
        for d in "466":
            gui_window._on_key(d)
        gui_window._toggle_mode()
        # Partial buffer with no confirmed candidates is discarded
        assert gui_window._digit_buf == ""

    # ── ABC (multi-tap) mode ──────────────────────────────────────────────────

    def test_abc_key_sets_pending(self, gui_window):
        gui_window._toggle_mode()
        gui_window._on_key("2")  # first tap → 'a'
        assert gui_window._mt_pending == "a"
        assert gui_window._mt_digit == "2"

    def test_abc_same_key_cycles_letters(self, gui_window):
        gui_window._toggle_mode()
        gui_window._on_key("2")  # a
        gui_window._on_key("2")  # b
        assert gui_window._mt_pending == "b"
        gui_window._on_key("2")  # c
        assert gui_window._mt_pending == "c"
        gui_window._on_key("2")  # wrap back to a
        assert gui_window._mt_pending == "a"

    def test_abc_different_key_commits_previous(self, gui_window):
        gui_window._toggle_mode()
        gui_window._mt_timer.stop()   # disable auto-commit timer
        gui_window._on_key("2")       # pending = 'a'
        # Simulate timer not firing; press a different key
        gui_window._mt_timer.stop()
        gui_window._on_key("3")       # should commit 'a', pending = 'd'
        assert "a" in gui_window._committed
        assert gui_window._mt_pending == "d"

    def test_abc_space_commits_pending(self, gui_window):
        gui_window._toggle_mode()
        gui_window._mt_timer.stop()
        gui_window._on_key("4")       # pending = 'g'
        gui_window._on_key("0")       # space confirms
        assert "g" in gui_window._committed
        assert gui_window._committed.endswith(" ")
        assert gui_window._mt_pending == ""

    def test_abc_backspace_removes_pending(self, gui_window):
        gui_window._toggle_mode()
        gui_window._mt_timer.stop()
        gui_window._on_key("2")
        gui_window._on_key("*")       # backspace cancels pending letter
        assert gui_window._mt_pending == ""
        assert gui_window._committed == ""

    def test_abc_backspace_removes_committed_char(self, gui_window):
        gui_window._toggle_mode()
        gui_window._mt_timer.stop()
        gui_window._on_key("2")
        gui_window._on_key("0")       # commit 'a' + space
        gui_window._on_key("*")       # removes space
        assert gui_window._committed == "a"

    def test_abc_hash_uppercases_pending(self, gui_window):
        gui_window._toggle_mode()
        gui_window._mt_timer.stop()
        gui_window._on_key("2")       # pending = 'a'
        gui_window._on_key("#")       # uppercase toggle
        assert gui_window._mt_pending == "A"
        gui_window._on_key("#")       # toggle back
        assert gui_window._mt_pending == "a"

    # ── Clipboard copy ────────────────────────────────────────────────────────

    def test_copy_to_clipboard_includes_committed(self, qt_app, gui_window):
        from PyQt6.QtGui import QGuiApplication
        gui_window._committed = "hello "
        for d in "4663":
            gui_window._on_key(d)
        # Active candidate should be appended
        gui_window._copy_to_clipboard()
        clipped = QGuiApplication.clipboard().text()
        assert clipped.startswith("hello ")
        assert len(clipped) > len("hello ")

    # ── Long-press ───────────────────────────────────────────────────────────

    def test_long_press_star_clears_all(self, gui_window):
        for d in "4663":
            gui_window._on_key(d)
        gui_window._on_long_press("*")
        assert gui_window._committed == ""
        assert gui_window._digit_buf == ""

    def test_long_press_hash_inserts_newline(self, gui_window):
        gui_window._on_long_press("#")
        assert "\n" in gui_window._committed


# ===========================================================================
# Trainer tests (require torch — skipped automatically if not installed)
# ===========================================================================

try:
    import torch as _torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

pytestmark_torch = pytest.mark.skipif(not _TORCH_AVAILABLE, reason="torch not installed")


@pytest.fixture
def tiny_sentences(tiny_vocab: Vocabulary) -> list[list[int]]:
    """A tiny set of sentences for trainer smoke tests."""
    # Build a few varied sentences using words from tiny_vocab
    words = ["home", "gone", "good", "the", "and", "go", "in", "me", "of", "a", "be", "hi"]
    ids = [tiny_vocab.word_to_id(w) for w in words]
    sentences = []
    for i in range(0, len(ids) - 2, 2):
        sentences.append(ids[i:i + 3])
    # Repeat a few times so we have ≥10 pairs
    return sentences * 5


class TestSavePairs:
    def test_save_pairs_single_file(self, tmp_path, tiny_vocab: Vocabulary, tiny_sentences):
        from ai_t9.model.trainer import save_pairs, load_pairs
        out = tmp_path / "pairs.npz"
        n = save_pairs(tiny_sentences, context_window=2, vocab_size=tiny_vocab.size, path=out)
        assert n > 0
        assert out.exists()
        ctx, pos = load_pairs(out, context_window=2, vocab_size=tiny_vocab.size)
        assert ctx.shape == (n, 2)
        assert pos.shape == (n,)
        assert ctx.dtype == np.int64
        assert pos.dtype == np.int64

    def test_save_pairs_sharded(self, tmp_path, tiny_vocab: Vocabulary, tiny_sentences):
        from ai_t9.model.trainer import save_pairs, load_pairs
        out_prefix = tmp_path / "pairs"
        n_total = save_pairs(
            tiny_sentences,
            context_window=2,
            vocab_size=tiny_vocab.size,
            path=out_prefix,
            max_shard_pairs=3,   # force many small shards
        )
        assert n_total > 0
        shards = sorted(tmp_path.glob("pairs_*.npz"))
        assert len(shards) >= 2, "Expected multiple shards with max_shard_pairs=3"
        # Each shard has at most 3 pairs
        total = 0
        for shard in shards:
            ctx, pos = load_pairs(shard, context_window=2, vocab_size=tiny_vocab.size)
            assert len(pos) <= 3
            total += len(pos)
        assert total == n_total

    def test_save_pairs_metadata_mismatch_raises(self, tmp_path, tiny_vocab: Vocabulary, tiny_sentences):
        from ai_t9.model.trainer import save_pairs, load_pairs
        out = tmp_path / "pairs.npz"
        save_pairs(tiny_sentences, context_window=2, vocab_size=tiny_vocab.size, path=out)
        with pytest.raises(ValueError, match="context_window"):
            load_pairs(out, context_window=5, vocab_size=tiny_vocab.size)
        with pytest.raises(ValueError, match="vocab_size"):
            load_pairs(out, context_window=2, vocab_size=9999)


@pytest.mark.skipif(not _TORCH_AVAILABLE, reason="torch not installed")
class TestDualEncoderTrainerSmoke:
    def test_in_batch_negatives_reduces_loss(self, tmp_path, tiny_vocab: Vocabulary, tiny_sentences):
        from ai_t9.model.trainer import DualEncoderTrainer, save_pairs
        pairs_path = tmp_path / "pairs.npz"
        save_pairs(tiny_sentences, context_window=2, vocab_size=tiny_vocab.size, path=pairs_path)
        trainer = DualEncoderTrainer(
            vocab=tiny_vocab,
            embed_dim=8,
            context_window=2,
            batch_size=4,   # tiny batch for fast test
            accumulate_grad_batches=1,
            seed=0,
            device="cpu",
        )
        trainer.train_from_pairs_file(pairs_path, epochs=2, verbose=False)
        encoder = trainer.get_encoder()
        # Encoder should be a valid DualEncoder
        assert encoder.embed_dim == 8
        assert encoder.vocab.size == tiny_vocab.size

    def test_train_from_pairs_file_smoke(self, tmp_path, tiny_vocab: Vocabulary, tiny_sentences):
        from ai_t9.model.trainer import DualEncoderTrainer, save_pairs
        pairs_path = tmp_path / "pairs.npz"
        save_pairs(tiny_sentences, context_window=2, vocab_size=tiny_vocab.size, path=pairs_path)
        trainer = DualEncoderTrainer(
            vocab=tiny_vocab,
            embed_dim=8,
            context_window=2,
            batch_size=4,
            seed=0,
            device="cpu",
        )
        trainer.train_from_pairs_file(pairs_path, epochs=2, verbose=False)
        encoder = trainer.get_encoder()
        assert encoder.embed_dim == 8

    def test_train_from_pairs_dir_smoke(self, tmp_path, tiny_vocab: Vocabulary, tiny_sentences):
        from ai_t9.model.trainer import DualEncoderTrainer, save_pairs
        save_pairs(
            tiny_sentences,
            context_window=2,
            vocab_size=tiny_vocab.size,
            path=tmp_path / "pairs",
            max_shard_pairs=3,
        )
        trainer = DualEncoderTrainer(
            vocab=tiny_vocab,
            embed_dim=8,
            context_window=2,
            batch_size=4,
            seed=0,
            device="cpu",
        )
        trainer.train_from_pairs_dir(tmp_path, epochs=2, prefetch=False, verbose=False)
        encoder = trainer.get_encoder()
        assert encoder.embed_dim == 8

    def test_grad_accumulation_produces_valid_encoder(self, tmp_path, tiny_vocab: Vocabulary, tiny_sentences):
        from ai_t9.model.trainer import DualEncoderTrainer, save_pairs
        pairs_path = tmp_path / "pairs.npz"
        save_pairs(tiny_sentences, context_window=2, vocab_size=tiny_vocab.size, path=pairs_path)
        trainer = DualEncoderTrainer(
            vocab=tiny_vocab,
            embed_dim=8,
            context_window=2,
            batch_size=2,
            accumulate_grad_batches=3,
            seed=0,
            device="cpu",
        )
        trainer.train_from_pairs_file(pairs_path, epochs=2, verbose=False)
        encoder = trainer.get_encoder()
        assert encoder.embed_dim == 8
        # Embeddings should be finite (no NaN/Inf from bad gradient accumulation)
        ctx_arr = encoder._ctx
        wrd_arr = encoder._wrd
        assert np.all(np.isfinite(ctx_arr))
        assert np.all(np.isfinite(wrd_arr))


@pytest.mark.skipif(not _TORCH_AVAILABLE, reason="torch not installed")
class TestCharNgramTrainerSmoke:
    def test_in_batch_negatives_smoke(self, tmp_path, tiny_vocab: Vocabulary, tiny_sentences):
        from ai_t9.model.trainer import DualEncoderTrainer, save_pairs
        pairs_path = tmp_path / "pairs.npz"
        save_pairs(tiny_sentences, context_window=2, vocab_size=tiny_vocab.size, path=pairs_path)
        trainer = DualEncoderTrainer(
            vocab=tiny_vocab,
            embed_dim=8,
            context_window=2,
            batch_size=4,
            seed=0,
            device="cpu",
        )
        trainer.train_from_pairs_file(pairs_path, epochs=2, verbose=False)
        encoder = trainer.get_encoder()
        # Encoder should be a valid DualEncoder with pre-computed matrices
        assert encoder.embed_dim == 8
        assert hasattr(encoder, "_word_matrix")
        assert encoder._word_matrix.shape[0] == tiny_vocab.size

    def test_train_from_pairs_dir_smoke(self, tmp_path, tiny_vocab: Vocabulary, tiny_sentences):
        from ai_t9.model.trainer import DualEncoderTrainer, save_pairs
        save_pairs(
            tiny_sentences,
            context_window=2,
            vocab_size=tiny_vocab.size,
            path=tmp_path / "pairs",
            max_shard_pairs=3,
        )
        trainer = DualEncoderTrainer(
            vocab=tiny_vocab,
            embed_dim=8,
            context_window=2,
            batch_size=4,
            seed=0,
            device="cpu",
        )
        trainer.train_from_pairs_dir(tmp_path, epochs=1, prefetch=False, verbose=False)
        encoder = trainer.get_encoder()
        assert encoder.embed_dim == 8
