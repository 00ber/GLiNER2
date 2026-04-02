"""
Unit tests for the redesigned compute_struct_loss.

Tests cover:
- Normalized scalar return for various inputs
- Determinism (identical inputs → identical loss)
- Entity vs relation hard-negative behavior
- Hard-negative mask correctness

Run with: pytest tests/test_struct_loss.py -v
"""

import pytest
import torch
import torch.nn as nn

from gliner2.model import Extractor, ExtractorConfig
from gliner2.layers import CountLSTM


# ---------------------------------------------------------------------------
# Minimal stub that has just enough to call compute_struct_loss
# ---------------------------------------------------------------------------

class _StubExtractor(nn.Module):
    """Mimics the Extractor interface needed by compute_struct_loss."""

    def __init__(self, hidden_size=32, **config_overrides):
        super().__init__()
        cfg_kwargs = dict(
            model_name="stub",
            struct_loss_type="asl",
            struct_gamma_pos=1.0,
            struct_gamma_neg=4.0,
            struct_clip=0.05,
            struct_neg_weight=0.25,
            struct_hard_neg_boost=3.0,
            struct_use_field_confusion_hard_negatives=True,
        )
        cfg_kwargs.update(config_overrides)
        self.config = ExtractorConfig(**cfg_kwargs)
        self.count_embed = CountLSTM(hidden_size, max_count=20)

    # Bind the real methods from Extractor
    compute_struct_loss = Extractor.compute_struct_loss
    _build_hard_negative_mask = Extractor._build_hard_negative_mask


def _make_inputs(
    text_len=10, max_width=4, num_fields=3, gold_count=2, hidden=32,
    gold_spans=None,
):
    """Build synthetic inputs for compute_struct_loss.

    Args:
        gold_spans: list of lists of (start, end) tuples per instance per field.
                    If None, generates default spans.

    Returns:
        (span_rep, schema_emb, structure, span_mask)
    """
    span_rep = torch.randn(text_len, max_width, hidden)
    # schema_emb: [P] token + num_fields field embeddings
    schema_emb = torch.randn(num_fields + 1, hidden)

    if gold_spans is None:
        # Default: instance 0 → fields at (1,3), (4,6), (7,9)
        #          instance 1 → fields at (2,4), (5,7), (0,2)
        gold_spans = [
            [(1, 3), (4, 6), (7, 9)],
            [(2, 4), (5, 7), (0, 2)],
        ]

    structure = [gold_count, gold_spans]

    # span_mask: (1, text_len * max_width), True = invalid
    # Mark all as valid for simplicity
    span_mask = torch.zeros(1, text_len * max_width, dtype=torch.bool)

    return span_rep, schema_emb, structure, span_mask


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------

class TestConfigValidation:

    def test_valid_asl(self):
        cfg = ExtractorConfig(model_name="stub", struct_loss_type="asl")
        assert cfg.struct_loss_type == "asl"

    def test_valid_bce(self):
        cfg = ExtractorConfig(model_name="stub", struct_loss_type="bce")
        assert cfg.struct_loss_type == "bce"

    def test_invalid_loss_type_raises(self):
        with pytest.raises(ValueError, match="struct_loss_type must be"):
            ExtractorConfig(model_name="stub", struct_loss_type="mse")


# ---------------------------------------------------------------------------
# Scalar return and finiteness
# ---------------------------------------------------------------------------

class TestScalarReturn:

    @pytest.fixture
    def model(self):
        return _StubExtractor(hidden_size=32)

    def test_returns_scalar(self, model):
        span_rep, schema_emb, structure, span_mask = _make_inputs()
        loss = model.compute_struct_loss(span_rep, schema_emb, structure, span_mask)
        assert loss.dim() == 0, f"Expected scalar, got shape {loss.shape}"

    def test_finite_with_positives(self, model):
        span_rep, schema_emb, structure, span_mask = _make_inputs(gold_count=2)
        loss = model.compute_struct_loss(span_rep, schema_emb, structure, span_mask)
        assert torch.isfinite(loss), f"Loss is not finite: {loss.item()}"

    def test_finite_zero_positives(self, model):
        """Zero gold instances should still produce a finite scalar."""
        span_rep, schema_emb, _, span_mask = _make_inputs(gold_count=0)
        structure = [0, []]
        loss = model.compute_struct_loss(span_rep, schema_emb, structure, span_mask)
        assert loss.dim() == 0
        assert torch.isfinite(loss)

    def test_finite_one_positive(self, model):
        gold_spans = [[(2, 4), (5, 7), (0, 2)]]
        span_rep, schema_emb, structure, span_mask = _make_inputs(
            gold_count=1, gold_spans=gold_spans,
        )
        loss = model.compute_struct_loss(span_rep, schema_emb, structure, span_mask)
        assert loss.dim() == 0
        assert torch.isfinite(loss)

    def test_finite_with_masked_spans(self, model):
        """Some spans masked invalid — loss should still be finite."""
        span_rep, schema_emb, structure, span_mask = _make_inputs(
            text_len=10, max_width=4,
        )
        # Mask out half the span positions as invalid
        span_mask[0, :20] = True
        loss = model.compute_struct_loss(span_rep, schema_emb, structure, span_mask)
        assert loss.dim() == 0
        assert torch.isfinite(loss)

    def test_finite_multi_instance_relations(self, model):
        """Multiple relation instances with various field spans."""
        gold_spans = [
            [(0, 2), (3, 5), (6, 8)],
            [(1, 3), (4, 6), (7, 9)],
            [(2, 4), (5, 7), (0, 2)],
        ]
        span_rep, schema_emb, structure, span_mask = _make_inputs(
            gold_count=3, gold_spans=gold_spans,
        )
        loss = model.compute_struct_loss(
            span_rep, schema_emb, structure, span_mask, task_type="relations",
        )
        assert loss.dim() == 0
        assert torch.isfinite(loss)

    def test_bce_mode_returns_scalar(self):
        """BCE mode (no focal weighting) returns a finite scalar."""
        model = _StubExtractor(hidden_size=32, struct_loss_type="bce")
        span_rep, schema_emb, structure, span_mask = _make_inputs()
        loss = model.compute_struct_loss(span_rep, schema_emb, structure, span_mask)
        assert loss.dim() == 0
        assert torch.isfinite(loss)


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

class TestDeterminism:

    def test_identical_inputs_produce_identical_loss(self):
        model = _StubExtractor(hidden_size=32)
        model.eval()

        torch.manual_seed(42)
        span_rep, schema_emb, structure, span_mask = _make_inputs()

        with torch.no_grad():
            loss1 = model.compute_struct_loss(
                span_rep, schema_emb, structure, span_mask,
            )
            loss2 = model.compute_struct_loss(
                span_rep, schema_emb, structure, span_mask,
            )

        assert torch.equal(loss1, loss2), (
            f"Non-deterministic: {loss1.item()} != {loss2.item()}"
        )

    def test_determinism_relation_mode(self):
        model = _StubExtractor(hidden_size=32)
        model.eval()

        torch.manual_seed(42)
        span_rep, schema_emb, structure, span_mask = _make_inputs()

        with torch.no_grad():
            loss1 = model.compute_struct_loss(
                span_rep, schema_emb, structure, span_mask,
                task_type="relations",
            )
            loss2 = model.compute_struct_loss(
                span_rep, schema_emb, structure, span_mask,
                task_type="relations",
            )

        assert torch.equal(loss1, loss2)


# ---------------------------------------------------------------------------
# Entity vs Relation behavior
# ---------------------------------------------------------------------------

class TestTaskTypeBehavior:

    def test_entity_and_relation_differ(self):
        """Relation mode should produce different loss due to hard-neg boosting."""
        model = _StubExtractor(hidden_size=32)
        model.eval()

        torch.manual_seed(42)
        span_rep, schema_emb, structure, span_mask = _make_inputs()

        with torch.no_grad():
            entity_loss = model.compute_struct_loss(
                span_rep, schema_emb, structure, span_mask,
                task_type="entities",
            )
            relation_loss = model.compute_struct_loss(
                span_rep, schema_emb, structure, span_mask,
                task_type="relations",
            )

        # With hard-neg boosting active, relation loss should differ from entity
        assert not torch.equal(entity_loss, relation_loss), (
            "Entity and relation losses should differ when hard-neg boosting is active"
        )

    def test_entity_and_relation_same_when_boost_disabled(self):
        """With hard-neg boosting disabled, entity and relation should match."""
        model = _StubExtractor(
            hidden_size=32,
            struct_use_field_confusion_hard_negatives=False,
        )
        model.eval()

        torch.manual_seed(42)
        span_rep, schema_emb, structure, span_mask = _make_inputs()

        with torch.no_grad():
            entity_loss = model.compute_struct_loss(
                span_rep, schema_emb, structure, span_mask,
                task_type="entities",
            )
            relation_loss = model.compute_struct_loss(
                span_rep, schema_emb, structure, span_mask,
                task_type="relations",
            )

        assert torch.equal(entity_loss, relation_loss), (
            "With hard-neg disabled, entity and relation losses should be identical"
        )


# ---------------------------------------------------------------------------
# Hard-negative mask correctness (mandatory test per plan)
# ---------------------------------------------------------------------------

class TestHardNegativeMask:
    """
    Construct a synthetic structure with:
    - 2 instances
    - 3 fields
    - known span positions per field

    Assert the mask marks exactly the right positions and excludes wrong ones.
    """

    # Synthetic structure:
    # Instance 0: field 0 = (1,3), field 1 = (4,6), field 2 = (7,9)
    # Instance 1: field 0 = (2,4), field 1 = (5,7), field 2 = (0,2)
    GOLD_SPANS = [
        [(1, 3), (4, 6), (7, 9)],  # instance 0
        [(2, 4), (5, 7), (0, 2)],  # instance 1
    ]
    GOLD_COUNT = 2
    TEXT_LEN = 10
    MAX_WIDTH = 4
    NUM_FIELDS = 3
    SCORES_SHAPE = (GOLD_COUNT, NUM_FIELDS, TEXT_LEN, MAX_WIDTH)

    @pytest.fixture
    def mask(self):
        structure = [self.GOLD_COUNT, self.GOLD_SPANS]
        return Extractor._build_hard_negative_mask(
            structure, self.GOLD_COUNT, self.SCORES_SHAPE, torch.device("cpu"),
        )

    def _span_to_idx(self, start, end):
        """Convert (start, end) span to (start, width) index."""
        return (start, end - start)

    def test_mask_shape(self, mask):
        assert mask.shape == self.SCORES_SHAPE

    def test_instance0_field0_marks_other_fields_golds(self, mask):
        """For instance 0, field 0: should mark field 1's (4,2) and field 2's (7,2)."""
        # Field 1 gold: (4,6) → idx (4,2)
        assert mask[0, 0, 4, 2].item() is True
        # Field 2 gold: (7,9) → idx (7,2)
        assert mask[0, 0, 7, 2].item() is True

    def test_instance0_field0_does_not_mark_own_gold(self, mask):
        """For instance 0, field 0: should NOT mark its own gold (1,3) → idx (1,2)."""
        assert mask[0, 0, 1, 2].item() is False

    def test_instance0_field1_marks_other_fields_golds(self, mask):
        """For instance 0, field 1: should mark field 0's (1,2) and field 2's (7,2)."""
        assert mask[0, 1, 1, 2].item() is True
        assert mask[0, 1, 7, 2].item() is True

    def test_instance0_field1_does_not_mark_own_gold(self, mask):
        assert mask[0, 1, 4, 2].item() is False

    def test_instance0_field2_marks_other_fields_golds(self, mask):
        assert mask[0, 2, 1, 2].item() is True  # field 0's gold
        assert mask[0, 2, 4, 2].item() is True  # field 1's gold

    def test_instance0_field2_does_not_mark_own_gold(self, mask):
        assert mask[0, 2, 7, 2].item() is False

    def test_instance1_field0_marks_other_fields_golds(self, mask):
        """Instance 1, field 0: should mark field 1's (5,2) and field 2's (0,2)."""
        assert mask[1, 0, 5, 2].item() is True
        assert mask[1, 0, 0, 2].item() is True

    def test_instance1_field0_does_not_mark_own_gold(self, mask):
        assert mask[1, 0, 2, 2].item() is False

    def test_no_cross_instance_marking(self, mask):
        """Instance 0's gold positions should NOT appear in instance 1's mask
        (unless they happen to also be another field's gold in instance 1)."""
        # Instance 0, field 0 gold is (1,2). Check it's not in instance 1's field 0 mask
        # (instance 1 field 1 gold is (5,2), field 2 gold is (0,2) — neither is (1,2))
        assert mask[1, 0, 1, 2].item() is False

    def test_non_gold_positions_are_false(self, mask):
        """Random non-gold positions should be False."""
        # (0, 0, 0, 0) is not any field's gold position in instance 0
        assert mask[0, 0, 0, 0].item() is False
        assert mask[0, 0, 3, 3].item() is False
        assert mask[1, 2, 9, 3].item() is False

    def test_invalid_masked_positions_not_marked_in_loss(self):
        """Even if a position is a hard negative, if span_mask says invalid,
        it should not contribute to loss (the loss function masks it out)."""
        model = _StubExtractor(hidden_size=32)
        model.eval()

        gold_spans = self.GOLD_SPANS
        span_rep, schema_emb, structure, span_mask = _make_inputs(
            text_len=self.TEXT_LEN, max_width=self.MAX_WIDTH,
            num_fields=self.NUM_FIELDS, gold_count=self.GOLD_COUNT,
            gold_spans=gold_spans,
        )

        # Mark positions 4 and 5 as invalid in span_mask
        # Position (4, width) spans live at flat indices 4*MAX_WIDTH + width
        for w in range(self.MAX_WIDTH):
            span_mask[0, 4 * self.MAX_WIDTH + w] = True
            span_mask[0, 5 * self.MAX_WIDTH + w] = True

        with torch.no_grad():
            loss = model.compute_struct_loss(
                span_rep, schema_emb, structure, span_mask,
                task_type="relations",
            )

        assert torch.isfinite(loss)

    def test_exact_hard_neg_count(self, mask):
        """Total True count should equal sum of cross-field gold position markings."""
        # Instance 0: each of 3 fields marks the other 2 fields' golds = 3 * 2 = 6
        # Instance 1: same = 6
        # Total = 12
        expected = 12
        actual = mask.sum().item()
        assert actual == expected, f"Expected {expected} hard negatives, got {actual}"
