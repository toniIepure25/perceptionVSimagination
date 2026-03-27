def test_canonical_imports():
    from fmri2img import evaluation, export, preprocessing, roi, targets, workflows
    from fmri2img.data import CanonicalDecoderDataset, PairedConditionBatchSampler
    from fmri2img.models import (
        DecoderBatch,
        DecoderConfig,
        ROISpecificHierarchicalEncoder,
        SharedPrivateMultitaskDecoder,
    )
    from fmri2img.training import CanonicalLossWeights, SharedPrivateTrainer

    assert evaluation is not None
    assert export is not None
    assert preprocessing is not None
    assert roi is not None
    assert targets is not None
    assert workflows is not None
    assert CanonicalDecoderDataset is not None
    assert PairedConditionBatchSampler is not None
    assert DecoderBatch is not None
    assert DecoderConfig is not None
    assert ROISpecificHierarchicalEncoder is not None
    assert SharedPrivateMultitaskDecoder is not None
    assert CanonicalLossWeights is not None
    assert SharedPrivateTrainer is not None
