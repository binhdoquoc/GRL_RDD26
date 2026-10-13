"""Analytical checks for domain BCE and domain-only backbone updates."""
import math

import torch
from torch import nn
from ultralytics.cfg import get_cfg

from grl_yolo.adaptation import MultiscaleDomainHeads, capture_features
from train import make_detector


def controlled_logits(scales=3):
    """Use prescribed classifier outputs to test the loss independently of features."""
    heads = MultiscaleDomainHeads([32] * scales)
    heads.heads = nn.ModuleList([nn.Identity() for _ in range(scales)])
    return heads


def test_domain_loss_zero_logits_is_log_two():
    # Actual classifiers with zero parameters produce probability 0.5 everywhere.
    heads = MultiscaleDomainHeads([32, 64, 128])
    with torch.no_grad():
        for parameter in heads.parameters():
            parameter.zero_()
    source = [torch.randn(2, c, 4, 4) for c in (32, 64, 128)]
    target = [torch.randn(3, c, 2, 2) for c in (32, 64, 128)]
    loss = heads(source, target)
    torch.testing.assert_close(loss, torch.tensor(math.log(2)))


def test_domain_loss_balances_domains_and_averages_scales():
    heads = controlled_logits()
    source_logits = (0.0, 2.0, -1.0)
    target_logits = (-3.0, 0.5, 4.0)
    # Different batch sizes and pixel counts must not change either domain's weight.
    source = [torch.full((2, 1, 8, 8), value) for value in source_logits]
    target = [torch.full((5, 1, 2, 2), value) for value in target_logits]
    expected = sum(
        0.5 * (math.log1p(math.exp(-s)) + math.log1p(math.exp(t)))
        for s, t in zip(source_logits, target_logits)
    ) / 3
    torch.testing.assert_close(heads(source, target), torch.tensor(expected))


def test_domain_loss_label_convention_and_extreme_logits():
    heads = controlled_logits(scales=1)
    positive = [torch.full((2, 1, 4, 4), 8.0)]
    negative = [torch.full((2, 1, 4, 4), -8.0)]
    # source=1 wants positive logits; target=0 wants negative logits.
    good = heads(positive, negative)
    wrong = heads(negative, positive)
    torch.testing.assert_close(good, torch.tensor(math.log1p(math.exp(-8))))
    torch.testing.assert_close(wrong, torch.tensor(math.log1p(math.exp(8))))
    assert good < 0.001 and wrong > 8.0
    # BCEWithLogits must remain finite even when predictions are confidently wrong.
    extreme = heads([torch.full((2, 1, 4, 4), -1000.0)],
                    [torch.full((2, 1, 4, 4), 1000.0)])
    torch.testing.assert_close(extreme, torch.tensor(1000.0))


def test_domain_loss_alone_updates_backbone_and_grl_zero_blocks_it():
    torch.manual_seed(7)
    torch.set_num_threads(2)
    model = make_detector({0: "damage"}, weights=None)
    model.args = get_cfg()
    model.train()
    source_images = torch.rand(2, 3, 64, 64)
    target_images = torch.rand(2, 3, 64, 64)
    _, source = capture_features(model, source_images)
    _, target = capture_features(model, target_images)
    heads = MultiscaleDomainHeads([f.shape[1] for f in source])
    backbone_weight = model.model[0].conv.weight
    optimizer = torch.optim.SGD(list(model.parameters()) + list(heads.parameters()), lr=0.01)
    before = backbone_weight.detach().clone()
    # No detection loss or labels: the update can only come from domain loss.
    heads(source, target, coefficient=0.1).backward()
    assert backbone_weight.grad is not None and backbone_weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in model.model[-1].parameters())
    optimizer.step()
    assert not torch.equal(before, backbone_weight.detach())

    optimizer.zero_grad(set_to_none=True)
    _, source = capture_features(model, source_images)
    _, target = capture_features(model, target_images)
    heads(source, target, coefficient=0.0).backward()
    torch.testing.assert_close(backbone_weight.grad, torch.zeros_like(backbone_weight))
    assert sum(p.grad.abs().sum().item() for p in heads.parameters() if p.grad is not None) > 0
