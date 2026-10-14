"""Audit the actual training model's gradient partition and native optimizer."""
from copy import deepcopy

import torch
from ultralytics.cfg import get_cfg

from grl_yolo.model import GRLDetectionModel
from grl_yolo.trainer import GRLTrainer


def batch():
    return {"img": torch.rand(2, 3, 64, 64), "target_img": torch.rand(2, 3, 64, 64),
            "batch_idx": torch.tensor([0, 1]), "cls": torch.zeros(2, 1),
            "bboxes": torch.tensor([[.5, .5, .25, .25]] * 2)}


def model():
    result = GRLDetectionModel("yolo11s.yaml", nc=1, verbose=False)
    result.args = get_cfg()
    return result.train()


def test_full_model_gradient_partition():
    torch.set_num_threads(2)
    torch.manual_seed(31)
    native = model()
    ordinary, reversed_model = deepcopy(native), deepcopy(native)
    inputs = batch()
    # -1 is used only as an audit reference: GRL backward becomes ordinary +1.
    ordinary.grl_coefficient = -1
    reversed_model.grl_coefficient = .1
    native({k: v for k, v in inputs.items() if k != "target_img"})[0].backward()
    ordinary(inputs)[0].backward()
    reversed_model(inputs)[0].backward()
    reference = dict(ordinary.named_parameters())
    detection = dict(native.named_parameters())
    for name, parameter in reversed_model.named_parameters():
        ref = reference[name].grad
        det = detection[name].grad
        if name.startswith("domain_heads."):
            # The classifier minimizes BCE with normal gradients, independent of lambda.
            torch.testing.assert_close(parameter.grad, ref)
        elif parameter.grad is not None:
            index = int(name.split(".")[1])
            expected = det - .1 * (ref - det) if index <= 10 else det
            torch.testing.assert_close(parameter.grad, expected, rtol=3e-3, atol=3e-5)


def test_target_bypasses_neck_and_detect_and_native_optimizer_updates_heads():
    torch.set_num_threads(2)
    torch.manual_seed(32)
    detector = model()
    calls = []
    handles = [module.register_forward_hook(lambda m, a, o, i=i: calls.append(i))
               for i, module in enumerate(detector.model) if i >= 11]
    try:
        loss, _ = detector(batch())
    finally:
        for handle in handles:
            handle.remove()
    assert calls == list(range(11, 24))  # one source pass; zero target passes
    trainer = object.__new__(GRLTrainer)
    optimizer = trainer.build_optimizer(detector, name="SGD", lr=.001, decay=0)
    ids = {id(p) for group in optimizer.param_groups for p in group["params"]}
    assert all(id(p) in ids for p in detector.domain_heads.parameters())
    backbone = detector.model[0].conv.weight
    head = detector.domain_heads.heads[0][0].weight
    before = [p.detach().clone() for p in (backbone, head)]
    loss.backward()
    optimizer.step()
    assert all(not torch.equal(old, new) for old, new in zip(before, (backbone, head)))
