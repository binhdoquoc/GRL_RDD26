import torch

from grl_yolo.adaptation import MultiscaleDomainHeads, capture_features, reverse_gradient
from grl_yolo.data import TargetImages
from train import make_detector, save_detector
from ultralytics import YOLO
from ultralytics.cfg import get_cfg


def test_reverse_gradient():
    x = torch.tensor([1.0, -2.0], requires_grad=True)
    y = reverse_gradient(x, 0.1)
    torch.testing.assert_close(y, x)
    (y * torch.tensor([3.0, 4.0])).sum().backward()
    torch.testing.assert_close(x.grad, torch.tensor([-0.3, -0.4]))


def test_only_feature_gradient_reverses():
    torch.manual_seed(2)
    heads = MultiscaleDomainHeads([32, 64, 128])
    src = [torch.randn(2, c, 4, 4, requires_grad=True) for c in (32, 64, 128)]
    tgt = [torch.randn(2, c, 4, 4, requires_grad=True) for c in (32, 64, 128)]
    heads(src, tgt, -1).backward()  # ordinary gradient for reference
    src_grads = [x.grad.clone() for x in src]
    param_grads = [p.grad.clone() for p in heads.parameters()]
    heads.zero_grad()
    for x in src + tgt:
        x.grad = None
    heads(src, tgt, 0.1).backward()
    for x, g in zip(src, src_grads):
        torch.testing.assert_close(x.grad, -0.1 * g)
    for p, g in zip(heads.parameters(), param_grads):
        torch.testing.assert_close(p.grad, g)


def test_yolo11s_backward_and_checkpoint(tmp_path):
    torch.set_num_threads(2)
    model = make_detector({0: "damage"}, weights=None)
    model.args = get_cfg()
    model.train()
    batch = {"img": torch.rand(2, 3, 64, 64), "batch_idx": torch.tensor([0, 1]),
             "cls": torch.zeros(2, 1), "bboxes": torch.tensor([[.5, .5, .25, .25]] * 2)}
    (loss, _), src = capture_features(model, batch)
    _, tgt = capture_features(model, torch.rand(2, 3, 64, 64))
    assert [f.shape[-1] for f in src] == [8, 4, 2]
    heads = MultiscaleDomainHeads([f.shape[1] for f in src])
    total = loss / 2 + heads(src, tgt)
    total.backward()
    assert torch.isfinite(total)
    assert model.model[0].conv.weight.grad.abs().sum() > 0
    assert all(p.grad is not None for p in heads.parameters())
    assert not any(layer._forward_hooks for layer in model.model)
    saved = tmp_path / "detector.pt"
    save_detector(model, saved, model.args)
    wrapper = YOLO(saved)
    loaded = wrapper.model.eval()
    with torch.no_grad():
        prediction = loaded(torch.rand(1, 3, 64, 64))
    assert prediction[0].shape[1] == 5  # four box coordinates + one class
    import numpy as np
    results = wrapper.predict(np.zeros((64, 64, 3), dtype=np.uint8), imgsz=64, device="cpu", verbose=False)
    assert len(results) == 1 and results[0].names == {0: "damage"}


def test_target_ignores_labels(tmp_path):
    import cv2
    import numpy as np
    images = tmp_path / "images"
    labels = tmp_path / "labels"
    images.mkdir()
    labels.mkdir()
    cv2.imwrite(str(images / "sample.jpg"), np.zeros((32, 48, 3), dtype=np.uint8))
    (labels / "sample.txt").write_text("deliberately invalid label", encoding="utf-8")
    dataset = TargetImages(images, 64)
    assert dataset[0].shape == (3, 64, 64)


