"""YOLO11 detection architecture plus registered training-only GRL branches."""
from copy import deepcopy

import torch
from ultralytics.nn.tasks import DetectionModel

from .adaptation import MultiscaleDomainHeads, capture_features
from .config import grl_settings


class GRLDetectionModel(DetectionModel):
    def __init__(self, cfg="yolo11s.yaml", nc=None, grl=None, verbose=True):
        super().__init__(cfg=cfg, nc=nc, verbose=verbose)
        self.grl_config = grl_settings({**self.yaml.get("grl", {}), **(grl or {})})
        layers = self.grl_config["layers"]
        if max(layers) >= len(self.yaml["backbone"]):
            raise ValueError("GRL feature layers must belong to the YOLO11 backbone")
        self.grl_coefficient = self.grl_config["lambda"]
        was_training = self.training
        self.eval()
        with torch.no_grad():
            features = self.backbone_features(torch.zeros(1, 3, 128, 128))
        self.train(was_training)
        self.domain_heads = MultiscaleDomainHeads(
            [f.shape[1] for f in features], self.grl_config["hidden_channels"],
            self.grl_config["norm"], self.grl_config["groups"],
        )
        self.last_domain_loss = 0.0
        self.last_total_mean = 0.0

    def backbone_features(self, images):
        """Target images stop at the backbone; the neck/head remain source-only."""
        layers = self.grl_config["layers"]
        features, saved = {}, []
        x = images
        for index, module in enumerate(self.model):
            if module.f != -1:
                x = saved[module.f] if isinstance(module.f, int) else [
                    x if source == -1 else saved[source] for source in module.f
                ]
            x = module(x)
            saved.append(x if module.i in self.save else None)
            if index in layers:
                features[index] = x
            if index == max(layers):
                break
        return [features[index] for index in layers]

    def loss(self, batch, preds=None):
        # Standard source-only loss is also used by the Ultralytics validator.
        if not self.training or "target_img" not in batch:
            return super().loss(batch, preds)
        predictions, source = capture_features(self, batch["img"], self.grl_config["layers"])
        detection_loss, items = super().loss(batch, predictions)
        target = self.backbone_features(batch["target_img"])
        domain_loss = self.domain_heads(source, target, self.grl_coefficient)
        # Keep Ultralytics' summed-over-images detection loss exactly unchanged.
        batch_size = batch["img"].shape[0]
        total = detection_loss + batch_size * self.grl_config["domain_weight"] * domain_loss
        self.last_domain_loss = float(domain_loss.detach())
        self.last_total_mean = float(total.detach()) / batch_size
        return total, items


def plain_detector(model):
    """Export the detector as a standard Ultralytics class, without custom modules."""
    architecture = deepcopy(model.yaml)
    architecture.pop("grl", None)
    detector = DetectionModel(architecture, nc=model.model[-1].nc, verbose=False)
    detector.load(model, verbose=False)
    detector.names = deepcopy(model.names)
    detector.args = deepcopy(model.args)
    return detector.cpu().float().eval()
