"""Apply local GRL changes using the standard Ultralytics model/trainer API."""
from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer

from .model import GRLDetectionModel
from .trainer import GRLTrainer


class GRLYOLO(YOLO):
    @property
    def task_map(self):
        mapping = super().task_map
        mapping["detect"] = {**mapping["detect"], "model": GRLDetectionModel, "trainer": GRLTrainer}
        return mapping

    def train(self, trainer=None, **kwargs):
        settings = getattr(self.model, "grl_config", {})
        if trainer is None and (not settings.get("enabled", True) or settings.get("domain_weight", 1.0) == 0):
            trainer = DetectionTrainer
        return super().train(trainer=trainer, **kwargs)
