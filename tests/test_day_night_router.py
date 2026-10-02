import unittest

import torch

from datasets.day_night import label_from_filename
from models.day_night_router import (
    build_day_night_router,
    prediction_to_route,
)


class DayNightRouterTest(unittest.TestCase):
    def test_filename_labels_are_only_training_targets(self):
        self.assertEqual(label_from_filename('IMG_1.jpg'), 0)
        self.assertEqual(label_from_filename('night_IMG_1.jpg'), 1)

    def test_day_prediction_routes_to_lora(self):
        domain, route, confidence = prediction_to_route(
            torch.tensor([[4.0, 1.0]]))
        self.assertEqual((domain, route), ('day', 'lora'))
        self.assertGreater(confidence, 0.5)

    def test_night_prediction_routes_to_dora(self):
        domain, route, confidence = prediction_to_route(
            torch.tensor([[1.0, 4.0]]))
        self.assertEqual((domain, route), ('night', 'dora'))
        self.assertGreater(confidence, 0.5)

    def test_model_output_and_frozen_features(self):
        model = build_day_night_router(
            pretrained=False, freeze_features=True)
        self.assertFalse(any(
            parameter.requires_grad
            for parameter in model.features.parameters()))
        self.assertTrue(any(
            parameter.requires_grad
            for parameter in model.classifier.parameters()))
        model.eval()
        with torch.inference_mode():
            output = model(torch.randn(2, 3, 224, 224))
        self.assertEqual(output.shape, (2, 2))


if __name__ == '__main__':
    unittest.main()
