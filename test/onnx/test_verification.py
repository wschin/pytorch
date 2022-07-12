# Owner(s): ["module: onnx"]

import torch
from torch.onnx import _experimental, verification
from torch.testing._internal import common_utils


class TestVerification(common_utils.TestCase):
    def setUp(self) -> None:
        super().setUp()
        torch.manual_seed(0)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(0)

    def test_check_export_model_diff_returns_diff_when_constant_mismatch(self):
        class UnexportableModel(torch.nn.Module):
            def forward(self, x, y):
                # tensor.data() will be exported as a constant,
                # leading to wrong model output under different inputs.
                return x + y.data

        test_input_groups = [
            ((torch.randn(2, 3), torch.randn(2, 3)), {}),
            ((torch.randn(2, 3), torch.randn(2, 3)), {}),
        ]

        results = verification.check_export_model_diff(
            UnexportableModel(), test_input_groups
        )
        self.assertRegex(
            results,
            r"Graph diff:(.|\n)*"
            r"First diverging operator:(.|\n)*"
            r"prim::Constant(.|\n)*"
            r"Former source location:(.|\n)*"
            r"Latter source location:",
        )

    def test_check_export_model_diff_returns_diff_when_dynamic_controlflow_mismatch(
        self,
    ):
        class UnexportableModel(torch.nn.Module):
            def forward(self, x, y):
                for i in range(x.size(0)):
                    y = x[i] + y
                return y

        test_input_groups = [
            ((torch.randn(2, 3), torch.randn(2, 3)), {}),
            ((torch.randn(4, 3), torch.randn(2, 3)), {}),
        ]

        export_options = _experimental.ExportOptions(
            input_names=["x", "y"], dynamic_axes={"x": [0]}
        )
        results = verification.check_export_model_diff(
            UnexportableModel(), test_input_groups, export_options
        )
        self.assertRegex(
            results,
            r"Graph diff:(.|\n)*"
            r"First diverging operator:(.|\n)*"
            r"prim::Constant(.|\n)*"
            r"Latter source location:(.|\n)*",
        )

    def test_check_export_model_diff_returns_empty_when_correct_export(self):
        class SupportedModel(torch.nn.Module):
            def forward(self, x, y):
                return x + y

        test_input_groups = [
            ((torch.randn(2, 3), torch.randn(2, 3)), {}),
            ((torch.randn(2, 3), torch.randn(2, 3)), {}),
        ]

        results = verification.check_export_model_diff(
            SupportedModel(), test_input_groups
        )
        self.assertEqual(results, "")
