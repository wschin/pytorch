# Owner(s): ["module: onnx"]
from __future__ import annotations

import copy
import dataclasses
import os
import sys
from typing import Tuple

import onnxruntime

import torch
import torch._dynamo.backends.registry
from parameterized import parameterized
from torch import nn
from torch.onnx import (
    _OrtBackend as OrtBackend,
    _OrtBackendOptions as OrtBackendOptions,
    ExportOptions,
)

from torch.testing._internal import common_utils

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import onnx_test_common


def make_aot_ort(dynamic: bool = False):
    ort_backend = OrtBackend(
        options=OrtBackendOptions(
            export_options=ExportOptions(
                dynamic_shapes=dynamic,
            )
        )
    )
    return ort_backend, ort_backend


class TestDynamoWithONNXRuntime(onnx_test_common._TestONNXRuntime):
    def setUp(self):
        super().setUp()
        torch._dynamo.reset()
        OrtBackend.clear_cached_instances()

    def tearDown(self):
        super().tearDown()
        torch._dynamo.reset()
        OrtBackend.clear_cached_instances()

    def test_torch_compile_backend_registration(self):
        self.assertIn("onnxrt", torch._dynamo.backends.registry.list_backends())
        backend = torch._dynamo.backends.registry.lookup_backend("onnxrt")
        self.assertEqual(backend.__module__, "torch.onnx._internal.onnxruntime")

    def _test_torch_compile_backend_caching_assert_reused(
        self, options: OrtBackendOptions
    ):
        self.assertFalse(OrtBackend.get_cached_instances())  # assert setUp/tearDown
        new_backend = OrtBackend.get_cached_instance_for_options(options)
        reused_backend = OrtBackend.get_cached_instance_for_options(options)
        self.assertEqual(len(OrtBackend.get_cached_instances()), 1)
        self.assertIs(reused_backend, new_backend)
        if options is None or options.ort_session_options is None:
            # OrtBackendOptions.ort_session_options is a pybind11 object that
            # cannot be pickled via dataclasses.asdict
            self.assertEqual(
                new_backend,
                OrtBackend.get_cached_instance_for_options(
                    dataclasses.asdict(options) if options else None
                ),
            )

    @parameterized.expand(
        [
            (None,),
            (OrtBackendOptions(),),
            (OrtBackendOptions(use_aot_autograd=True),),
            (OrtBackendOptions(use_aot_autograd=False),),
            (OrtBackendOptions(preallocate_output=True),),
            (OrtBackendOptions(preallocate_output=False),),
            (OrtBackendOptions(infer_execution_providers=True),),
            (OrtBackendOptions(infer_execution_providers=False),),
            (OrtBackendOptions(preferred_execution_providers=["A", "B", "C"]),),
            (
                OrtBackendOptions(
                    preferred_execution_providers=["A", "B", ("C", {"option": "value"})]
                ),
            ),
            (OrtBackendOptions(default_execution_providers=["Something"]),),
            (
                OrtBackendOptions(
                    export_options=ExportOptions(
                        dynamic_shapes=True,
                    )
                ),
            ),
            (
                OrtBackendOptions(
                    use_aot_autograd=False,
                    export_options=ExportOptions(
                        op_level_debug=True,
                        dynamic_shapes=True,
                    ),
                ),
            ),
        ]
    )
    def test_torch_compile_backend_caching_assert_reused(
        self, options: OrtBackendOptions
    ):
        self._test_torch_compile_backend_caching_assert_reused(options)

    @parameterized.expand(
        [
            (OrtBackendOptions(ort_session_options=onnxruntime.SessionOptions()),),
        ]
    )
    def test_torch_compile_backend_caching_assert_not_reused(
        self, options: OrtBackendOptions
    ):
        with self.assertRaises(AssertionError):
            self._test_torch_compile_backend_caching_assert_reused(options)

    def _test_model_numerically(
        self,
        model,
        dynamo_backend,
        example_args_collection,
        fullgraph: bool = False,
    ):
        """Run original and compiled model and compare the results.

        Args:
            model: The model to test.
            dynamo_backend: The dynamo backend to use. Here we use string `onnxrt` or
              the first returned value of `make_aot_ort(dynamic=True)`.
            example_args_collection: A tuple of example arguments to test. E.g.,
                (
                  (torch.randn(2), torch.randn(2)),
                  (torch.randn(4), torch.randn(4)),
                )
              if you want to test
                model(torch.randn(2), torch.randn(2)) and
                model(torch.randn(4), torch.randn(4))
              .
        """
        compiled_model = torch.compile(
            model if not isinstance(model, torch.nn.Module) else copy.deepcopy(model),
            backend=dynamo_backend,
            dynamic=True,
            fullgraph=fullgraph,
        )

        for example_args in example_args_collection:
            baseline_result = model(*example_args)
            result = compiled_model(*example_args)
            if isinstance(baseline_result, torch.Tensor):
                torch.testing.assert_close(baseline_result, result)
            else:
                for baseline_elem, result_elem in zip(baseline_result, result):
                    torch.testing.assert_close(baseline_elem, result_elem)

    def _assert_counting_information(
        self,
        ort_backend: OrtBackend,
        # Number of session runs.
        # If there is no graph break, this should be the same as
        # total number of forward calls.
        expected_execution_count: int,
        # Number of GraphModule's cached.
        # With one graph break, a model will be mapped
        # to two GraphModule's.
        number_of_cached_graph_modules: int,
        # Number of ONNX models cached for each GraphModule,
        # number_of_exported_onnx_models[i] contains # of ONNX models exported from
        # the i-th element (type: torch.fx.GraphModule) in
        # OrtBackend._all_ort_execution_info.execution_info_per_graph_module.values().
        number_of_exported_onnx_models_for_all_graph_modules: Tuple[int, ...],
    ):
        self.assertEqual(expected_execution_count, ort_backend.execution_count)
        self.assertEqual(
            len(ort_backend._all_ort_execution_info.execution_info_per_graph_module),
            number_of_cached_graph_modules,
        )
        self.assertEqual(
            len(ort_backend._all_ort_execution_info.execution_info_per_graph_module),
            len(number_of_exported_onnx_models_for_all_graph_modules),
        )
        for (
            onnx_info,
            expected_number_of_onnx_models,
        ) in zip(
            ort_backend._all_ort_execution_info.execution_info_per_graph_module.values(),
            number_of_exported_onnx_models_for_all_graph_modules,
        ):
            self.assertEqual(len(onnx_info), expected_number_of_onnx_models)

    @parameterized.expand(
        [
            (True,),
            (False,),
        ]
    )
    def test_elementwise_function_single_output(self, test_local_backend: bool):
        example_args_collection = tuple(
            (torch.randn(batch, dtype=torch.float32),) for batch in (2, 4, 6, 8, 10)
        )

        def elementwise_model(x: torch.Tensor):
            y = x.relu()
            z = y.sigmoid()
            return z

        if test_local_backend:
            local_aot_ort, local_ort = make_aot_ort(dynamic=True)
        else:
            # This will use the global ONNXRuntime backend registered
            # in Dynamo to compile the tested model.
            local_aot_ort, local_ort = "onnxrt", None

        self._test_model_numerically(
            elementwise_model,
            local_aot_ort,
            example_args_collection,
        )

        # We can only check local backend's counting information
        # since global backend's counting information comes from
        # all compiled models.
        if test_local_backend:
            assert local_ort is not None
            self._assert_counting_information(
                local_ort,
                # OrtBackend._ort_acclerated_call should have been called 5 times because
                # we have 5 different batch sizes to test.
                expected_execution_count=len(example_args_collection),
                # Since this local_ort only compiled one function,
                # there should be only one GraphModule in its cached.
                number_of_cached_graph_modules=1,
                # Since dynamic shape is enabled, we should only have one ONNX model
                # to support different batch sizes.
                number_of_exported_onnx_models_for_all_graph_modules=(1,),
            )

    @parameterized.expand(
        [
            (True,),
            (False,),
        ]
    )
    def test_elementwise_function_multiple_output(self, test_local_backend: bool):
        example_args_collection = tuple(
            (torch.randn(batch, dtype=torch.float32),) for batch in (2, 4, 8)
        )

        def elementwise_model_with_multiple_outputs(w: torch.Tensor):
            x = w + w
            y = x.relu()
            z = y * y
            return x, y, z

        if test_local_backend:
            local_aot_ort, local_ort = make_aot_ort(dynamic=True)
        else:
            local_aot_ort, local_ort = "onnxrt", None

        self._test_model_numerically(
            elementwise_model_with_multiple_outputs,
            local_aot_ort,
            example_args_collection,
        )

        if test_local_backend:
            assert local_ort is not None
            self._assert_counting_information(
                local_ort,
                expected_execution_count=len(example_args_collection),
                number_of_cached_graph_modules=1,
                number_of_exported_onnx_models_for_all_graph_modules=(1,),
            )

    @parameterized.expand(
        [
            (True,),
            (False,),
        ]
    )
    def test_mlp_with_local_backend(self, test_local_backend: bool):
        example_args_collection = tuple(
            (torch.randn(batch, 2, dtype=torch.float32),) for batch in (1, 2, 4, 6, 8)
        )

        class MLP(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(2, 4, bias=True)
                self.fc2 = nn.Linear(4, 2, bias=True)

            def forward(self, tensor_x: torch.Tensor):
                tensor_x = self.fc1(tensor_x)
                tensor_x = torch.sigmoid(tensor_x)
                tensor_x = self.fc2(tensor_x)
                tensor_x = torch.sigmoid(tensor_x)
                return tensor_x

        if test_local_backend:
            local_aot_ort, local_ort = make_aot_ort(dynamic=True)
        else:
            local_aot_ort, local_ort = "onnxrt", None

        self._test_model_numerically(
            MLP(),
            local_aot_ort,
            example_args_collection,
        )

        if test_local_backend:
            assert local_ort is not None
            self._assert_counting_information(
                local_ort,
                # OrtBackend._ort_acclerated_call should have been called 5 times because
                # we have 5 different batch sizes to test.
                expected_execution_count=len(example_args_collection),
                # Since this local_ort only compiled one function, there should be only two
                # GraphModule's in its cached. One for batch sizes 2, 4, 6, 8 and the other
                # for batch size 1.
                number_of_cached_graph_modules=2,
                # Since dynamic shape is enabled, we should only have one ONNX model
                # to support different batch sizes.
                number_of_exported_onnx_models_for_all_graph_modules=(1, 1),
            )

    @parameterized.expand(
        [
            (True,),
        ]
    )
    def test_llama_attention_with_local_backend(self, test_local_backend: bool):
        from transformers import LlamaConfig  # noqa: F811
        from transformers.models.llama.modeling_llama import (  # noqa: F811
            LlamaAttention,
        )

        hidden_size = 16

        config = LlamaConfig(
            num_hidden_layers=1,
            vocab_size=1024,
            hidden_size=hidden_size,
            intermediate_size=16,
            max_position_embeddings=256,
            num_attention_heads=2,
            hidden_dropout_prob=0.0,
            attention_dropout_prob=0.0,
        )

        class LlamaAttentionWrapper(torch.nn.Module):
            def __init__(self, config):
                super().__init__()
                self.attention = LlamaAttention(config)

            def forward(self, hidden_states, attention_mask, position_ids):
                attn_output, _, _ = self.attention(
                    hidden_states, attention_mask, position_ids
                )
                return attn_output

        def generate_example_inputs(batch: int, seq: int, hidden_size: int):
            # shape: batch x seq x hidden_size
            hidden_state = torch.randn(batch, seq, hidden_size)
            # [0.0000e+00, ..., 0.0000e+00, -3.4028e+38, ...]
            # shape: batch x 1 x seq x seq
            attention_mask = torch.zeros(batch, 1, seq, seq, dtype=torch.float)
            position_ids = torch.arange(0, seq, dtype=torch.int64)
            position_ids = position_ids.unsqueeze(0).view(-1, seq)

            return hidden_state, attention_mask, position_ids

        # Reason for using multiple example argument groups:
        #  Export model to ONNX with one example argument group
        #  and test it with other example argument groups.
        example_args_collection = (
            generate_example_inputs(2, 8, hidden_size),
            generate_example_inputs(4, 7, hidden_size),
            generate_example_inputs(9, 15, hidden_size),
        )

        if test_local_backend:
            local_aot_ort, local_ort = make_aot_ort(dynamic=True)
        else:
            local_aot_ort, local_ort = "onnxrt", None

        model = LlamaAttentionWrapper(config).eval()

        self._test_model_numerically(
            model,
            local_aot_ort,
            example_args_collection,
            fullgraph=True,
        )

        if test_local_backend:
            assert local_ort is not None
            self._assert_counting_information(
                local_ort,
                # The whole attention is captured and executed
                # as a single graph. Thus, the # of test cases
                # is the # of session runs.
                expected_execution_count=len(example_args_collection),
                # This should be one because there is no graph break.
                # If you have 1 graph break, this value should be 2.
                number_of_cached_graph_modules=1,
                # Since dynamic shape is enabled, we should only have
                # one ONNX model to support different batch sizes.
                number_of_exported_onnx_models_for_all_graph_modules=(1,),
            )

    @parameterized.expand(
        [
            (True,),
        ]
    )
    def test_llama_decoder_with_local_backend(self, test_local_backend: bool):
        import inspect

        from transformers import LlamaConfig  # noqa: F811
        from transformers.models.llama.modeling_llama import (  # noqa: F811
            LlamaDecoderLayer,
        )

        hidden_size = 16

        config = LlamaConfig(
            num_hidden_layers=1,
            vocab_size=1024,
            hidden_size=hidden_size,
            intermediate_size=16,
            max_position_embeddings=256,
            num_attention_heads=2,
            hidden_dropout_prob=0.0,
            attention_dropout_prob=0.0,
        )

        class LlamaDecoderWrapper(torch.nn.Module):
            def __init__(self, config):
                super().__init__()
                if len(inspect.signature(LlamaDecoderLayer).parameters) == 3:
                    self.decoder = LlamaDecoderLayer(config, 0)
                else:
                    self.decoder = LlamaDecoderLayer(config)

            def forward(self, hidden_states, attention_mask, position_ids):
                (decoder_output,) = self.decoder(
                    hidden_states, attention_mask, position_ids
                )
                return decoder_output

        def generate_example_inputs(batch: int, seq: int, hidden_size: int):
            # shape: batch x seq x hidden_size
            hidden_state = torch.randn(batch, seq, hidden_size)
            # [0.0000e+00, ..., 0.0000e+00, -3.4028e+38, ...]
            # shape: batch x 1 x seq x seq
            attention_mask = torch.zeros(batch, 1, seq, seq, dtype=torch.float)
            position_ids = torch.arange(0, seq, dtype=torch.int64)
            position_ids = position_ids.unsqueeze(0).view(-1, seq)
            return hidden_state, attention_mask, position_ids

        # Reason for using multiple example argument groups:
        #  Export model to ONNX with one example argument group
        #  and test it with other example argument groups.
        example_args_collection = (
            generate_example_inputs(2, 8, hidden_size),
            generate_example_inputs(4, 7, hidden_size),
            generate_example_inputs(9, 15, hidden_size),
        )

        if test_local_backend:
            local_aot_ort, local_ort = make_aot_ort(dynamic=True)
        else:
            local_aot_ort, local_ort = "onnxrt", None

        model = LlamaDecoderWrapper(config).eval()

        self._test_model_numerically(
            model,
            local_aot_ort,
            example_args_collection,
            fullgraph=True,
        )

        if test_local_backend:
            assert local_ort is not None
            self._assert_counting_information(
                local_ort,
                # The whole attention is captured and executed
                # as a single graph. Thus, the # of test cases
                # is the # of session runs.
                expected_execution_count=len(example_args_collection),
                # This should be one because there is no graph break.
                # If you have 1 graph break, this value should be 2.
                number_of_cached_graph_modules=1,
                # Since dynamic shape is enabled, we should only have
                # one ONNX model to support different batch sizes.
                number_of_exported_onnx_models_for_all_graph_modules=(1,),
            )

    @parameterized.expand(
        [
            (True,),
        ]
    )
    def test_llama_with_local_backend(self, test_local_backend: bool):
        from transformers import LlamaConfig  # noqa: F811
        from transformers.models.llama.modeling_llama import LlamaModel  # noqa: F811

        config = LlamaConfig(
            num_hidden_layers=1,
            vocab_size=1024,
            hidden_size=16,
            intermediate_size=16,
            max_position_embeddings=256,
            num_attention_heads=2,
            hidden_dropout_prob=0.0,
            attention_dropout_prob=0.0,
        )

        config._attn_implementation = "eager"

        class LlamaModelWrapper(torch.nn.Module):
            def __init__(self, config):
                super().__init__()
                self.llama = LlamaModel(config)

            def forward(self, input_ids, attention_mask, position_ids):
                decoder_output = self.llama(
                    input_ids, attention_mask, position_ids, return_dict=False
                )
                return decoder_output

        def generate_example_inputs(batch: int, seq: int):
            # shape: batch x seq x hidden_size
            input_ids = torch.randint(0, 7, size=(batch, seq), dtype=torch.int64)
            # Usually, its shape is a tensor with shape batch x seq x seq.
            # However, to bypass some control flow in the model, we use None.
            attention_mask = None
            position_ids = torch.arange(0, seq, dtype=torch.int64)
            position_ids = position_ids.unsqueeze(0).view(-1, seq)
            return input_ids, attention_mask, position_ids

        # Reason for using multiple example argument groups:
        #  Export model to ONNX with one example argument group
        #  and test it with other example argument groups.
        example_args_collection = (
            generate_example_inputs(2, 8),
            generate_example_inputs(4, 7),
            generate_example_inputs(9, 15),
        )

        if test_local_backend:
            local_aot_ort, local_ort = make_aot_ort(dynamic=True)
        else:
            local_aot_ort, local_ort = "onnxrt", None

        model = LlamaModelWrapper(config).eval()

        self._test_model_numerically(
            model,
            local_aot_ort,
            example_args_collection,
            fullgraph=True,
        )

        if test_local_backend:
            assert local_ort is not None
            self._assert_counting_information(
                local_ort,
                # The whole attention is captured and executed
                # as a single graph. Thus, the # of test cases
                # is the # of session runs.
                expected_execution_count=len(example_args_collection),
                # This should be one because there is no graph break.
                # If you have 1 graph break, this value should be 2.
                number_of_cached_graph_modules=1,
                # Since dynamic shape is enabled, we should only have
                # one ONNX model to support different batch sizes.
                number_of_exported_onnx_models_for_all_graph_modules=(1,),
            )


if __name__ == "__main__":
    common_utils.run_tests()
