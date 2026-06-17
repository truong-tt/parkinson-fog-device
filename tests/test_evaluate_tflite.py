"""C3: int8 eval quant helpers — softmax, affine quant/dequant roundtrip, and
the inference loop wired against a fake TFLite interpreter (no TF dependency)."""

import numpy as np

from edge_conversion.evaluate_tflite import (_softmax2, quantize_input,
                                             dequantize_output, run_tflite)


def test_softmax2_basic():
    assert abs(_softmax2(np.array([[0.0, 0.0]]))[0] - 0.5) < 1e-9
    assert _softmax2(np.array([[0.0, 20.0]]))[0] > 0.999
    assert _softmax2(np.array([[20.0, 0.0]]))[0] < 1e-3


def test_quant_dequant_roundtrip():
    scale, zp = 0.05, -3
    x = np.linspace(-1, 1, 50).astype(np.float32)
    q = quantize_input(x, scale, zp)
    assert q.dtype == np.int8
    back = dequantize_output(q, scale, zp)
    np.testing.assert_allclose(back, x, atol=scale)  # within one quant step


def test_quant_passthrough_when_scale_zero():
    x = np.array([0.1, -0.2], np.float32)
    np.testing.assert_array_equal(quantize_input(x, 0, 0), x)
    np.testing.assert_array_equal(dequantize_output(x, 0, 0), x)


class _FakeInterp:
    """Minimal stand-in for tf.lite.Interpreter returning fixed int8 logits."""
    def get_input_details(self):
        return [{'index': 0, 'quantization': (0.5, -128)}]

    def get_output_details(self):
        return [{'index': 1, 'quantization': (0.25, 0)}]

    def set_tensor(self, index, value):
        self._last = value

    def invoke(self):
        pass

    def get_tensor(self, index):
        # int8 logits -> dequant (v-0)*0.25 -> [0, 2] -> P(1) ~ 0.88
        return np.array([[0, 8]], dtype=np.int8)


def test_run_tflite_loop():
    X = np.zeros((3, 8, 9), dtype=np.float32)
    probs = run_tflite(_FakeInterp(), X)
    assert probs.shape == (3,)
    assert np.all(probs > 0.8) and np.all(probs < 0.95)
