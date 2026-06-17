"""PyTorch -> ONNX -> TF -> int8 TFLite -> C header (edge-phase script).

Reads architecture/window settings from ``config.py`` so the graph matches the
trained checkpoint. The heavy edge stack (onnx/onnx_tf/tensorflow) is imported
lazily, so this module imports fine with only the training deps (CLI unit tests).

Usage::

    python src/edge_conversion/quantize_model.py --subject 3
    python src/edge_conversion/quantize_model.py --subject 3 --window 128 \\
        --checkpoint models/win_128/hopegait_tcn_best_subj3.pth
"""

import os
import sys
import glob
import argparse


SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

import numpy as np
import torch

from config import (MODELS_DIR, PROCESSED_DATA_DIR, NUM_INPUTS, NUM_CHANNELS,
                    KERNEL_SIZE, DROPOUT, DROP_PATH, USE_SE, NUM_CLASSES,
                    WINDOW_SIZES)
from models.tcn_model import HopeGaitTCN
from data_pipeline.dsp import RobustScaler


_EDGE_DEPS_HINT = (
    "Edge conversion requires the optional edge stack. Install it with:\n"
    "    pip install -r requirements-edge.txt\n"
    "(do this in a separate venv from the training stack — protobuf / TF "
    "version pins conflict with PyTorch's wheels.)"
)


def _load_edge_deps():
    """Import onnx / onnx_tf / tensorflow lazily, with a clear error message."""
    try:
        import onnx  # noqa: F401
        from onnx_tf.backend import prepare  # noqa: F401
        import tensorflow as tf  # noqa: F401
    except ImportError as e:
        sys.stderr.write(f"ERROR: missing edge dependency ({e}).\n{_EDGE_DEPS_HINT}\n")
        sys.exit(2)
    return onnx, prepare, tf


def representative_data_gen_factory(seq_length, num_inputs, processed_dir,
                                    scaler=None):
    """Return a generator that yields ~100 calibration samples.

    Prefers real preprocessed windows from ``processed_dir/win_<seq_length>/*_x.npy``
    (so int8 scales reflect actual signal statistics). Falls back to standard
    normal noise only when no real data is available.

    IMPORTANT: when a ``scaler`` is given the windows are scaled exactly as in
    training/inference. The model is trained on RobustScaler-normalized inputs,
    so calibrating on raw windows would set the int8 input range from the wrong
    distribution and inflate the quantization error. Pass the fold's scaler.
    """
    win_dir = os.path.join(processed_dir, f'win_{seq_length}')
    real_files = sorted(glob.glob(os.path.join(win_dir, '*_x.npy')))

    def _prep(window):
        w = window.astype(np.float32)
        return scaler.transform(w) if scaler is not None else w

    if real_files:
        def gen():
            yielded = 0
            for path in real_files:
                arr = np.load(path)
                if arr.ndim != 3 or arr.shape[1] != seq_length or arr.shape[2] != num_inputs:
                    continue
                for i in range(arr.shape[0]):
                    if yielded >= 100:
                        return
                    yield [_prep(arr[i:i + 1])]
                    yielded += 1
        return gen

    sys.stderr.write(
        f"WARN: no calibration data at {win_dir}; falling back to random noise. "
        "int8 scales will not reflect real signal statistics.\n"
    )

    def gen():
        rng = np.random.default_rng(0)
        for _ in range(100):
            yield [rng.standard_normal((1, seq_length, num_inputs)).astype(np.float32)]
    return gen


def convert_to_c_array(tflite_path, header_path):
    """Write the ``.tflite`` bytes as a C header array for the MCU runtime."""
    with open(tflite_path, 'rb') as f:
        tflite_content = f.read()
    hex_lines = []
    for i in range(0, len(tflite_content), 12):
        chunk = tflite_content[i:i + 12]
        hex_lines.append('    ' + ', '.join(f'0x{b:02x}' for b in chunk))
    hex_array = ',\n'.join(hex_lines)

    c_code = f"""#ifndef HOPEGAIT_MODEL_DATA_H
#define HOPEGAIT_MODEL_DATA_H
extern const unsigned char hopegait_model_tflite[];
extern const unsigned int hopegait_model_tflite_len;
const unsigned char hopegait_model_tflite[] = {{
{hex_array}
}};
const unsigned int hopegait_model_tflite_len = {len(tflite_content)};
#endif
"""
    os.makedirs(os.path.dirname(header_path) or '.', exist_ok=True)
    with open(header_path, 'w') as f:
        f.write(c_code)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument(
        '--subject', required=True,
        help='Subject ID whose checkpoint to convert (e.g. "3" or "S03").',
    )
    parser.add_argument(
        '--window', type=int, default=WINDOW_SIZES[0],
        help=f'Window length in samples (default: {WINDOW_SIZES[0]}).',
    )
    parser.add_argument(
        '--checkpoint', default=None,
        help='Path to the .pth checkpoint. Defaults to '
             '<MODELS_DIR>/win_<window>/hopegait_tcn_best_subj<subject>.pth.',
    )
    parser.add_argument(
        '--output-prefix', default=None,
        help='Output prefix; produces <prefix>.onnx, <prefix>_int8.tflite, '
             '<prefix>_model_data.h. Defaults to <MODELS_DIR>/hopegait.',
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    # Training writes checkpoints under MODELS_DIR/win_<window>/ (see train.py),
    # so default there — not the MODELS_DIR root.
    checkpoint = args.checkpoint or os.path.join(
        MODELS_DIR, f'win_{args.window}', f'hopegait_tcn_best_subj{args.subject}.pth'
    )
    if not os.path.exists(checkpoint):
        sys.stderr.write(
            f"ERROR: checkpoint not found at {checkpoint}. Train first or pass "
            f"--checkpoint explicitly.\n"
        )
        sys.exit(1)

    output_prefix = args.output_prefix or os.path.join(MODELS_DIR, 'hopegait')
    onnx_path = f'{output_prefix}.onnx'
    tf_dir = f'{output_prefix}_tf'
    tflite_path = f'{output_prefix}_int8.tflite'
    header_path = f'{output_prefix}_model_data.h'

    onnx_mod, prepare, tf = _load_edge_deps()

    device = torch.device('cpu')
    model = HopeGaitTCN(
        num_inputs=NUM_INPUTS,
        num_channels=tuple(NUM_CHANNELS),
        kernel_size=KERNEL_SIZE,
        num_classes=NUM_CLASSES,
        dropout=DROPOUT,
        drop_path=DROP_PATH,
        use_se=USE_SE,
    )
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    model.eval()

    os.makedirs(os.path.dirname(onnx_path) or '.', exist_ok=True)
    dummy_input = torch.randn(1, args.window, NUM_INPUTS)
    torch.onnx.export(
        model, dummy_input, onnx_path,
        export_params=True, opset_version=13,
        do_constant_folding=True,
        input_names=['input'], output_names=['output'],
    )
    print(f"Wrote ONNX: {onnx_path}")

    onnx_model = onnx_mod.load(onnx_path)
    prepare(onnx_model).export_graph(tf_dir)
    print(f"Wrote TF SavedModel: {tf_dir}")

    # Calibrate on windows scaled with the SAME fold scaler the model trained
    # on. Without it, int8 input ranges come from the raw distribution and the
    # quantization error is artificially large.
    scaler_path = os.path.join(MODELS_DIR, f'win_{args.window}',
                               f'scaler_subj{args.subject}.npz')
    calib_scaler = None
    if os.path.exists(scaler_path):
        calib_scaler = RobustScaler.load(scaler_path)
    else:
        sys.stderr.write(
            f"WARN: no scaler at {scaler_path}; calibrating on RAW windows. "
            "int8 input range will not match the scaled inputs the model "
            "expects — train the fold first so the scaler exists.\n"
        )

    converter = tf.lite.TFLiteConverter.from_saved_model(tf_dir)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_data_gen_factory(
        args.window, NUM_INPUTS, PROCESSED_DATA_DIR, scaler=calib_scaler,
    )
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    tflite_model = converter.convert()

    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)
    print(f"Wrote int8 TFLite: {tflite_path} ({len(tflite_model)} bytes)")

    convert_to_c_array(tflite_path, header_path)
    print(f"Wrote C header: {header_path}")


if __name__ == "__main__":
    main()
