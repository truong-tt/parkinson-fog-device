"""Measure the int8-vs-fp32 accuracy delta (edge-phase deliverable).

Runs the fp32 PyTorch checkpoint and the int8 TFLite model over the same fold's
test windows (scaled with the fold's RobustScaler) and reports the MCC/sens/spec
delta, on-disk size, and parameter count. Both get identical input, so any gap
is attributable to quantization alone.

Run in the edge venv::

    pip install -r requirements-edge.txt
    python src/edge_conversion/evaluate_tflite.py --subject 3
"""

import os
import sys
import json
import argparse
import numpy as np

SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from config import (MODELS_DIR, PROCESSED_DATA_DIR, NUM_INPUTS, NUM_CHANNELS,
                    KERNEL_SIZE, DROPOUT, DROP_PATH, USE_SE, NUM_CLASSES,
                    WINDOW_SIZES, SEED, BATCH_SIZE)


def _softmax2(logits):
    """P(class=1) from a (N, 2) logit array, numerically stable."""
    logits = np.asarray(logits, dtype=np.float64)
    m = logits.max(axis=1, keepdims=True)
    e = np.exp(logits - m)
    return (e[:, 1] / e.sum(axis=1))


def quantize_input(x, scale, zero_point):
    """Float -> int8 using the TFLite input tensor's affine params."""
    if scale == 0:  # model takes float input
        return x.astype(np.float32)
    q = np.round(x / scale + zero_point)
    return np.clip(q, -128, 127).astype(np.int8)


def dequantize_output(y, scale, zero_point):
    """int8 -> float using the TFLite output tensor's affine params."""
    if scale == 0:
        return y.astype(np.float32)
    return (y.astype(np.float32) - zero_point) * scale


def run_tflite(interpreter, X):
    """Probabilities P(FoG) for each (T, C) window in X via the int8 model."""
    inp = interpreter.get_input_details()[0]
    out = interpreter.get_output_details()[0]
    in_scale, in_zp = inp['quantization']
    out_scale, out_zp = out['quantization']
    probs = []
    for i in range(len(X)):
        window = X[i:i + 1].astype(np.float32)  # (1, T, C)
        interpreter.set_tensor(inp['index'], quantize_input(window, in_scale, in_zp))
        interpreter.invoke()
        logits = dequantize_output(interpreter.get_tensor(out['index']), out_scale, out_zp)
        probs.append(_softmax2(logits.reshape(1, -1))[0])
    return np.array(probs)


def _collect_scaled_windows(data_dir, test_subject, scaler):
    """Scaled test windows + last-step targets for the fold (same as eval)."""
    from data_pipeline.dataset import create_loso_dataloaders
    _, _, test_loader, _, _ = create_loso_dataloaders(
        data_dir, test_subject=test_subject, batch_size=BATCH_SIZE,
        scaler=scaler, augment_train=False, seed=SEED)
    xs, ys = [], []
    for x, y_dense in test_loader:
        xs.append(x.numpy())
        ys.append(y_dense[:, -1].numpy())
    return np.concatenate(xs), np.concatenate(ys)


def _fp32_probs(model_path, X):
    import torch
    from models.tcn_model import HopeGaitTCN
    model = HopeGaitTCN(num_inputs=NUM_INPUTS, num_channels=tuple(NUM_CHANNELS),
                        kernel_size=KERNEL_SIZE, num_classes=NUM_CLASSES,
                        dropout=DROPOUT, drop_path=DROP_PATH, use_se=USE_SE)
    model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=True))
    model.eval()
    n_params = int(sum(p.numel() for p in model.parameters()))
    with torch.no_grad():
        logits = model(torch.from_numpy(X.astype(np.float32))).numpy()
    return _softmax2(logits), n_params


def evaluate(subject, seq_length):
    """Compute fp32-vs-int8 metrics for one fold.

    Args:
        subject: Test-subject id whose checkpoint/scaler/tflite to load.
        seq_length: Window length; selects the ``win_<seq_length>`` directories.

    Returns:
        Dict with per-model metrics, deltas, model size, params, and prob MAE.
    """
    from data_pipeline.dsp import RobustScaler
    from training.evaluate import _metrics_from_preds, _load_fold_threshold

    target_dir = os.path.join(MODELS_DIR, f'win_{seq_length}')
    model_path = os.path.join(target_dir, f'hopegait_tcn_best_subj{subject}.pth')
    scaler_path = os.path.join(target_dir, f'scaler_subj{subject}.npz')
    meta_path = os.path.join(target_dir, f'fold_meta_subj{subject}.json')
    tflite_path = os.path.join(MODELS_DIR, 'hopegait_int8.tflite')
    for pth in (model_path, scaler_path, tflite_path):
        if not os.path.exists(pth):
            sys.stderr.write(f"ERROR: missing {pth}. Train + quantize first.\n")
            sys.exit(1)

    try:
        import tensorflow as tf
    except ImportError:
        sys.stderr.write("ERROR: tensorflow missing. pip install -r requirements-edge.txt\n")
        sys.exit(2)

    data_dir = os.path.join(PROCESSED_DATA_DIR, f'win_{seq_length}')
    scaler = RobustScaler.load(scaler_path)
    X, y = _collect_scaled_windows(data_dir, subject, scaler)
    threshold = _load_fold_threshold(meta_path)

    p_fp32, n_params = _fp32_probs(model_path, X)
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    p_int8 = run_tflite(interpreter, X)

    m_fp32 = _metrics_from_preds(y, (p_fp32 >= threshold).astype(np.int64), threshold)
    m_int8 = _metrics_from_preds(y, (p_int8 >= threshold).astype(np.int64), threshold)

    result = {
        'subject': subject, 'window': seq_length, 'threshold': threshold,
        'n_test_windows': int(len(y)), 'n_params': n_params,
        'tflite_bytes': int(os.path.getsize(tflite_path)),
        'fp32': m_fp32, 'int8': m_int8,
        'delta_mcc': float(m_int8['mcc'] - m_fp32['mcc']),
        'delta_sensitivity': float(m_int8['sensitivity'] - m_fp32['sensitivity']),
        'delta_specificity': float(m_int8['specificity'] - m_fp32['specificity']),
        'prob_mae': float(np.mean(np.abs(p_fp32 - p_int8))),
    }
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description="int8 vs fp32 accuracy delta.")
    p.add_argument('--subject', required=True)
    p.add_argument('--window', type=int, default=WINDOW_SIZES[0])
    args = p.parse_args(argv)

    r = evaluate(args.subject, args.window)
    print(f"\n=== int8 vs fp32 (subj {r['subject']}, win {r['window']}, "
          f"thr {r['threshold']:.3f}) ===")
    print(f"  params={r['n_params']:,}  tflite={r['tflite_bytes']/1024:.1f} KiB  "
          f"windows={r['n_test_windows']}")
    print(f"  MCC   fp32={r['fp32']['mcc']:+.3f}  int8={r['int8']['mcc']:+.3f}  "
          f"delta={r['delta_mcc']:+.3f}")
    print(f"  Sens  fp32={r['fp32']['sensitivity']*100:.1f}%  "
          f"int8={r['int8']['sensitivity']*100:.1f}%")
    print(f"  Spec  fp32={r['fp32']['specificity']*100:.1f}%  "
          f"int8={r['int8']['specificity']*100:.1f}%")
    print(f"  prob MAE={r['prob_mae']:.4f}")

    out = os.path.join(MODELS_DIR, 'int8_delta.json')
    with open(out, 'w') as f:
        json.dump(r, f, indent=2)
    print(f"Wrote -> {out}")


if __name__ == "__main__":
    main()
