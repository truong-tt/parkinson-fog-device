"""Dataset EDA for the report's Problem-Analysis section (stage 1).

Characterizes the preprocessed Stanford NMBL windows: class imbalance,
per-subject FoG prevalence, and FoG episode-length distribution. Writes
figures to reports/figures/ and a markdown summary to reports/eda_summary.md.

Episode stats are computed per recording (using recording boundaries) so an
episode is never miscounted across a recording seam — same convention as the
evaluator's event metrics.

Run (after preprocessing):
    python scripts/eda.py            # all window sizes in config
    python scripts/eda.py --window 128
"""

import os
import sys
import json
import argparse
import numpy as np

SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from config import (WINDOW_SIZES, PROCESSED_DATA_DIR, SAMPLING_RATE,
                    WINDOW_OVERLAP)
from data_pipeline.dataset import (_group_files_by_subject, _last_step,
                                   recording_lengths)
from training.evaluate import _find_runs, _split_by_lengths

REPORTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'reports'))
FIG_DIR = os.path.join(REPORTS_DIR, 'figures')


def _step_seconds(seq_length):
    step_samples = max(1, int(seq_length * (1.0 - WINDOW_OVERLAP)))
    return step_samples / float(SAMPLING_RATE)


def collect_stats(data_dir, seq_length):
    groups = _group_files_by_subject(data_dir)
    subjects = sorted(groups)
    if not subjects:
        return None

    step_s = _step_seconds(seq_length)
    per_subject = {}
    episode_lengths_s = []  # pooled across all subjects/recordings
    total_pos = total_n = 0

    for subj in subjects:
        labels = []
        for xf in groups[subj]:
            yf = xf.replace('_x.npy', '_y.npy')
            if not os.path.exists(yf):
                continue
            labels.append(_last_step(np.load(yf)))
        if not labels:
            continue
        y = np.concatenate(labels).astype(np.int64)
        rec_lengths = recording_lengths(data_dir, subj)

        # Episodes per recording (no seam merge).
        for seg in _split_by_lengths(y, rec_lengths):
            for start, end in _find_runs(seg, 1):
                episode_lengths_s.append((end - start + 1) * step_s)

        n_pos = int(y.sum())
        per_subject[subj] = {
            'n_windows': int(len(y)),
            'pos_rate': float(np.mean(y)),
            'n_fog_windows': n_pos,
        }
        total_pos += n_pos
        total_n += len(y)

    return {
        'window': seq_length,
        'n_subjects': len(per_subject),
        'overall_pos_rate': (total_pos / total_n) if total_n else 0.0,
        'n_windows': total_n,
        'n_episodes': len(episode_lengths_s),
        'episode_lengths_s': episode_lengths_s,
        'per_subject': per_subject,
    }


def make_figures(stats):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    os.makedirs(FIG_DIR, exist_ok=True)
    w = stats['window']

    # 1) Class imbalance.
    pos = stats['overall_pos_rate']
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.bar(['non-FoG', 'FoG'], [1 - pos, pos], color=['#4c72b0', '#c44e52'])
    ax.set_ylabel('fraction of windows')
    ax.set_title(f'Class balance (win {w}) — FoG {pos*100:.1f}%')
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, f'class_balance_win{w}.png'), dpi=120)
    plt.close(fig)

    # 2) Per-subject FoG prevalence.
    subs = sorted(stats['per_subject'])
    rates = [stats['per_subject'][s]['pos_rate'] * 100 for s in subs]
    fig, ax = plt.subplots(figsize=(max(5, len(subs) * 0.5), 3))
    ax.bar(subs, rates, color='#c44e52')
    ax.axhline(stats['overall_pos_rate'] * 100, ls='--', c='k', lw=1,
               label='overall')
    ax.set_ylabel('FoG window %')
    ax.set_title(f'Per-subject FoG prevalence (win {w})')
    ax.legend()
    plt.setp(ax.get_xticklabels(), rotation=60, ha='right', fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, f'per_subject_prevalence_win{w}.png'), dpi=120)
    plt.close(fig)

    # 3) Episode-length distribution.
    lengths = stats['episode_lengths_s']
    if lengths:
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.hist(lengths, bins=30, color='#55a868')
        ax.set_xlabel('FoG episode length (s)')
        ax.set_ylabel('count')
        ax.set_title(f'Episode lengths (win {w}) — n={len(lengths)}')
        fig.tight_layout()
        fig.savefig(os.path.join(FIG_DIR, f'episode_lengths_win{w}.png'), dpi=120)
        plt.close(fig)


def write_summary(all_stats, path):
    lines = ["# HopeGait — Dataset EDA (Stanford NMBL)", ""]
    for stats in all_stats:
        w = stats['window']
        lengths = np.array(stats['episode_lengths_s']) if stats['episode_lengths_s'] else np.array([0.0])
        lines += [
            f"## Window {w}",
            "",
            f"- Subjects: {stats['n_subjects']}",
            f"- Windows: {stats['n_windows']:,}",
            f"- Overall FoG rate: {stats['overall_pos_rate']*100:.2f}%",
            f"- FoG episodes: {stats['n_episodes']}",
            f"- Episode length (s): median {np.median(lengths):.1f}, "
            f"mean {np.mean(lengths):.1f}, max {np.max(lengths):.1f}",
            "",
            "| Subject | Windows | FoG % |",
            "|---|---|---|",
        ]
        for s in sorted(stats['per_subject']):
            v = stats['per_subject'][s]
            lines.append(f"| {s} | {v['n_windows']} | {v['pos_rate']*100:.1f}% |")
        lines.append("")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))


def main():
    p = argparse.ArgumentParser(description="HopeGait dataset EDA.")
    p.add_argument('--window', type=int, default=None)
    args = p.parse_args()
    windows = [args.window] if args.window is not None else WINDOW_SIZES

    all_stats = []
    for seq in windows:
        data_dir = os.path.join(PROCESSED_DATA_DIR, f'win_{seq}')
        stats = collect_stats(data_dir, seq)
        if stats is None:
            print(f"No processed data at {data_dir}; skipping.")
            continue
        make_figures(stats)
        all_stats.append(stats)
        print(f"win {seq}: {stats['n_subjects']} subjects, "
              f"FoG {stats['overall_pos_rate']*100:.2f}%, "
              f"{stats['n_episodes']} episodes")

    if not all_stats:
        print("Nothing to summarize. Run preprocessing first.")
        return
    summary_path = os.path.join(REPORTS_DIR, 'eda_summary.md')
    write_summary(all_stats, summary_path)
    # Drop the raw episode-length arrays from the JSON to keep it small.
    compact = [{k: v for k, v in s.items() if k != 'episode_lengths_s'}
               for s in all_stats]
    with open(os.path.join(REPORTS_DIR, 'eda_summary.json'), 'w') as f:
        json.dump(compact, f, indent=2)
    print(f"Figures -> {FIG_DIR}\nSummary -> {summary_path}")


if __name__ == "__main__":
    main()
