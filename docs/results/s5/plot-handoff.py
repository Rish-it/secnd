"""Run from repo root: uv run --with matplotlib python docs/results/s5/plot-handoff.py."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

root = Path('docs/results/s5')
report = json.loads((root / 'handoff.json').read_text())
history = report['history']
x = [row['human_labels'] for row in history]
plt.rcParams['svg.hashsalt'] = 'ren-handoff'
fig, (shares, quality) = plt.subplots(2, 1, figsize=(9, 7.5), sharex=True)
shares.stackplot(x, *[[row['shares'][name] for row in history] for name in ('muse', 'ren', 'human')],
                 labels=['Muse teacher', 'Ren automatic', 'Human'],
                 colors=['#d69a39', '#3e73a1', '#a3adad'], alpha=0.95)
shares.set_title('Ren handoff on MTOP · seed 0\nCompleted decisions only; pending questions excluded')
shares.set_ylabel('Cumulative share')
shares.set_ylim(0, 1)
shares.legend(loc='upper right', framealpha=0.95)
quality.plot(x, [row['coverage'] for row in history], label='Test coverage', color='#3e73a1', lw=2)
quality.plot(x, [row['precision'] for row in history], label='Test automatic precision', color='#79528d', lw=2)
quality.axhline(0.99, color='#555555', ls='--', lw=1, label='99% precision target')
quality.set_ylabel('Fixed test result')
quality.set_xlabel('Simulated human answers (including spot-checks)')
quality.set_ylim(0, 1.03)
quality.legend(loc='lower right')
for axis in (shares, quality):
    axis.yaxis.set_major_formatter(PercentFormatter(1))
    axis.grid(axis='y', alpha=0.2)
    axis.spines[['top', 'right']].set_visible(False)
last = history[-1]
fig.text(0.5, 0.015,
         f"{report['teacher_rows']} teacher labels; fixed calibration/test. "
         f"{last['pending']:,} pending questions excluded from shares.\n"
         f"True stream labels simulate people. Teacher spend: ${float(report['task_spend_usd']):.4f}.",
         ha='center', fontsize=9)
fig.tight_layout(rect=(0, 0.06, 1, 1))
fig.savefig(root / 'handoff.svg', metadata={'Date': None, 'Creator': None})
fig.savefig(root / 'handoff.png', dpi=160, metadata={'Software': None})
