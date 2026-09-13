"""Plot the paper2-native primary evidence from frozen findings (read only)."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

def main():
    data = json.loads((ROOT/'results/courtdyn/courtdyn_t28_findings.json').read_text())['seqs']
    metre, pixel, ruler, ratios, factors = [], [], [], [], []
    for seq in ['Q2_top_480-510', 'Q1_top_0-30']:
        d = data[seq]
        for family in ['speed', 'path']:
            r = d['cells'][f'cdnative@ruler@{family}']
            p = d['cells'][f'cdnative@pxunit@{family}']
            assert r['full']['n'] == p['pxunit']['n'] == r['ruler']['n'] == 140
            metre.append(r['full']['tmra'] - d['const'][f'{family}_v3'])
            pixel.append(p['pxunit']['tmra'] - d['const'][f'{family}_px'])
            ruler.append(r['ratio'])
            ratios.append(p['ratio_to_full'])
            factors.append(d['k_px_per_m'])
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'pdf.fonttype': 42, 'axes.spines.top': False,
                         'axes.spines.right': False})
    fig, axs = plt.subplots(1, 2, figsize=(7.1, 3.3), gridspec_kw={'wspace': .35})
    x = np.arange(4)
    blue, orange = '#226b98', '#be5726'
    for ax in axs:
        ax.axvspan(1.5, 3.5, color='#e9e9e9', zorder=0)
        ax.set_xlim(-.55, 3.55)
        ax.set_xticks(x, ['Q2\nspeed', 'Q2\npath', 'Q1\nspeed', 'Q1\npath'])
        ax.tick_params(axis='x', length=0)
    ax = axs[0]
    ax.set_title('(a) Score minus constant', loc='left', fontsize=10.5, pad=10)
    ax.axhline(0, color='#444444', linewidth=.8)
    ax.bar(x-.17, metre, width=.32, label='Metre', color=blue, zorder=2)
    ax.bar(x+.17, pixel, width=.32, label='Pixel', color=orange, zorder=2)
    ax.set_ylim(-53, 49)
    ax.set_ylabel('T-MRA difference (points)')
    for i, (m, p) in enumerate(zip(metre, pixel)):
        ax.text(i-.17, m+1.1, f'+{m:.1f}', ha='center', va='bottom', fontsize=8)
        ax.text(i+.17, p-1.1, f'{p:.1f}', ha='center', va='top', fontsize=8)
    ax.legend(loc='upper left', ncol=2, fontsize=8.5, frameon=False, handlelength=1, columnspacing=1)
    ax = axs[1]
    ax.set_title('(b) Median paired ratios', loc='left', fontsize=10.5, pad=10)
    ax.set_yscale('log')
    ax.set_ylim(.32, 63)
    ax.axhline(1, color='#888888', linewidth=.8)
    ax.plot(x, factors, '--', color='#444444', linewidth=1.2, label='Pixel reference K')
    ax.scatter(x-.08, ruler, marker='s', s=28, color=blue, label='Ruler / full', zorder=3)
    ax.scatter(x+.08, ratios, marker='o', s=28, color=orange, label='Pixel / full', zorder=3)
    for i, p in enumerate(ratios):
        ax.annotate(f'{p:.2f}', (i+.08, p), xytext=(0, -14), textcoords='offset points',
                    ha='center', fontsize=8, color=orange)
    ax.set_yticks([.5, 1, 5, 27], ['0.5', '1', '5', '27'])
    ax.minorticks_off()
    ax.set_ylabel('Prediction ratio (log scale)')
    ax.legend(loc='upper left', bbox_to_anchor=(-.02, .75), fontsize=8,
              frameon=False, labelspacing=.3, handlelength=1.5)
    fig.subplots_adjust(left=.095, right=.99, bottom=.24, top=.87)
    fig.text(.5, .025, 'CourtDyn-native only | 140 items per cell | Shaded Q1 shares a training event',
             ha='center', fontsize=8.5)
    output = ROOT/'paper2/figs/fig_native_core'
    fig.savefig(output.with_suffix('.pdf'))
    fig.savefig(output.with_suffix('.png'), dpi=200)
    plt.close(fig)
    print(output)

if __name__ == '__main__':
    main()
