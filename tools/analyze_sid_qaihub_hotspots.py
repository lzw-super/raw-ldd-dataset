"""Aggregate existing three-round NPU cycle profiles; no cloud calls or training."""
import csv
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path('ref-doc/qaihub_w8a8_20261007')


def haar_group(name):
    if name.startswith('/wavelet/ll_restorer/'):
        return 'LL restoration CNN (including fusion)'
    match = re.fullmatch(r'/wavelet/(Sub|Relu|Neg)(?:_(\d+))?', name)
    if match:
        op, index = match[1], int(match[2] or 0)
        return f'HF soft L{index // {"Sub": 9, "Relu": 6, "Neg": 3}[op] + 1}'
    if name.startswith('/wavelet/Split'):
        return 'DWT band split/slice'
    if re.fullmatch(r'/wavelet/Conv(?:_\d+)?', name):
        return 'DWT fixed convolution'
    if name.startswith('/wavelet/ConvTranspose'):
        return 'IDWT fixed transposed convolution'
    if name.startswith('/wavelet/Concat'):
        return 'IDWT band concat'
    if re.fullmatch(r'/refiner/s2d(?:_\d+)?/Conv', name):
        return 'Refiner two S2D convolutions'
    if name.startswith('/refiner/d2s/'):
        return 'Refiner D2S convolution'
    if name == '/refiner/Concat':
        return 'Refiner concat'
    if any(name.startswith('/refiner/' + prefix) for prefix in ('stem/', 'blocks/', 'head/', 'fusion/')):
        return 'Refiner CNN (including activations/fusion)'
    if name.startswith(('noisy_raw', 'output_0')):
        return 'Boundary layout entries'
    if name in ('Input', 'Output'):
        return 'Input/Output entries'
    if 'Constant' in name:
        return 'Constants'
    raise ValueError(f'Unclassified Haar profile entry: {name}')


def main():
    manifest = json.loads((ROOT / 'jobs.json').read_text())
    output = {}
    for label in ('haar_ll_no_norm', 'mrlfn', 'splitternet'):
        rounds = [json.loads((ROOT / label / f'profile_{i}.json').read_text()) for i in (1, 2, 3)]
        by_round = []
        for profile in rounds:
            entries = profile['execution_detail']
            assert all(e['execution_time'] == 0 for e in entries)
            assert all(e['compute_unit'] == 'NPU' for e in entries)
            assert len({e['name'] for e in entries}) == len(entries)
            by_round.append({e['name']: e for e in entries})
        assert all(set(r) == set(by_round[0]) for r in by_round)
        totals = [sum(e['execution_cycles'] for e in r.values()) for r in by_round]
        total_mean = statistics.mean(totals)
        nodes, groups = [], defaultdict(lambda: [0, 0, 0])
        for name in by_round[0]:
            cycles = [r[name]['execution_cycles'] for r in by_round]
            mean = statistics.mean(cycles)
            group = haar_group(name) if label == 'haar_ll_no_norm' else ('Pad-named entries' if name.endswith('/Pad') else 'Other entries')
            for i, c in enumerate(cycles):
                groups[group][i] += c
            nodes.append(dict(name=name, group=group, cycles=cycles, mean_cycles=mean,
                              share_percent=mean / total_mean * 100,
                              round_share_percent=[c/t*100 for c,t in zip(cycles,totals)]))
        grouped = [dict(group=g, cycles=c, mean_cycles=statistics.mean(c),
                        share_percent=statistics.mean(c)/total_mean*100,
                        round_share_percent=[v/t*100 for v,t in zip(c,totals)]) for g,c in groups.items()]
        assert abs(sum(g['mean_cycles'] for g in grouped) - total_mean) < 1e-6
        output[label] = dict(profile_jobs=[p['id'] for p in manifest['models'][label]['profile_jobs']],
                             entry_count=len(nodes), total_cycles_by_round=totals, mean_total_cycles=total_mean,
                             groups=sorted(grouped,key=lambda x:x['mean_cycles'],reverse=True),
                             nodes=sorted(nodes,key=lambda x:x['mean_cycles'],reverse=True))
    (ROOT / 'hotspots.json').write_text(json.dumps(output,indent=2)+'\n')
    with (ROOT/'haar_hotspots.csv').open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(['name','group','round1_cycles','round2_cycles','round3_cycles','mean_cycles','cycle_share_percent'])
        for n in output['haar_ll_no_norm']['nodes']:
            writer.writerow([n['name'],n['group'],*n['cycles'],n['mean_cycles'],n['share_percent']])
    for label, value in output.items():
        print(label, value['entry_count'], 'entries')
        for group in value['groups']:
            print(f"  {group['group']}: {group['mean_cycles']:.0f} cycles ({group['share_percent']:.2f}%)")


if __name__ == '__main__':
    main()
