"""保存指幅から同じ容器モデルだけを再生し、残圧を採点専用に調べる。"""
import argparse
import hashlib
import json
from pathlib import Path

from resumable_job import atomic_json
from eyedrop.bottle import Bottle
from panda_trace_quality import expected_physics_samples

PHYSICS_HZ = 240
BOTTLE_DEPTH_M = .0159
OPEN_THRESHOLD = .9
STRICT_OPEN_THRESHOLD=.95
PRESSURE_EPSILON_KPA = 1e-9


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('trial')
    parser.add_argument('--input-contract',choices=('command','measured'),default='command')
    parser.add_argument('--open-threshold',type=float,choices=(OPEN_THRESHOLD,STRICT_OPEN_THRESHOLD),default=OPEN_THRESHOLD)
    args=parser.parse_args()
    root=Path(args.trial)
    protocol=json.loads((root/'protocol.json').read_text())
    expected=expected_physics_samples(protocol)
    if 'bottle_input' in protocol and protocol['bottle_input']!=args.input_contract:
        raise ValueError('監査の容器入力方式が実験と異なります')
    if args.input_contract=='measured' and protocol.get('bottle_input')!='measured':
        raise ValueError('実測変位方式の明示された実験だけを監査できます')
    result=json.loads((root/'result.json').read_text())
    if not result['completed']:
        raise ValueError('終了済みの記録が必要です')
    rows=[]
    for item in result['rows']:
        directory=root/f'{item["condition"]}-{item["case"]}'
        samples=json.loads((directory/'pressure-samples.json').read_text())
        if len(samples)!=expected:
            raise ValueError('全物理時点が揃っていません')
        bottle=Bottle()
        closed_positive=closed_flowing=0
        maximum=maximum_displacement=closed_flow_ml=0.
        for sample in samples[1:]:
            displacement=max(0.,min((BOTTLE_DEPTH_M-sample['gap_m'])*1000,bottle.config.max_displacement_mm))
            volume_before=bottle.reservoir_ml
            (bottle.step_measured if args.input_contract=='measured' else bottle.step)(displacement,1/PHYSICS_HZ)
            if sample['open']<args.open_threshold:
                maximum=max(maximum,bottle.pressure_kpa)
                maximum_displacement=max(maximum_displacement,bottle.displacement_mm)
                closed_positive+=int(bottle.pressure_kpa>PRESSURE_EPSILON_KPA)
                closed_flowing+=int(bottle.pressure_kpa>bottle.config.opening_pressure_kpa)
                closed_flow_ml+=volume_before-bottle.reservoir_ml
        verified=bottle.drop_count==item['drops']
        rows.append({'case':item['case'],'condition':item['condition'],'drop_count_matches':verified,
                     'replayed_drop_count':bottle.drop_count,'closed_pressure_max_kpa':maximum,
                     'closed_bottle_displacement_max_mm':maximum_displacement,
                     'closed_positive_pressure_seconds':closed_positive/PHYSICS_HZ,
                     'closed_above_opening_pressure_seconds':closed_flowing/PHYSICS_HZ,
                     'closed_flow_ml':closed_flow_ml,
                     'pressure_record_sha256':hashlib.sha256((directory/'pressure-samples.json').read_bytes()).hexdigest()})
    report={'completed':True,'training_updates':0,'input_contract':args.input_contract,'open_threshold':args.open_threshold,'rows':rows,
            'replay_drop_counts_match':all(r['drop_count_matches'] for r in rows),
            'scope':'Audit-only deterministic Bottle replay of saved240Hz measured gaps; internal states never provided to controller. Pressure is a simulation-model value, not a real-bottle measurement.'}
    name='bottle-pressure-audit.json' if args.open_threshold==OPEN_THRESHOLD else 'bottle-pressure-audit-095.json'
    atomic_json(root/name,report)
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
