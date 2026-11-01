"""保存した全物理時点から閉眼中の残圧を独立に集計する。"""
import argparse
import json
from pathlib import Path

from resumable_job import atomic_json
from panda_trace_quality import expected_physics_samples

OPEN_MIN = .9
STRICT_OPEN_MIN = .95
COMPRESSED_GAP_M = .0158
PHYSICS_HZ = 240
EXPECTED_SAMPLES = 15 * PHYSICS_HZ + 1


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('trial')
    args=parser.parse_args()
    root=Path(args.trial)
    protocol=json.loads((root/'protocol.json').read_bytes())
    expected=expected_physics_samples(protocol)
    open_min=protocol.get('open_threshold',OPEN_MIN)
    if open_min not in (OPEN_MIN,STRICT_OPEN_MIN):raise ValueError('押圧監査の開眼基準が未登録です')
    result=json.loads((root/'result.json').read_text())
    if not result['completed']:
        raise ValueError('試験が完了していません')
    rows=[]
    for item in result['rows']:
        directory=root/f'{item["condition"]}-{item["case"]}'
        samples=json.loads((directory/'pressure-samples.json').read_text())
        trace=json.loads((directory/'trace.json').read_text())
        ticks=[round(s['time']*PHYSICS_HZ) for s in samples]
        complete=len(samples)==expected and ticks==list(range(expected))
        closed=[s for s in samples if s['open']<open_min]
        actual=sum(s['gap_m']<COMPRESSED_GAP_M for s in closed)
        commanded=sum(r['pressure_interval']['closed_command_samples'] for r in trace if r['pressure_interval'] is not None)
        false_open=sum(r.get('prediction') is not None and r['prediction']['openness']>=STRICT_OPEN_MIN and r['audit_open']<open_min for r in trace)
        agreement=(actual==item['closed_measured_samples'] and commanded==item['closed_command_samples'])
        rows.append({'case':item['case'],'condition':item['condition'],'complete':complete,
                     'closed_physics_samples':len(closed),'closed_measured_samples':actual,
                     'closed_command_samples':commanded,'false_open_frames':false_open,
                     'closed_residual_seconds':actual/PHYSICS_HZ,'matches_evaluator':agreement})
    verified=bool(rows) and all(r['complete'] and r['matches_evaluator'] for r in rows)
    record={'completed':True,'rows':rows,'training_updates':0,'open_threshold':open_min,
            'records_verified':verified,
            'closed_pressure_zero':verified and all(not r['closed_measured_samples'] and not r['closed_command_samples'] for r in rows),
            'scope':'Saved full240Hz pressure and20Hz prediction audit, not a second physics execution'}
    atomic_json(root/'pressure-audit.json',record)
    print(json.dumps(record),flush=True)


if __name__=='__main__':main()
