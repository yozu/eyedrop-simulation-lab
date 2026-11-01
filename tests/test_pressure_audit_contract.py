import json
import sys
from audit_panda_bottle_pressure import main as bottle_main
from audit_panda_fixed_pressure import main as pressure_main

PHYSICS_HZ=240


def test_saved_pressure_audit_uses_explicit_strict_protocol(tmp_path,monkeypatch):
    (tmp_path/'protocol.json').write_text(json.dumps({'frames':300,'open_threshold':.95}))
    row=dict(case='x',condition='natural',closed_measured_samples=1,closed_command_samples=1)
    (tmp_path/'result.json').write_text(json.dumps({'completed':True,'rows':[row]}))
    trial=tmp_path/'natural-x';trial.mkdir()
    samples=[dict(time=i/PHYSICS_HZ,open=.92,gap_m=.015 if i==4 else .0343) for i in range(3601)]
    (trial/'pressure-samples.json').write_text(json.dumps(samples))
    (trial/'trace.json').write_text(json.dumps([{'pressure_interval':{'closed_command_samples':1}}]))
    monkeypatch.setattr(sys,'argv',['audit',str(tmp_path)]);pressure_main()
    result=json.loads((tmp_path/'pressure-audit.json').read_bytes())
    assert result['open_threshold']==.95 and result['records_verified']
    assert result['rows'][0]['closed_measured_samples']==1 and not result['closed_pressure_zero']


def test_partial_zero_samples_do_not_prove_closed_pressure_zero(tmp_path,monkeypatch):
    (tmp_path/'protocol.json').write_text(json.dumps({'frames':300}))
    row={'case':'x','condition':'closed','closed_measured_samples':0,'closed_command_samples':0}
    (tmp_path/'result.json').write_text(json.dumps({'completed':True,'rows':[row]}))
    trial=tmp_path/'closed-x';trial.mkdir()
    (trial/'pressure-samples.json').write_text(json.dumps([{'time':0.,'gap_m':.0343,'open':0.}]))
    (trial/'trace.json').write_text(json.dumps([{'pressure_interval':None}]))
    monkeypatch.setattr(sys,'argv',['audit',str(tmp_path)]);pressure_main()
    report=json.loads((tmp_path/'pressure-audit.json').read_bytes())
    assert not report['records_verified'] and not report['closed_pressure_zero']


def test_strict_open_threshold_has_separate_pressure_report(tmp_path,monkeypatch):
    (tmp_path/'protocol.json').write_text(json.dumps({'frames':300,'bottle_input':'measured'}))
    (tmp_path/'result.json').write_text(json.dumps({'completed':True,'rows':[{'case':'x','condition':'natural','drops':0}]}))
    trial=tmp_path/'natural-x';trial.mkdir()
    samples=[{'time':i/PHYSICS_HZ,'gap_m':.0158 if i==10 else .0159,'open':.92} for i in range(3601)]
    (trial/'pressure-samples.json').write_text(json.dumps(samples))
    monkeypatch.setattr(sys,'argv',['audit',str(tmp_path),'--input-contract','measured']);bottle_main()
    monkeypatch.setattr(sys,'argv',['audit',str(tmp_path),'--input-contract','measured','--open-threshold','.95']);bottle_main()
    legacy=json.loads((tmp_path/'bottle-pressure-audit.json').read_bytes());strict=json.loads((tmp_path/'bottle-pressure-audit-095.json').read_bytes())
    assert legacy['rows'][0]['closed_positive_pressure_seconds']==0.
    assert strict['rows'][0]['closed_positive_pressure_seconds']>0.
    assert strict['open_threshold']==.95
