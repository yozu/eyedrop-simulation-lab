"""画像ガード前の要求と区間内に実行した指令を区別する。"""
import hashlib
import json
import math
from pathlib import Path
from .smolvla_observation import ACTION_HZ,COMMAND_TIMING,PressHistory

PHYSICS_SUBSTEPS=12
ACTION_DIM=8
STATE_DIM=32
HISTORY_TOLERANCE=1e-9
FRAME_KEYS={'step','time','state','action','left','right','executed_end_gap_m'}

def guarded_action_contract():
 return {'version':1,'fps':ACTION_HZ,'physics_substeps':PHYSICS_SUBSTEPS,
         'action':'policy_request_before_guard','history_command':'last_applied_substep_gap',
         'guard_required':True,'continuous_press_time':'20hz_end_sample_count'}

def load_guarded_trace(raw,episode):
 root=Path(raw).resolve();directory=(root/episode['id']).resolve()
 if not directory.is_relative_to(root) or directory==root:
  raise ValueError('実演の保存先が範囲外です')
 path=(directory/'executed-substeps.json').resolve()
 if not path.is_relative_to(directory):raise ValueError('物理指令の保存先が範囲外です')
 raw_bytes=path.read_bytes()
 if hashlib.sha256(raw_bytes).hexdigest()!=episode.get('execution_trace_sha256'):
  raise ValueError('物理指令の記録SHAが一致しません')
 return json.loads(raw_bytes)

def validate_guarded_episode(rows,execution_trace,rest_gap,initial_gap_command=None):
 if not rows or len(rows)!=len(execution_trace):raise ValueError('実演と物理指令の数が不一致です')
 history=PressHistory(rest_gap,timing=COMMAND_TIMING);history.reset(rows[0]['state'][:16],rest_gap if initial_gap_command is None else initial_gap_command)
 for index,(row,actions) in enumerate(zip(rows,execution_trace)):
  if set(row)!=FRAME_KEYS or row['step']!=index or not math.isfinite(row['time']) or abs(row['time']-index/ACTION_HZ)>HISTORY_TOLERANCE:
   raise ValueError('観測列または時刻が契約と一致しません')
  if len(actions)!=PHYSICS_SUBSTEPS or any(len(a)!=ACTION_DIM or not all(math.isfinite(v) for v in a) for a in actions):
   raise ValueError('物理区間の実行指令が不正です')
  if len(row['state'])!=STATE_DIM or len(row['action'])!=ACTION_DIM or not all(math.isfinite(v) for v in row['state']+row['action']):
   raise ValueError('観測または教師要求が不正です')
  if not math.isfinite(row['executed_end_gap_m']) or abs(row['executed_end_gap_m']-actions[-1][7])>HISTORY_TOLERANCE:
   raise ValueError('区間末指令と物理記録が一致しません')
  if index:history.append(index,row['state'][:16],execution_trace[index-1][-1][7])
  if any(abs(a-b)>HISTORY_TOLERANCE for a,b in zip(row['state'],history.observation())):
   raise ValueError('実行済み指令の履歴が一致しません')
