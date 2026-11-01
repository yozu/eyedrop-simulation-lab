"""画像から得た変化量で物理区間の押圧を中断する。"""
import math
from .panda_world import BOTTLE_DEPTH_M,MIN_GAP_M,MAX_GAP_M,SUBSTEPS

class ImagePressGuard:
 def __init__(self,rest_gap):
  if not math.isfinite(rest_gap) or not BOTTLE_DEPTH_M<rest_gap<=MAX_GAP_M:
   raise ValueError('解除幅が範囲外です')
  self.rest_gap=rest_gap
  self.latched=False

 def apply(self,command_gap,changed_pixels):
  if not math.isfinite(command_gap) or not MIN_GAP_M<=command_gap<=MAX_GAP_M:
   raise ValueError('指幅指令が範囲外です')
  if changed_pixels is not None and (type(changed_pixels) is not int or changed_pixels<0):
   raise ValueError('画像差分が不正です')
  if command_gap<BOTTLE_DEPTH_M and (changed_pixels is None or changed_pixels>0):
   self.latched=True
  return (self.rest_gap if self.latched else command_gap,self.latched)

def step_guarded_interval(world,request,observe,guard,*,normal_release_speed,guarded_release_speed):
 if len(request)!=8 or not all(math.isfinite(v) for v in request):raise ValueError('政策の要求が不正です')
 if not all(math.isfinite(v) and v>0 for v in (normal_release_speed,guarded_release_speed)):
  raise ValueError('解除速度が不正です')
 actions=[];changes=[];segments=[];domains=[];motion=[];pressure=[];contacts=[];hold=False;observation_error=None
 for _ in range(SUBSTEPS):
  if observation_error is None:
   try:changed=observe()
   except (OSError,RuntimeError,ValueError) as error:
    observation_error={'type':type(error).__name__,'message':str(error)}
    guard.latched=True;hold=True;changed=None
  else:changed=None
  was_latched=guard.latched;gap,latched=guard.apply(request[7],changed)
  if latched and not was_latched:hold=True
  action=list(request);action[7]=gap
  if hold:action[:7]=world.own_state()[:7]
  world.finger_release_speed_m_s=guarded_release_speed if latched else normal_release_speed
  result=world.step(action,audit_pressure=True,audit_motion=True,substeps=1)
  actions.append(action);changes.append(changed)
  samples=result.get('pressure_samples',[])
  if samples:
   if not pressure:pressure.append(samples[0])
   if samples[-1]['time']>pressure[-1]['time']:pressure.append(samples[-1])
  segments.extend(result.get('segments',[]));domains.extend(result.get('drop_domain_samples',[]));motion.extend(result.get('motion_samples',[]))
  contacts.extend(result.get('contacts',[]))
  if contacts:break
 return {'complete':len(actions)==SUBSTEPS and not contacts and observation_error is None,'observation_error':observation_error,'substep_actions':actions,'observed_changes':changes,
         'last_applied_gap_m':actions[-1][7],'pressure_samples':pressure,'segments':segments,
         'drop_domain_samples':domains,'motion_samples':motion,'contacts':contacts,'time':world.time}
