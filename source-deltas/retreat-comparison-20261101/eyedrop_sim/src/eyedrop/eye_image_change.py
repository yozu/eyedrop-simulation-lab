"""過去画像から予測された眼領域の画素変化だけを数える。"""
import numpy as np
from .face_eye_patches import PATCH_WIDTH_RATIO,PATCH_MIN_WIDTH,PATCH_MAX_WIDTH,PATCH_HEIGHT_RATIO

IMAGE_SIDE=256
CHANNEL_THRESHOLD=5

class EyeImageChange:
 def __init__(self):
  self.previous=None
  self.previous_region=None

 def observe(self,rgb,eye_centers):
  image=np.asarray(rgb);centers=np.asarray(eye_centers,dtype=float)
  if image.shape!=(IMAGE_SIDE,IMAGE_SIDE,3) or image.dtype!=np.uint8:
   raise ValueError('眼画像は256角のRGB整数画素が必要です')
  if centers.shape!=(2,2) or not np.isfinite(centers).all() or (centers<0).any() or (centers>1).any():
   raise ValueError('画像から予測した眼位置が範囲外です')
  width=float(np.clip(np.linalg.norm(centers[0]-centers[1])*PATCH_WIDTH_RATIO,PATCH_MIN_WIDTH,PATCH_MAX_WIDTH))
  y,x=np.mgrid[:IMAGE_SIDE,:IMAGE_SIDE];x=x/(IMAGE_SIDE-1);y=y/(IMAGE_SIDE-1);region=np.zeros((IMAGE_SIDE,IMAGE_SIDE),dtype=bool)
  for cx,cy in centers:region|=(abs(x-cx)<=width/2)&(abs(y-cy)<=width*PATCH_HEIGHT_RATIO/2)
  current=image.astype(np.int16)
  changed=None if self.previous is None else int((np.any(np.abs(current-self.previous)>CHANNEL_THRESHOLD,axis=2)&self.previous_region).sum())
  self.previous=current;self.previous_region=region
  return changed

 def observe_visible(self,rgb,eye_centers,robot_mask):
  mask=np.asarray(robot_mask)
  if mask.shape!=(IMAGE_SIDE,IMAGE_SIDE) or mask.dtype!=np.bool_:
   raise ValueError('自己形状マスクは256角の真偽値画像が必要です')
  previous_region=self.previous_region
  changed=self.observe(rgb,eye_centers)
  if (mask & self.previous_region).any() or (previous_region is not None and (mask & previous_region).any()):
   self.previous=None;self.previous_region=None
   return None
  return changed
