"""Source-grounded adaptive crops; no class labels or reward targets."""
import base64
PLAN="""你为机器人视频选择需要放大核查的视觉区域，不给任务结论或reward。根据实际图片和任务原文，指出一个关键可见问题，例如物体是否仍受夹爪支撑、是否离开桌面、夹爪是否张开，选择1至2个已提供时刻的区域。
区域须覆盖目标及判断接触/支撑所必需的夹爪或参照面；不要只截目标内部。坐标为各原帧全景的0至1000归一化坐标，x从左到右，y从上到下，x0<x1、y0<y1。归一化坐标宽高均须至少100（不是像素单位）。可以选择同一目标的初态和末态。问题中不预设成功或失败。
只输出JSON：{"question":"一个简短视觉核查问题","crops":[{"source_frame_index":实际帧号,"x0":左,"y0":上,"x1":右,"y1":下}]}。不写额外解释。"""
def _plan_validator(value,ids):
 assert set(value)=={'question','crops'} and isinstance(value['question'],str) and value['question'].strip()
 assert 1<=len(value['crops'])<=2
 for c in value['crops']:
  assert set(c)=={'source_frame_index','x0','y0','x1','y1'} and type(c['source_frame_index'])is int and c['source_frame_index'] in ids
  assert all(type(c[k])is int and 0<=c[k]<=1000 for k in ['x0','y0','x1','y1'])
  assert c['x1']-c['x0']>=100 and c['y1']-c['y0']>=100

def review_validator(x,ids):
 assert set(x)=={'initial_state','final_state','visible_change','reason','verdict','confidence','evidence_frames'}
 assert x['verdict'] in ['complete','mostly_complete','partial','failed','unobservable'] and x['confidence'] in ['high','medium','low']
 assert 1<=len(x['evidence_frames'])<=3 and all(type(i)is int and i in ids for i in x['evidence_frames'])
 assert all(isinstance(x[k],str) and x[k].strip() for k in ['initial_state','final_state','visible_change','reason'])

def crop_content(video,plan):
 from reward_as_agent.video import get_cv2
 cv2=get_cv2();parts=[];manifest=[]
 for c in plan['crops']:
  i=c['source_frame_index'];frame=video.frames[i];height,width=frame.shape[:2]
  x0,y0=c['x0']*width//1000,c['y0']*height//1000;x1,y1=c['x1']*width//1000,c['y1']*height//1000
  crop=frame[y0:y1,x0:x1];assert crop.size
  crop=cv2.resize(crop,(2*(x1-x0),2*(y1-y0)),interpolation=cv2.INTER_LINEAR)
  ok,buf=cv2.imencode('.jpg',crop,[cv2.IMWRITE_JPEG_QUALITY,90]);assert ok
  view=f'targeted crop original pixels x={x0}:{x1}, y={y0}:{y1}; display enlargement only'
  manifest.append({'source_frame_index':i,'timestamp_seconds':round(i/video.fps,4),'view':view})
  parts += [{'type':'text','text':f'SOURCE_FRAME {i}; timestamp_seconds={round(i/video.fps,4)}; view={view}. Same original frame, not a new time sample.'},{'type':'image_url','image_url':{'url':'data:image/jpeg;base64,'+base64.b64encode(buf).decode('ascii')}}]
 return parts,manifest


def plan_validator(value, ids):
 try:
  _plan_validator(value,ids)
 except (AssertionError,TypeError,KeyError) as exc:
  raise ValueError("Invalid crop plan: require 1-2 supplied frame IDs and normalized in-bounds boxes of at least 100 by 100") from exc


def full_frame_plan(ids):
    """Use actual supplied frames when adaptive planning is structurally invalid."""
    ordered = sorted(set(ids))
    if not ordered or any(type(i) is not int or i < 0 for i in ordered):
        raise ValueError("Fallback requires nonnegative source frame IDs")
    plan = {"question": "What visible object, gripper and support relationships change between these supplied frames?",
            "crops": [{"source_frame_index": i, "x0": 0, "y0": 0, "x1": 1000, "y1": 1000}
                      for i in sorted(set((ordered[0], ordered[-1])))]}
    plan_validator(plan, set(ordered))
    return plan
