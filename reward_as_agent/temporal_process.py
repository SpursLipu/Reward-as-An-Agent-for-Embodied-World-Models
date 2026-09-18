"""Task-aware local observations without access to final outcome or reward labels."""
import asyncio
from reward_as_agent.evidence_prompts import _manifest_context, _GROUNDING
from reward_as_agent.evidence_video import uniform_indices

VERSION='local-process-six-second-overlap-v1'


def segment_indices(total,fps):
    if total<1 or fps<=0: raise ValueError('Invalid video timeline')
    stride=max(1,round(6*fps)); padding=max(1,round(.5*fps))
    segments=[]
    for start in range(0,total,stride):
        lo=max(0,start-padding); hi=min(total,start+stride+padding)
        segments.append([lo+i for i in uniform_indices(hi-lo,12)])
    return segments


def validate_local(value,valid_ids):
    if not isinstance(value,dict) or set(value)!={'events','limitations'}: raise ValueError('Invalid local observation fields')
    if not isinstance(value['events'],list) or len(value['events'])>4: raise ValueError('At most four local events')
    if not isinstance(value['limitations'],str): raise ValueError('Need local limitations text')
    for event in value['events']:
        if not isinstance(event,dict) or set(event)!={'frames','description','execution_error','error_kind','alternative_explanation'}:
            raise ValueError('Invalid local event fields')
        ids=event['frames']
        if not isinstance(ids,list) or len(ids)<2 or len(set(ids))!=len(ids) or any(type(i) is not int or i not in valid_ids for i in ids):
            raise ValueError('Local event requires at least two distinct supplied frames')
        if event['execution_error'] not in {'none','confirmed','uncertain'}: raise ValueError('Invalid local certainty')
        if event['error_kind'] not in {'none','failed_grasp','loss_of_control','counterproductive_motion','repeated_failed_attempt'}:
            raise ValueError('Invalid local error kind')
        if (event['execution_error']=='none') != (event['error_kind']=='none'): raise ValueError('Local error category mismatch')
        for k in ('description','alternative_explanation'):
            if not isinstance(event[k],str) or not event[k].strip(): raise ValueError('Need local evidence explanation')
    return value


def local_prompt(manifest):
    return _manifest_context(manifest)+_GROUNDING+'''
你只看整段rollout中的一个局部时间窗口，无法知道最终是否成功。描述此处实际动作、
手与工具的相对变化、工具与目标的变化。不要用“看起来正在完成任务”的整体印象补写抓取。
只在实际可见时声称握住工具：手靠近/遮挡工具不足以证明握持；观察接近、接触、离开时工具
是否随手运动。抓取失败须有实际抓取尝试且未建立控制的时序证据，单纯准备、松手或工具
本来不需要抬起不能算失败。判断推动动作也以工具和目标的真实变化为准。
只报告最多4项局部事件。区分正常有效动作与具体执行失误；未完成全任务或未归位不是局部
执行错误。短暂接触、合理调整、必要分步、等待和正常释放不算错误。
错误类型仅允许：failed_grasp（有证据的抓取失败）、loss_of_control（非正常释放导致失控）、
counterproductive_motion（动作明确破坏/逆转任务进展）、repeated_failed_attempt（重复无效尝试）。
必须考虑正常替代解释。证据不充分用uncertain，不补造失败次数，不估计错误时间百分比。
只返回JSON，事件frames必须为实际提供的至少两个不同原帧号：
{"events":[{"frames":[0,1],"description":"可见的前后变化，不猜测操作者意图",
 "execution_error":"none|confirmed|uncertain","error_kind":"none|failed_grasp|loss_of_control|counterproductive_motion|repeated_failed_attempt",
 "alternative_explanation":"正常替代解释，以及现有图片是否能够排除"}],
 "limitations":"本局部窗口中的具体证据限制；没有则空字符串"}
none事件的error_kind必须none，其余必须选择具体类型。无法辨认时events可为空。
'''


async def observe_segments(pipeline,video,description,trace):
    # Two concurrent calls, bounded independently of video length.
    semaphore=asyncio.Semaphore(2)
    async def one(index,ids):
        async with semaphore:
            local_trace=[]
            try:
                report=await pipeline.stage('local_process_observation',local_prompt(video.manifest(ids)),
                    {'task_description':description,'window_index':index,'scope':'Only this local interval; final outcome not supplied'},
                    video.content(ids),ids,validate_local,local_trace)
                return {'window_index':index,'provided_frames':ids,'status':'ok','report':report,'trace':local_trace}
            except Exception as e:
                return {'window_index':index,'provided_frames':ids,'status':'error','error':str(e),'trace':local_trace}
    results=await asyncio.gather(*(one(i,ids) for i,ids in enumerate(segment_indices(len(video.frames),video.fps))))
    for r in results:
        trace.extend(r.pop('trace'))
    return results


def refinement_from_local(existing,results,max_frames=81):
    candidates=set()
    for result in results:
        for event in result.get('report',{}).get('events',[]):
            if event['execution_error'] in {'confirmed','uncertain'}: candidates.update(event['frames'])
    additional=sorted(candidates-set(existing)); budget=max(0,max_frames-len(existing))
    if len(additional)>budget:
        additional=[] if budget==0 else [additional[i] for i in uniform_indices(len(additional),max(2,budget))][:budget]
    return sorted(set(existing)|set(additional))
