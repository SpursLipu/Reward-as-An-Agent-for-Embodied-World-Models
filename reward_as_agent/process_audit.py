"""Separate task execution quality from final success; provisional evidence gates."""
import copy
from reward_as_agent.evidence_prompts import _manifest_context, _GROUNDING

VERSION='outcome-process-evidence-v4.1-provisional'


def process_prompt(manifest):
    return _manifest_context(manifest)+_GROUNDING+'''
独立检查机器人执行过程。最终成功与执行质量分别判断：最终成功也可能经历空扫、错抓、
掉落、推动目标远离目的地或反复无效重试；真实机器人也会执行错误动作，这不等于违反物理。
正常必要调整、合理分步操作、动作慢、任务未要求的复位或美观不能算错误。
只依据实际图片判断，不可从视频文件名、已有分数、工具正常运行或模型惊讶度推导质量。
输入原任务定义目标。不得改变目标，不得把过程错误偷偷改成任务失败。

检查 action_evidence：observed 表示看见任务相关动作带来的实质状态变化及可理解的过程；
absent 表示画面可以辨认，但只展示静态姿态/末态，未展示所要求的执行过程；
uncertain 表示关键过程看不清，且这种歧义会改变结论。末态正确不能替代过程证据。
检查 continuity：coherent 表示展示的变化具有合理时序；discontinuous 表示可见断层、
乱序或状态跳变实际破坏核心动作的因果可理解性；uncertain 用于有具体异常但原因/影响不清。
普通抽帧间隔不能单独证明 discontinuous，也不自动导致 uncertain。
检查 quality：no_observed_error、minor_errors、substantial_errors、unobservable。
minor_errors 是短暂局部失误且主要执行过程有效；substantial_errors 是反复或长时间的无效/
错误执行，或显著破坏任务进展的失误；不要凭图片数估算全片错误百分比。
为质量与过程结论列出实际支持及反证。每条 claim 必须直接引用本次实际提供的原帧。
时序变化、执行错误和 absent 结论必须有至少两个不同原帧支持；不能靠凑帧号满足要求。
结构要求：action_evidence 不是 uncertain 时，claims 必须有独立的 kind="action" 项；
continuity 不是 uncertain 时，必须另有独立的 kind="continuity" 项，即使未见异常也需引用
支持时序可理解的至少两个原帧。reason 里的解释不能代替对应 claim。
quality 为 minor_errors/substantial_errors 时，另加 kind="execution_error" 项；
quality 为 no_observed_error 时，不得同时声称有 execution_error。
只有会影响这些结论的具体歧义才用 uncertain/unobservable，不要求排除一切隐藏可能。
只返回下面 JSON，枚举选择一个，不输出数值奖励：
{"action_evidence":"observed|absent|uncertain",
 "continuity":"coherent|discontinuous|uncertain",
 "quality":"no_observed_error|minor_errors|substantial_errors|unobservable",
 "claims":[{"kind":"action","frames":[0,1],"description":"支持动作证据结论的可见依据"},
           {"kind":"continuity","frames":[0,1],"description":"支持时序结论的可见依据"}],
 "reason":"各结论由哪些可见事实支持，以及具体限制"}
上述帧号仅为结构占位，实际必须换成本次图像清单中的支持帧。按上述规则增删 claim 项。
'''


def validate_process(value,valid_ids):
    if not isinstance(value,dict) or set(value)!={'action_evidence','continuity','quality','claims','reason'}:
        raise ValueError('Invalid process audit fields')
    for key,choices in {
        'action_evidence':{'observed','absent','uncertain'},
        'continuity':{'coherent','discontinuous','uncertain'},
        'quality':{'no_observed_error','minor_errors','substantial_errors','unobservable'}}.items():
        if value[key] not in choices: raise ValueError('Invalid '+key)
    if not isinstance(value['reason'],str) or not value['reason'].strip(): raise ValueError('Need process reason')
    if not isinstance(value['claims'],list): raise ValueError('Need process claims')
    kinds=set()
    for claim in value['claims']:
        if not isinstance(claim,dict) or set(claim)!={'kind','frames','description'}: raise ValueError('Invalid claim')
        if claim['kind'] not in {'action','continuity','execution_error','counterevidence'}: raise ValueError('Invalid claim kind')
        ids=claim['frames']
        if not isinstance(ids,list) or not ids or any(type(i) is not int or i not in valid_ids for i in ids):
            raise ValueError('Claim must cite supplied source frames')
        if len(set(ids))!=len(ids): raise ValueError('Repeated claim frames')
        if claim['kind']!='counterevidence' and len(ids)<2: raise ValueError('Process claims require temporal evidence')
        if not isinstance(claim['description'],str) or not claim['description'].strip(): raise ValueError('Need claim description')
        kinds.add(claim['kind'])
    if value['action_evidence']!='uncertain' and 'action' not in kinds: raise ValueError('Need action evidence')
    if value['continuity']!='uncertain' and 'continuity' not in kinds: raise ValueError('Need continuity evidence')
    if value['quality'] in {'minor_errors','substantial_errors'} and 'execution_error' not in kinds:
        raise ValueError('Execution penalty needs visible errors')
    if value['quality']=='no_observed_error' and 'execution_error' in kinds:
        raise ValueError('No observed error contradicts an execution error claim')
    return value


def score_process(scoring,report,process):
    """No quality floor; unknown remains unknown, with inspectable components."""
    valid={i for c in process['claims'] for i in c['frames']}
    validate_process(process,valid)
    out=copy.deepcopy(scoring)
    reasons=[]
    task=report['task_assessment']; physical=report['physics_assessment']; visual=report['visual_assessment']
    factors={
        'task':{'failed':0.,'partial':.45,'mostly_complete':.8,'complete':1.}.get(task['verdict']),
        'execution':{'no_observed_error':1.,'minor_errors':.85,'substantial_errors':.6}.get(process['quality']),
        'physics':{'plausible':1.,'minor_defect':.75,'major_defect':0.}.get(physical['verdict']),
        'visual':{'clear':1.,'minor_degradation':.9,'severe_degradation':0.}.get(visual['verdict'])}
    if process['action_evidence']=='uncertain': reasons.append('process: key action evidence uncertain')
    if process['continuity']=='uncertain': reasons.append('process: concrete continuity ambiguity unresolved')
    if process['quality']=='unobservable': reasons.append('process: execution quality unobservable')
    score=None
    if scoring['total_score'] is not None:
        if task['target_match']=='mismatch' or process['action_evidence']=='absent' or process['continuity']=='discontinuous':
            score=0.
        elif all(v is not None for v in factors.values()):
            score=1.
            for factor in factors.values(): score*=factor
            score=round(score,8)
    out['total_score']=score
    out['review_reasons'].extend(reasons)
    out['review_required']=bool(out['review_reasons'])
    out['scoring_version']=VERSION
    out['reward_gates']={'policy':'evidence_process','ungated_score':scoring['total_score'],
        'factors':factors,'process_audit':process,'reasons':reasons,
        'calibration':'Provisional ordinal rubric; not fitted human numerical rewards'}
    return out


def scope_claim_indices(process):
    return [i for i,c in enumerate(process['claims']) if c['kind']=='execution_error'
        or c['kind']=='continuity' and process['continuity']=='discontinuous'
        or c['kind']=='action' and process['action_evidence']=='absent']


def validate_process_scope(value,process):
    if not isinstance(value,dict) or set(value)!={'checks'} or not isinstance(value['checks'],list):
        raise ValueError('Expected process scope checks')
    expected=scope_claim_indices(process)
    if [c.get('claim_index') for c in value['checks']]!=expected:
        raise ValueError('Check each execution error and hard-gate claim exactly once in original order')
    for check in value['checks']:
        if set(check)!={'claim_index','status','reason'} or check['status'] not in {'supported_scope','extra_requirement','no_execution_error','sampling_gap_only','uncertain'}:
            raise ValueError('Invalid process scope check')
        if not isinstance(check['reason'],str) or not check['reason'].strip(): raise ValueError('Need scope reason')
    return value


PROCESS_SCOPE_PROMPT='''你只审核过程扣分及硬门控是否有合法评价依据，不判断图像真伪、不预测奖励。
输入含冻结任务原文、实际帧清单和视觉过程报告。仅审核required_claim_indices指定的每条claim，
claim_index为claims数组中从0开始的位置。指定项包含execution_error，以及判discontinuous时的
continuity、判absent时的action。其余不审核。
supported_scope：描述明确的抓取失败、失控掉落、无效重试、把目标推离任务方向等执行错误；
或任务原文明示条件的违反。对continuity是描述了不能仅用采样遗漏解释的可见时序/因果破坏；
对action是可辨认但仅呈现静态/末态等，确实无任务执行过程证据。过程错误必须涉及实际执行，
不能仅把未最终完成重述成过程错误，也不能用最终未完成推导完全没有发生有效动作。
extra_requirement：把任务未要求的工具归位、工具放入容器、额外整理、姿态美观等当作错误。
no_execution_error：只是未完成任务的末态、等待、普通合理调整，或并未描述具体错误动作。
sampling_gap_only：仅以提供的两帧间姿态/位置不同、缺少中间释放/撤离画面，推导视频自身断层。
务必核对帧清单：间隔采样可自然省略中间动作；例如机器人在相隔若干原帧的图片中已经松手
或撤离，不能单凭状态变化证明原视频被剪辑或不连续。反复乱序、逆向状态重现等具体证据仍需保留。
uncertain：文字不足以确认此扣分理由是否落在上述合法范围。
最终成功不洗掉可见失误，最终失败也不自动证明每一步执行错误。任务要求以原文为准。
只输出JSON：{"checks":[{"claim_index":0,"status":"supported_scope|extra_requirement|no_execution_error|sampling_gap_only|uncertain","reason":"明确依据"}]}
required_claim_indices为空才checks=[]。这是文字依据审查，不是新的视觉事实；不要补造物体移动或失误。
'''


async def audit_process_with_scope(pipeline,description,contract,manifest,frames,valid_ids,coverage,trace,local_observations=None,initial_report=None):
    payload={'task_description':description,'task_contract':contract,'video_coverage':coverage}
    if local_observations is not None:
        payload['independent_local_observations']=local_observations
        payload['local_observation_rule']='Local observations are fallible. Recheck cited supplied frames; no inherited classifications or scores.'
    if initial_report is None:
        report=await pipeline.stage('execution_process_audit',process_prompt(manifest),payload,
            frames,valid_ids,validate_process,trace)
    else:
        report=copy.deepcopy(initial_report);validate_process(report,valid_ids)
        trace.append({'stage':'frozen_initial_process_report','output':copy.deepcopy(report)})
    audits=[]
    for attempt in range(2):
        required=scope_claim_indices(report)
        if not required:
            audits.append({'checks':[]}); break
        scope=await pipeline.stage('execution_process_scope_audit',PROCESS_SCOPE_PROMPT,
            {'task_description':description,'task_contract':contract,'process_report':report,
             'required_claim_indices':required,'frame_manifest':manifest,'video_coverage':coverage},[],[],
            lambda v,_:validate_process_scope(v,report),trace)
        audits.append(scope)
        if all(c['status']=='supported_scope' for c in scope['checks']) or attempt==1: break
        report=await pipeline.stage('execution_process_scope_repair',process_prompt(manifest),
            {**payload,'draft_process_report':report,'scope_audit':scope,
             'repair_instruction':'Recheck images and scope. Do not invent new errors to preserve a quality category. Keep real visible execution errors; remove unsupported added requirements.'},
            frames,valid_ids,validate_process,trace)
    return report,audits
