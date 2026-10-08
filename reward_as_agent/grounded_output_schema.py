"""Candidate generation constraints; original semantic validators remain authoritative."""
def grounded_schema(frame_ids, contract=None, blind=False):
    def obj(properties):return {'type':'object','properties':properties,'required':list(properties),'additionalProperties':False}
    def arr(items):return {'type':'array','items':items}
    def enum(values):return {'type':'string','enum':list(values)}
    text={'type':'string','minLength':1}
    evidence=arr({'type':'string','pattern':'^E[1-9][0-9]*$'})
    observations=arr(obj({'id':{'type':'string','pattern':'^E[1-9][0-9]*$'},'frames':{'type':'array','minItems':1,'items':{'type':'integer','enum':sorted(set(frame_ids))}},'description':text}))
    if blind:return obj({'observations':observations,'uncertainties':arr(text)})
    def assessment(verdicts,extra=None):
        return obj({'verdict':enum(verdicts+['unobservable']),'confidence':enum(['high','medium','low']),'evidence':evidence,'reason':text,**(extra or {})})
    issue=obj({'kind':enum(['interpenetration','contact','shape','motion','other']),'severity':enum(['minor','major']),'certainty':enum(['confirmed','uncertain']),'evidence':evidence,'reason':text,'alternative_explanation':text})
    task=obj({k:enum([v]) for k,v in contract['task'].items()}) if contract else obj({k:text for k in ['requested_action','target_description','final_state_requirement']})
    properties={'schema_version':enum(['evidence-v2']),'task':task,'observations':observations,
        'task_assessment':assessment(['complete','mostly_complete','partial','failed'],{'target_match':enum(['match','mismatch','uncertain'])}),
        'physics_assessment':assessment(['plausible','minor_defect','major_defect'],{'issues':arr(issue)}),
        'visual_assessment':assessment(['clear','minor_degradation','severe_degradation']),'uncertainties':arr(text)}
    if contract:
        properties['requirement_checks']=arr(obj({'requirement_id':enum([r['id'] for r in contract['requirements']]),'status':enum(['met','partial','not_met','uncertain','unobservable']),'evidence':evidence,'reason':text}))
    return obj(properties)
