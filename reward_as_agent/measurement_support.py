"""Necessary coverage checks for conditional image-plane motion evidence."""

def measurement_support(physical,entities,event_frames):
    """Zero mask/track agreement cannot support continuous relative motion."""
    # A query about displaced trash depends on that target's tracks as well as
    # hand/tool tracks. No queried entity can silently escape the support check.
    required=[e['entity_id'] for e in entities]
    if len(set(required))!=len(required):raise ValueError('Duplicate requested entity')
    wanted=list(range(min(event_frames),max(event_frames)+1))
    indexed={}
    for obs in physical.get('observations',[]):
        if obs.get('metric')=='conditional_mask_and_tracks' and len(obs.get('entity_ids',[]))==1:
            k=(obs['entity_ids'][0],obs['source_frame_index'])
            if k in indexed:raise ValueError('Duplicate physical measurement')
            indexed[k]=obs
    gaps=[];stats=[]
    for entity in required:
        visible=[];inside=[]
        for frame in wanted:
            obs=indexed.get((entity,frame))
            if obs is None or type(obs.get('points_inside_mask')) is not int or obs['points_inside_mask']<=0:
                gaps.append({'entity_id':entity,'source_frame_index':frame,'reason':'missing measurement or no visible tracked point agrees with mask'})
            if obs is not None:
                visible.append(obs.get('visible_points',0));inside.append(obs.get('points_inside_mask',0))
        stats.append({'entity_id':entity,'minimum_visible_points':min(visible,default=0),'minimum_in_mask_points':min(inside,default=0)})
    complete=physical.get('status')=='ok' and bool(required) and not gaps
    return {'continuous_motion_support':complete,'event_interval':[wanted[0],wanted[-1]],
        'required_entities':stats,'measurement_gaps':gaps,
        'limits':'Nonzero agreement is only a necessary measurement check, not semantic identity, grip, stability, or physical correctness. Gaps are not video defects.'}
