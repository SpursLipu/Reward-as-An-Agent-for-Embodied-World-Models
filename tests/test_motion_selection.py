import pytest
from reward_as_agent.motion_selection import select_display_tracks


def track(i, x, motion, group='full_frame_grid'):
    return {'id':i,'start_xy':[x,0],'displacement_xy':[motion,0],'group':group}


def test_region_quota_retains_low_motion_evidence_under_background_motion():
    background = [track(i, i*40, 30-i) for i in range(18)]
    region = [track(18+i, 800+i*40, 0.1, 'inspection_region_0') for i in range(6)]
    assert all(t['id'] < 18 for t in select_display_tracks(background+region))
    result = select_display_tracks(background+region, inspection_quota=6)
    assert len(result) == 18
    assert {t['id'] for t in region}.issubset(t['id'] for t in result)


def test_overlapping_seeds_do_not_duplicate_a_display_location():
    tracks = [track(0,0,10),track(1,0,1,'inspection_region_0'),
              track(2,0,1,'inspection_region_1'),track(3,40,5)]
    assert [t['id'] for t in select_display_tracks(tracks,inspection_quota=6)] == [1,3]


def test_empty_or_unavailable_region_does_not_invent_tracks():
    assert select_display_tracks([],inspection_quota=6) == []
    tracks = [track(1,0,2),track(2,40,1)]
    assert select_display_tracks(tracks,inspection_quota=6) == tracks


@pytest.mark.parametrize('kwargs',[{'limit':0},{'inspection_quota':19},{'spacing':-1}])
def test_invalid_budgets_fail(kwargs):
    with pytest.raises(ValueError):
        select_display_tracks([],**kwargs)
