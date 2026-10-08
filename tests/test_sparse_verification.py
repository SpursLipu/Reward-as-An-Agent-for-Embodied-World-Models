import pytest

from reward_as_agent.evidence_video import bounded_verification_indices


def test_disabled_preserves_every_unique_original_frame():
    assert bounded_verification_indices([12, 0, 4, 4, 7]) == [0, 4, 7, 12]


def test_sparse_keeps_temporal_endpoints_and_off_grid_crop_source():
    result = bounded_verification_indices(range(81), 16, (1, 79))
    assert {0, 1, 79, 80}.issubset(result)
    assert len(result) == 18
    assert result == sorted(set(result))
    assert set(result).issubset(range(81))


def test_short_video_is_not_duplicated_or_padded():
    assert bounded_verification_indices([0, 2, 5], 16, [2]) == [0, 2, 5]


@pytest.mark.parametrize('indices,limit,required', [
    ([], 16, []), ([0, 2], 1, []), ([0, 2], -1, []),
    ([0, 2], 16, [1]), ([-1, 0], 16, []), ([True, 2], 16, [])])
def test_invalid_inventory_fails_instead_of_inventing_evidence(indices, limit, required):
    with pytest.raises(ValueError):
        bounded_verification_indices(indices, limit, required)
