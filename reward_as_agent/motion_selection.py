"""Select display tracks; inspection-region membership is not object identity."""
import math


def select_display_tracks(tracks, limit=18, spacing=32, inspection_quota=0):
    if limit < 1 or not 0 <= inspection_quota <= limit or spacing < 0:
        raise ValueError('Invalid track display budget')
    ranked = sorted(tracks, key=lambda t: -math.hypot(*t['displacement_xy']))
    selected = []

    def append_if_separate(track):
        if all(math.dist(track['start_xy'], other['start_xy']) > spacing for other in selected):
            selected.append(track)

    if inspection_quota:
        for track in ranked:
            if track['group'].startswith('inspection_region_'):
                append_if_separate(track)
                if len(selected) >= inspection_quota:
                    break
    for track in ranked:
        if len(selected) >= limit:
            break
        append_if_separate(track)
    return selected
