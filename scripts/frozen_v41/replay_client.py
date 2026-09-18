"""Reuse frozen actual tool outputs for matched agent comparisons."""
import copy
import hashlib
import json
from pathlib import Path
import time


def request_key(request):
    # A path is a locator; the exact source bytes and every semantic query field bind evidence.
    return json.dumps({k: v for k, v in request.items() if k != 'video_uri'},
                      sort_keys=True, ensure_ascii=False, allow_nan=False)


class ReplayClient:
    def __init__(self, path, repeat=0):
        self.path = str(Path(path).resolve())
        data = Path(path).read_bytes()
        self.artifact_sha256 = hashlib.sha256(data).hexdigest()
        self.index = {}
        for line_no, line in enumerate(data.decode().splitlines(), 1):
            row = json.loads(line)
            if row.get('repeat') != repeat:
                continue
            key = request_key(row['request'])
            if key in self.index:
                raise ValueError('Ambiguous replay key; choose one recorded repeat')
            self.index[key] = (line_no, row['result'])
        if not self.index:
            raise ValueError('No replay records for requested repeat')

    async def evaluate(self, request):
        started = time.monotonic()
        entry = self.index.get(request_key(request))
        if entry is None:
            return {'status': 'error', 'raw_score': None, 'error_type': 'ReplayMiss',
                    'error': 'No exact source/query match in frozen tool outputs', 'cache_hit': False}
        line_no, original = entry
        result = copy.deepcopy(original)
        result.update(cache_hit=True, recorded_latency_ms=original.get('latency_ms'),
                      latency_ms=(time.monotonic()-started)*1000,
                      replay_provenance={'artifact': self.path, 'artifact_sha256': self.artifact_sha256,
                        'line': line_no, 'original_result_sha256': hashlib.sha256(json.dumps(
                            original,sort_keys=True,ensure_ascii=False,allow_nan=False).encode()).hexdigest()})
        return result

    async def close(self):
        pass
