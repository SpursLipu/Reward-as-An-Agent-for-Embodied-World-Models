"""Offline, resident WMReward worker. Surprise is model evidence, not probability."""
import argparse
import contextlib
import copy
import json
import math
from pathlib import Path
import sys
import time

from scripts.frozen_v41.tool_protocol import evidence_id, video_sha256

WM_REVISION = 'f2af53737f64f12915f249f3ad18012e6afea1ba'
VJEPA_REVISION = 'c2963a47433ecca0ad4f06ec28bcfa8cb5b5cefb'
DEFAULTS = {'window_size': 16, 'context_frames': 8, 'stride': 8,
            'seed': 42, 'max_frames': 49}


def validated_params(request):
    supplied = request.get('params', {})
    if not isinstance(supplied, dict) or set(supplied) - set(DEFAULTS):
        raise ValueError('Unsupported WMReward parameters')
    params = {**DEFAULTS, **supplied}
    if any(type(value) is not int for value in params.values()):
        raise ValueError('WMReward parameters must be integers')
    window, context = params['window_size'], params['context_frames']
    if window < 4 or window % 2 or context < 2 or context % 2 or context >= window:
        raise ValueError('Window/context must respect two-frame tubelets')
    if not 1 <= params['stride'] <= window or not window <= params['max_frames'] <= 256:
        raise ValueError('Invalid stride or frame budget')
    if not 0 <= params['seed'] < 2**32 - 256:
        raise ValueError('Invalid seed')
    return params


class WMRewardWorker:
    def __init__(self, repo, checkpoint, checkpoint_sha256, device='cuda:0'):
        started = time.monotonic()
        if len(checkpoint_sha256) != 64 or video_sha256(checkpoint) != checkpoint_sha256:
            raise ValueError('Checkpoint SHA256 mismatch')
        import torch
        self.torch = torch
        self.device = torch.device(device)
        if self.device.type != 'cuda':
            raise ValueError('This worker requires the validated CUDA execution path')
        torch.cuda.set_device(self.device)
        sys.path.insert(0, str(Path(repo) / 'vjepa2'))
        sys.path.insert(0, str(Path(repo)))
        from src.hub.backbones import vjepa2_vit_giant, _clean_backbone_key
        from utils import compute_vjepa_loss_sliding_window
        self.compute = compute_vjepa_loss_sliding_window
        encoder, predictor = vjepa2_vit_giant(pretrained=False)
        state = torch.load(checkpoint, map_location='cpu', weights_only=True)
        self.load_keys = {}
        for name, model in [('encoder', encoder), ('predictor', predictor)]:
            keys = model.load_state_dict(_clean_backbone_key(state[name]), strict=False)
            # Official loader permits pos_embed because the architecture uses RoPE.
            if keys.missing_keys or any(k != 'pos_embed' for k in keys.unexpected_keys):
                raise ValueError(f'Unexpected checkpoint keys for {name}: {keys}')
            self.load_keys[name] = {'missing': keys.missing_keys, 'unexpected': keys.unexpected_keys}
        del state
        self.encoder = encoder.to(self.device).eval()
        self.target_encoder = copy.deepcopy(self.encoder).eval()
        self.predictor = predictor.to(self.device).eval()
        self.model_revision = f'{WM_REVISION}:vjepa2-{VJEPA_REVISION}:vitg:{checkpoint_sha256}'
        self.preprocessing_revision = 'wmreward-official-linspace49-resize256-fp32-v1'
        self.cold_start_ms = (time.monotonic() - started) * 1000

    def evaluate(self, request):
        started = time.monotonic()
        result = {'status': 'error', 'tool_name': 'wmreward', 'raw_score': None,
                  'score_direction': 'lower_better', 'score_semantics': 'mean_one_minus_cosine',
                  'model_revision': self.model_revision,
                  'preprocessing_revision': self.preprocessing_revision,
                  'cache_hit': False, 'observations': [], 'artifact_refs': [],
                  'evidence_scope': 'whole_sampled_video', 'peak_memory_mb': None,
                  'cold_start_ms': self.cold_start_ms}
        try:
            import numpy as np
            from decord import VideoReader
            from torchvision.transforms.functional import resize
            if request.get('operation') != 'check_physics':
                raise ValueError('WMReward only supports check_physics')
            if any(request.get(k) is not None for k in ('interval_s', 'frame_indices', 'rule')):
                raise ValueError('WMReward has no rule/subset adapter')
            params = validated_params(request)
            result['params'] = params
            result['evidence_id'] = evidence_id({**request, 'params': params},
                self.model_revision, self.preprocessing_revision)
            path = request['video_uri']
            digest = video_sha256(path)
            if digest != request['video_sha256']:
                raise ValueError('Source video hash mismatch')
            reader = VideoReader(path)
            count, fps = len(reader), float(reader.get_avg_fps())
            if count == 0 or not math.isfinite(fps) or fps <= 0:
                raise ValueError('Invalid video time base')
            indices = np.linspace(0, count - 1, min(params['max_frames'], count), dtype=int)
            result.update(video_sha256=digest, decoded_frames=count, original_fps=fps,
                          frame_indices=indices.tolist(), timestamps_s=(indices / fps).tolist())
            if len(indices) < params['window_size']:
                result.update(status='abstain', abstain_reason='insufficient_frames')
            else:
                frames = reader.get_batch(indices).asnumpy()
                tensor = self.torch.from_numpy(frames).permute(3, 0, 1, 2).float()
                tensor = resize(tensor.permute(1, 0, 2, 3), [256, 256]).permute(1, 0, 2, 3)
                tensor = ((tensor / 127.5) - 1).unsqueeze(0).to(self.device)
                self.torch.cuda.reset_peak_memory_stats(self.device)
                with self.torch.inference_mode():
                    score = self.compute(tensor, self.encoder, self.target_encoder, self.predictor,
                        img_size=256, window_size=params['window_size'], loss_exp=2,
                        masking_mode='causal', context_frames=params['context_frames'],
                        is_vae_output=True, seed=params['seed'], stride=params['stride'], mode='mean').item()
                self.torch.cuda.synchronize(self.device)
                if not math.isfinite(score):
                    raise ValueError('Nonfinite surprise score')
                result.update(status='ok', raw_score=score, raw_output={'surprise': score},
                    peak_memory_mb=self.torch.cuda.max_memory_allocated(self.device) / 1024**2,
                    window_frame_indices=[indices[i:i+params['window_size']].tolist()
                        for i in range(0, len(indices)-params['window_size']+1, params['stride'])])
            if video_sha256(path) != digest:
                raise ValueError('Source video changed during inference')
        except Exception as exc:
            result.update(status='error', raw_score=None, error_type=type(exc).__name__, error=str(exc))
        result['latency_ms'] = (time.monotonic() - started) * 1000
        return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--checkpoint-sha256', required=True)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    with contextlib.redirect_stdout(sys.stderr):
        worker = WMRewardWorker(args.repo, args.checkpoint, args.checkpoint_sha256, args.device)
    for line in sys.stdin:
        try:
            request = json.loads(line)
            with contextlib.redirect_stdout(sys.stderr):
                result = worker.evaluate(request)
        except Exception as exc:
            result = {'status': 'error', 'raw_score': None, 'error_type': type(exc).__name__}
        print(json.dumps(result, ensure_ascii=False, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
