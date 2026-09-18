<h1 align="center">Reward as An Agent for Embodied World Models</h1>

<p align="center">
  <a href="https://arxiv.org/abs/2606.19990">📄 Paper</a> ·
  <a href="#quick-start">🚀 Quick Start</a> ·
  <a href="#demo-videos">🎬 Demos</a> ·
  <a href="#citation">📝 Citation</a>
</p>

A multimodal reward agent that evaluates task progress, physical plausibility, and visual quality in embodied world-model rollouts. Supports Doubao/Volcengine Ark and OpenAI-compatible backends.

## Framework

<img src="assets/reward_as_agent_framework.png" alt="Planning Module, Gated Multi-dimensional Reward Module, Reflection Module, and reward aggregation with external-tool feedback" width="1100">

- **Planning Module:** observe visible actions and state changes without task text; separately extract and freeze the requested action, target object, and required end state.
- **Gated Multi-dimensional Reward Module:** jointly assess task completion, physical plausibility, and visual quality against the observations and frozen requirements.
- **Reflection Module:** recheck relevant frames, incorporate WMReward feedback, audit requirement scope, and revise inconsistent judgements.
- **Reward aggregation:** apply fixed scoring and review rules after Reflection, with optional process checks, to return a reward or `needs_review`.

The current release includes required **WMReward** integration, progress-aware rewards, and tool-backed demos. WMReward provides physical evidence—not the final task reward. Additional tools can be connected through the [worker interface](scripts/frozen_v41/worker_client.py) and [evidence hook](scripts/frozen_v41/physics_integration.py).


## Quick Start

Requires Python 3.10+, a model endpoint, and a CUDA environment for [WMReward](#wmreward-deployment).

```bash
python -m pip install -e .
cp .env.example .env
# Configure model credentials and REWARD_WMREWARD_* paths in .env.
python -m reward_as_agent.cli serve
```

See [.env.example](.env.example) for configuration. The service loads WMReward at startup and requires tool-backed Reflection; missing tools or tool failures never silently fall back to model-only evaluation.

Evaluate a video accessible to the server:

```bash
curl -N http://127.0.0.1:7024/eval_video \
  -H "Content-Type: application/json" \
  -d '{"video_path":["/absolute/path/video.mp4"],"prompt":"Place the cube in the bowl.","return_details":true}'
```

`POST /eval_video` streams JSONL results. `GET /health` reports worker readiness.

## Demo Videos

All seven demos were evaluated with real WMReward inference and Reflection. Click a preview for the full video, or a result for its evidence and trace.

| Demo | Preview | Doubao task | Doubao reward | Qwen task | Qwen reward |
| --- | --- | --- | --- | --- | --- |
| Sweep a carton and peel | [<img src="assets/demos/demo_06.gif" width="200" alt="Sweep carton preview">](examples/demo_06/video.mp4) | Complete | [**1.0**](examples/demo_06/response.json) | — | — |
| Sweep several pieces of litter | [<img src="assets/demos/demo_07.gif" width="200" alt="Sweep litter preview">](examples/demo_07/video.mp4) | Complete | [**1.0**](examples/demo_07/response.json) | — | — |
| Cloth manipulation | [<img src="assets/demos/demo_01.gif" width="200" alt="Cloth manipulation preview">](examples/demo_01/video_1.mp4) | Partial | [0.615](examples/demo_01/response.json) | partial | [0.615](runs/qwen38_20260918/final/demo_01) |
| Refrigerator drawer opening | [<img src="assets/demos/demo_02.gif" width="200" alt="Drawer opening preview">](examples/demo_02/video_1.mp4) | Partial | [0.595](examples/demo_02/response.json) | complete | [**1.0**](runs/qwen38_20260918/retry/final/demo_02) |
| Basket handle grasping | [<img src="assets/demos/demo_03.gif" width="200" alt="Basket handle preview">](examples/demo_03/video_0.mp4) | Partial / needs review | [null](examples/demo_03/response.json) | partial | [0.615](runs/qwen38_20260918/final/demo_03) |
| Green cube placing | [<img src="assets/demos/demo_04.gif" width="200" alt="Cube placing preview">](examples/demo_04/video_0.mp4) | Failed | [0](examples/demo_04/response.json) | failed | [0](runs/qwen38_20260918/final/demo_04) |
| Box relocation | [<img src="assets/demos/demo_05.gif" width="200" alt="Box relocation preview">](examples/demo_05/video_0.mp4) | Failed | [0](examples/demo_05/response.json) | failed | [0](runs/qwen38_20260918/final/demo_05) |

These are recorded Agent outputs, not ground-truth labels. The two 1.0 cases are real-robot recordings with 3× previews; evaluation used the original videos. Each [demo folder](examples/) includes `tools.json` (raw tool output), `reports.json` (Reflection), and `run.json` (provenance).

Doubao and Qwen results use the same WMReward and Reflection pipeline. `—` means no valid final output was available in that run.

With the service running, reproduce all demos:

```bash
python scripts/run_demos.py --output runs/my_tool_demos --jobs 2
```

Use a fresh output directory; repeat `--demo demo_XX` to select cases. The runner verifies actual tool completion and Reflection.

## WMReward Deployment

Install the scoring worker in a separate CUDA environment:

```bash
git clone https://github.com/facebookresearch/WMReward.git
cd WMReward
git checkout f2af53737f64f12915f249f3ad18012e6afea1ba
git submodule update --init --recursive vjepa2
python3.10 -m venv .venv-worker
.venv-worker/bin/python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu126
.venv-worker/bin/python -m pip install numpy==1.26.4 decord==0.6.0 diffusers==0.33.1 einops==0.8.1 pillow timm==1.0.20
.venv-worker/bin/python -c 'import torch, torchvision, decord; from utils import compute_vjepa_loss_sliding_window; assert torch.cuda.is_available()'
mkdir -p checkpoints
curl -L https://dl.fbaipublicfiles.com/vjepa2/vitg.pt -o checkpoints/vitg.pt
sha256sum checkpoints/vitg.pt
```

In the Reward Agent repository's `.env`, set:

```dotenv
REWARD_WMREWARD_PYTHON=/absolute/path/WMReward/.venv-worker/bin/python
REWARD_WMREWARD_REPO=/absolute/path/WMReward
REWARD_WMREWARD_CHECKPOINT=/absolute/path/WMReward/checkpoints/vitg.pt
REWARD_WMREWARD_SHA256=67129f011434e605d894e69f2c8e13d9db118deabe59d54bf6e0fa62c2c5cb8e
REWARD_WMREWARD_DEVICE=cuda:0
```

Verify the downloaded digest matches. The checkpoint is approximately 16.5 GB; allow sufficient host RAM and GPU memory (one recorded run used about 8.4 GiB of PyTorch allocation; reserve extra headroom). A compatible NVIDIA driver is required. These packages match the experiment environment; a fresh installation has not been retested. No MAGI-1 generator weights are needed. See the [upstream installation guide](https://github.com/facebookresearch/WMReward#installation) for more details.

For standalone evaluation without an HTTP server, use [scripts/run_wmreward_demo.py](scripts/run_wmreward_demo.py) (`--help` lists worker and checkpoint options).

## Reward & RL Usage

- **Complete / partial:** reward reflects supported completion or effective progress; unmet requirements do not automatically erase progress.
- **Failed:** confirmed failure without effective progress returns `0`.
- **Needs review:** unresolved evidence affecting the reward returns `score: null`, not zero.
- **Errors:** tool or Reflection failures are not valid training results.

`status: success` means evaluation completed, not task success. For RL, use `score` only when `training_eligible: true`; retry or mask review/error samples. `diagnostic_score` and WMReward's raw surprise are not replacement rewards. Trainer-side masking is not implemented here, and reward calibration/stable RL suitability remains unvalidated. Exact rules: [training_reward.py](reward_as_agent/training_reward.py).

## Development

```bash
python -m pytest -q tests
```

## Citation

```bibtex
@misc{li2026rewardagentembodiedworld,
  title={Reward as An Agent for Embodied World Models},
  author={Pu Li and Zhigang Lin and Qiang Wu and Yongxuan Lv and Fei Wang and Shan You},
  year={2026},
  eprint={2606.19990},
  archivePrefix={arXiv},
  primaryClass={cs.AI},
  url={https://arxiv.org/abs/2606.19990}
}
```

When using the external-tool integration, also cite [WMReward](https://arxiv.org/abs/2601.10553) and [V-JEPA 2](https://arxiv.org/abs/2506.09985). External code and weights are not redistributed here; follow [WMReward's CC BY-NC 4.0 license](https://github.com/facebookresearch/WMReward) and [V-JEPA 2's code/model license terms](https://github.com/facebookresearch/vjepa2).
