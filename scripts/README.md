# Public script entry points

Configure credentials and local model/tool paths in `.env` before live inference. Run commands from the repository root after `python -m pip install -e .`.

| Entry point | Purpose |
| --- | --- |
| `bash start_evidence_reward.sh` (repository root) | Start the selected reward service |
| `python scripts/run_demos.py --help` | Evaluate bundled videos with the running service and retain tool/source provenance |
| `python scripts/run_wmreward_demo.py --help` | Run a bundled video directly with WMReward and the Agent |
| `python scripts/replay_demo_reward.py --help` | Migrate supported historical evidence through the deterministic reward policy; no fresh inference |
| `python scripts/evidence/build_task_contracts.py --help` | Prepare a frozen task registry from a supplied manifest |
| `python scripts/evidence/check_contract_results.py --help` | Check supplied evaluation records against their task contracts |
| `python scripts/human/review_server.py --help` | Serve a supplied video manifest for independent human review |
| `python scripts/human/human_alignment.py --help` | Prepare manifests and compare independently collected human labels |
| `bash scripts/setup_env.sh` | Create the Python environment |
| `python scripts/render_framework.py` | Regenerate SVG, PNG and PDF framework assets; requires Matplotlib |

`scripts/frozen_v41/` retains the historical package name because the production implementation imports its WMReward worker, transport, protocol and visibility modules. These are required runtime dependencies.

Historical probes, candidate experiments, private proxy/debug scripts and fixed-case diagnostics are not part of the public script collection.
