# PA CB demo — NPUW PagedAttention front-end, PR 1

Demonstrates the first change of the NPUW PA front-end series
([openvino#37282](https://github.com/openvinotoolkit/openvino/pull/37282)):
the OpenVINO GenAI `ContinuousBatchingPipeline` deploying its dynamic,
stateless PA model **through the NPU plugin**, executed 1:1 on the PA
fallback device (CPU at this stage of the series), with token-identical
output to the plain CPU pipeline.

The build also carries a GenAI routing change
([`dneve/pa-frontend-npu-cb-routing`](https://github.com/dylanneve1/openvino.genai/tree/dneve/pa-frontend-npu-cb-routing)):
a plain `LLMPipeline(model, "NPU", NPUW_PA=YES)` is routed through the CB/PA
pipeline, so the PA path is one property away for a normal user.

## Sources

| Component | Branch |
|---|---|
| OpenVINO | [`dylanneve1/openvino @ pa/01-frontend-skeleton`](https://github.com/dylanneve1/openvino/tree/pa/01-frontend-skeleton) (PR [#37282](https://github.com/openvinotoolkit/openvino/pull/37282)) |
| OpenVINO GenAI | [`dylanneve1/openvino.genai @ dneve/pa-frontend-npu-cb-routing`](https://github.com/dylanneve1/openvino.genai/tree/dneve/pa-frontend-npu-cb-routing) |

Build both from source, or use a prebuilt OpenVINO + GenAI archive of those
branches; the runners below take either a wheel-carrying archive or an
activated python environment.

Python prerequisites (latest everything, including optimum-intel from git for
the model export) are in `requirements.txt`. Install order matters: the
requirements first, the build archive's `openvino` / `openvino_genai` wheels
second, so the build's wheels override the PyPI openvino that optimum-intel
pulls in. `run_demo.ps1` does this automatically; for a hand-built env:

```bash
pip install -r requirements.txt
pip install --force-reinstall --no-deps /path/to/build/wheels/*.whl
```

## Model

Qwen3-0.6B, int4 symmetric, stateful export — the same model the PR's parity
evidence used. On the NPU machines it is already at
`C:\npuw\models\current\LLM\Qwen3-0.6B_int4_sym_group-1_dyn_stateful`, which
the scripts pick up automatically. Any int4 stateful LLM export works;
point `MODEL_DIR` at it. To export one:

```bash
optimum-cli export openvino -m Qwen/Qwen3-0.6B \
    --weight-format int4_sym_g-1 --task text-generation-with-past Qwen3-0.6B_int4
```

## Config

`config/npuw_pa.json` is the whole thing — usable with any driver that takes a
benchmark_app-style plugin config on device `NPU`:

```json
{
    "NPU_USE_NPUW": "YES",
    "NPUW_PA": "YES"
}
```

Nothing else is needed at this stage of the series: the PA fallback device is
internal (defaults to CPU; `OPENVINO_NPUW_PA_DEVICE` env var for development),
and `KV_CACHE_PRECISION` is left to the executing device's default — if you do
set it, set the same value for any pipeline you compare against.

## Run

Windows, from an environment that already has the build's wheels:

```powershell
.\run_demo.ps1
```

`ModelDir` defaults to the NPU machines' standard layout
(`C:\npuw\models\current\LLM\Qwen3-0.6B_int4_sym_group-1_dyn_stateful`);
the python scripts use the same default when `MODEL_DIR` is unset. To set up
from a build archive instead:

```powershell
.\run_demo.ps1 -Artifact C:\path\to\<build-archive>.zip [-ModelDir C:\models\Qwen3-0.6B_int4]
```

Linux (env with the build's wheels installed):

```bash
MODEL_DIR=/path/to/Qwen3-0.6B_int4 ./run_demo.sh /path/to/venv
```

Running a script by hand, `--npuw-log` turns on NPUW plugin logging without
touching the environment:

```powershell
python pa_demo.py --npuw-log INFO      # or VERBOSE; pa_parity.py takes it too
```

Three runs land in `logs/`, timestamped:

| Log | Shows |
|---|---|
| `*_01_demo.log` | Batched CB generation + `LLMPipeline` routing, readable output, tok/s |
| `*_02_demo_npuw_info.log` | Same with `OPENVINO_NPUW_LOG_LEVEL=INFO` — the PA front-end visibly engaging |
| `*_03_parity.log` | Greedy token parity, plain `CPU` vs `NPU + NPU_USE_NPUW/NPUW_PA`, three prompt classes (short / >128 / >1024) — expected `PARITY: PASS`, token-identical on all three |

The parity expectation matches the PR's Validation section: token-identical
on all three prompt classes at PR 1. Note for anyone extending the harness:
pass the same `KV_CACHE_PRECISION` to both pipelines if you set it at all —
CPU defaults to a u8 KV cache.

## What to look for in the logs

- Compile succeeds on `NPU` with `NPU_USE_NPUW=YES, NPUW_PA=YES` and no
  other configuration: no geometry stamping, no model mutation — the ports
  the pipeline's KVCacheManager reads are the executing device's own.
- The INFO log names `PACompiledModel` taking the model and the fallback
  device it compiled on.
- `PARITY: PASS`.
