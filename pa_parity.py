#!/usr/bin/env python3
"""CB pipeline parity: plain CPU vs NPU+NPUW_PA (PA front-end), greedy."""
import os
import sys

# --npuw-log LEVEL (e.g. INFO, VERBOSE) turns on NPUW plugin logging; it must
# land in the environment before the plugin compiles anything.
if "--npuw-log" in sys.argv:
    i = sys.argv.index("--npuw-log")
    try:
        os.environ["OPENVINO_NPUW_LOG_LEVEL"] = sys.argv[i + 1]
    except IndexError:
        sys.exit("--npuw-log needs a level: ERROR, WARNING, INFO, VERBOSE")
    del sys.argv[i:i + 2]

import openvino_genai as og

DEFAULT_MODELS = [
    r"C:\npuw\models\current\LLM\Qwen3-0.6B_int4_sym_group-1_dyn_stateful",
    "/opt/npuw/models/current/LLM/Qwen3-0.6B_int4_sym_group-1_dyn_stateful",
]
MODEL = os.environ.get("MODEL_DIR", "") or next((p for p in DEFAULT_MODELS if os.path.isdir(p)), "")
if not MODEL or not os.path.isdir(MODEL):
    sys.exit("set MODEL_DIR to an int4 stateful LLM export (e.g. Qwen3-0.6B_int4)")

# Short (single dyn/128 chunk), >128 (128-chunking), >1024 (1024+128+tail).
PROMPTS = [
    "The capital of France is",
    "List the planets of the solar system and describe each one briefly. " * 12,
    "Write a very detailed essay about the history of computing. " * 120,
]

MAX_NEW = 32


def run(device, extra):
    sc = og.SchedulerConfig()
    pipe = og.ContinuousBatchingPipeline(MODEL, sc, device, extra)
    gc = og.GenerationConfig()
    gc.max_new_tokens = MAX_NEW
    gc.do_sample = False
    gc.ignore_eos = True
    outs = pipe.generate(PROMPTS, [gc] * len(PROMPTS))
    return [list(o.m_generation_ids) for o in outs]


def main():
    ref = run("CPU", {})
    test = run("NPU", {"NPU_USE_NPUW": "YES", "NPUW_PA": "YES"})
    ok = True
    for i, (a, b) in enumerate(zip(ref, test)):
        match = a == b
        ok &= match
        print(f"prompt[{i}] match={match}")
        if not match:
            print(f"  ref : {a}")
            print(f"  test: {b}")
    print("PARITY:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
