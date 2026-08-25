#!/usr/bin/env python3
"""PA front-end demo, PR 1 (1:1 CPU passthrough).

Two views of the same path:

  1. ContinuousBatchingPipeline on NPU with NPUW_PA=YES -- the pipeline's own
     dynamic PA model deployed through the NPU plugin, executed 1:1 on the
     PA fallback device (CPU at this stage of the series).
  2. LLMPipeline on NPU with NPUW_PA=YES -- the user-facing entry: the GenAI
     routing change sends this through the CB/PA pipeline, so a plain
     LLMPipeline user gets the PA path with one property.

Set MODEL_DIR to an int4 stateful LLM export (the PR's parity evidence used
Qwen3-0.6B int4). Set OPENVINO_NPUW_LOG_LEVEL=INFO or VERBOSE to see the PA
front-end engage.
"""
import os
import sys
import time

import openvino_genai as og

DEFAULT_MODELS = [
    r"C:\npuw\models\current\LLM\Qwen3-0.6B_int4_sym_group-1_dyn_stateful",
    "/opt/npuw/models/current/LLM/Qwen3-0.6B_int4_sym_group-1_dyn_stateful",
]
MODEL = os.environ.get("MODEL_DIR", "") or next((p for p in DEFAULT_MODELS if os.path.isdir(p)), "")
if not MODEL or not os.path.isdir(MODEL):
    sys.exit("set MODEL_DIR to an int4 stateful LLM export (e.g. Qwen3-0.6B_int4)")

PROMPTS = [
    "The capital of France is",
    "Explain in two sentences why paged attention helps batched LLM serving.",
    "List the planets of the solar system and describe each one briefly. " * 12,
]

PA_PROPS = {"NPU_USE_NPUW": "YES", "NPUW_PA": "YES"}
MAX_NEW = 48


def banner(title):
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def gen_config():
    gc = og.GenerationConfig()
    gc.max_new_tokens = MAX_NEW
    gc.do_sample = False
    gc.ignore_eos = True
    return gc


def demo_cb_pipeline():
    banner("1) ContinuousBatchingPipeline on NPU + NPUW_PA (PA device: CPU)")
    t0 = time.perf_counter()
    pipe = og.ContinuousBatchingPipeline(MODEL, og.SchedulerConfig(), "NPU", PA_PROPS)
    print(f"[compile: {time.perf_counter() - t0:.1f}s]")

    gc = gen_config()
    t0 = time.perf_counter()
    results = pipe.generate(PROMPTS, [gc] * len(PROMPTS))
    dt = time.perf_counter() - t0
    # ignore_eos + fixed max_new_tokens: every request generates exactly MAX_NEW.
    total_tokens = MAX_NEW * len(results)
    for i, (prompt, res) in enumerate(zip(PROMPTS, results)):
        shown = prompt if len(prompt) < 70 else prompt[:67] + "..."
        print(f"\n--- request[{i}] prompt: {shown!r}")
        # m_generation_ids carries the decoded text, one entry per sequence.
        print(res.m_generation_ids[0] if res.m_generation_ids else "(empty)")
    print(f"\n[{len(PROMPTS)} requests batched, {total_tokens} tokens, {dt:.1f}s, {total_tokens / dt:.1f} tok/s]")


def demo_llm_pipeline():
    banner("2) LLMPipeline on NPU + NPUW_PA (routed through the CB/PA pipeline)")
    t0 = time.perf_counter()
    pipe = og.LLMPipeline(MODEL, "NPU", **PA_PROPS)
    print(f"[compile: {time.perf_counter() - t0:.1f}s]")

    gc = gen_config()
    prompt = PROMPTS[1]
    print(f"\n--- prompt: {prompt!r}")
    t0 = time.perf_counter()
    out = pipe.generate(prompt, gc)
    dt = time.perf_counter() - t0
    print(out)
    print(f"\n[{dt:.1f}s]")


def main():
    print(f"model:  {MODEL}")
    print(f"props:  {PA_PROPS}")
    print(f"npuw log level: {os.environ.get('OPENVINO_NPUW_LOG_LEVEL', '(default)')}")
    demo_cb_pipeline()
    demo_llm_pipeline()
    print("\nDONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
