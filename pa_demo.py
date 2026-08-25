#!/usr/bin/env python3
"""PA CB demo -- NPUW PagedAttention front-end, PR 1 (1:1 CPU passthrough).

Parts, runnable individually or together (default: all):

  cb        ContinuousBatchingPipeline on NPU + NPUW_PA, batched requests
  llm       LLMPipeline on NPU + NPUW_PA, routed through the CB/PA pipeline
  parity    greedy token parity, plain CPU vs NPU + NPUW_PA

Examples:

  python pa_demo.py                      # everything
  python pa_demo.py cb                   # one part
  python pa_demo.py cb llm --npuw-log INFO
  python pa_demo.py parity --model C:\\models\\some-other-int4-export
"""
import argparse
import difflib
import os
import sys
import time

# --npuw-log must reach the environment before the plugin compiles anything.
if "--npuw-log" in sys.argv:
    i = sys.argv.index("--npuw-log")
    try:
        os.environ["OPENVINO_NPUW_LOG_LEVEL"] = sys.argv[i + 1]
    except IndexError:
        sys.exit("--npuw-log needs a level: ERROR, WARNING, INFO, VERBOSE")
    del sys.argv[i:i + 2]

import openvino_genai as og

# ── colors ───────────────────────────────────────────────────────────────

if os.name == "nt":
    os.system("")  # enables ANSI in the classic console; no-op elsewhere

USE_COLOR = sys.stdout.isatty() and os.environ.get("NO_COLOR") is None


def c(code, text):
    return f"\033[{code}m{text}\033[0m" if USE_COLOR else str(text)


def bold(t):    return c("1", t)
def cyan(t):    return c("96", t)
def green(t):   return c("92", t)
def red(t):     return c("91", t)
def yellow(t):  return c("93", t)
def dim(t):     return c("2", t)


# ── setup ────────────────────────────────────────────────────────────────

DEFAULT_MODELS = [
    r"C:\npuw\models\current\LLM\Qwen3-0.6B_int4_sym_group-1_dyn_stateful",
    "/opt/npuw/models/current/LLM/Qwen3-0.6B_int4_sym_group-1_dyn_stateful",
]

PA_PROPS = {"NPU_USE_NPUW": "YES", "NPUW_PA": "YES"}

PROMPTS = [
    ("short", "The capital of France is"),
    ("mid", "Explain in two sentences why paged attention helps batched LLM serving."),
    ("long", "List the planets of the solar system and describe each one briefly. " * 12),
]

# Parity stresses the scheduler's chunking: single chunk, >128, >1024.
PARITY_PROMPTS = [
    ("short", "The capital of France is"),
    (">128", "List the planets of the solar system and describe each one briefly. " * 12),
    (">1024", "Write a very detailed essay about the history of computing. " * 120),
]


def gen_config(max_new):
    gc = og.GenerationConfig()
    gc.max_new_tokens = max_new
    gc.do_sample = False
    gc.ignore_eos = True
    return gc


def header(step, total, title):
    print()
    print(bold(cyan(f"[{step}/{total}] {title}")))


def show_prompt(label, prompt):
    shown = prompt if len(prompt) < 64 else prompt[:61] + "..."
    print(f"  {dim(label + ':')} {shown}")


def texts(result):
    # GenerationResult.m_generation_ids carries the decoded text per sequence.
    return result.m_generation_ids[0] if result.m_generation_ids else "(empty)"


# ── parts ────────────────────────────────────────────────────────────────

def part_cb(model, max_new):
    header(*part_cb.pos, "ContinuousBatchingPipeline on NPU + NPUW_PA")
    t0 = time.perf_counter()
    pipe = og.ContinuousBatchingPipeline(model, og.SchedulerConfig(), "NPU", PA_PROPS)
    print(f"  {dim('compile')} {yellow(f'{time.perf_counter() - t0:.1f}s')}")

    gc = gen_config(max_new)
    prompts = [p for _, p in PROMPTS]
    t0 = time.perf_counter()
    results = pipe.generate(prompts, [gc] * len(prompts))
    dt = time.perf_counter() - t0

    for (label, prompt), res in zip(PROMPTS, results):
        print()
        show_prompt(label, prompt)
        print(f"  {green(texts(res).strip())}")

    total = max_new * len(results)  # ignore_eos + fixed budget
    print()
    print(f"  {len(results)} requests batched | {total} tokens | "
          f"{yellow(f'{dt:.1f}s')} | {yellow(f'{total / dt:.1f} tok/s')}")
    return True


def part_llm(model, max_new):
    header(*part_llm.pos, "LLMPipeline on NPU + NPUW_PA (routed through CB/PA)")
    t0 = time.perf_counter()
    pipe = og.LLMPipeline(model, "NPU", **PA_PROPS)
    print(f"  {dim('compile')} {yellow(f'{time.perf_counter() - t0:.1f}s')}")

    label, prompt = PROMPTS[1]
    print()
    show_prompt(label, prompt)
    t0 = time.perf_counter()
    out = pipe.generate(prompt, gen_config(max_new))
    print(f"  {green(str(out).strip())}")
    print()
    print(f"  {yellow(f'{time.perf_counter() - t0:.1f}s')}")
    return True


def part_parity(model, max_new):
    header(*part_parity.pos, "Parity: plain CPU vs NPU + NPUW_PA (greedy)")

    def run(device, extra):
        pipe = og.ContinuousBatchingPipeline(model, og.SchedulerConfig(), device, extra)
        gc = gen_config(max_new)
        prompts = [p for _, p in PARITY_PROMPTS]
        return [texts(r) for r in pipe.generate(prompts, [gc] * len(prompts))]

    print(f"  {dim('reference')} CPU")
    ref = run("CPU", {})
    print(f"  {dim('test')}      NPU + NPUW_PA")
    test = run("NPU", PA_PROPS)

    ok = True
    scores = []
    for (label, _), a, b in zip(PARITY_PROMPTS, ref, test):
        match = a == b
        ok &= match
        score = difflib.SequenceMatcher(None, a, b).ratio()
        scores.append(score)
        shade = green if score >= 0.99 else (yellow if score >= 0.9 else red)
        print(f"  {label:<8} {green('PASS') if match else red('FAIL')}  "
              f"{dim('similarity')} {shade(f'{score * 100:.1f}%')}")
        if not match:
            print(f"    {dim('ref:')}  {a!r}")
            print(f"    {dim('test:')} {b!r}")
    print()
    mean = sum(scores) / len(scores) if scores else 0.0
    print(f"  PARITY: {green(bold('PASS')) if ok else red(bold('FAIL'))}  "
          f"{dim('mean similarity')} {yellow(f'{mean * 100:.1f}%')}")
    return ok


PARTS = {"cb": part_cb, "llm": part_llm, "parity": part_parity}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("parts", nargs="*", choices=[*PARTS, "all"], default=["all"],
                    help="which parts to run (default: all)")
    ap.add_argument("--model", default=None, help="model dir (default: MODEL_DIR env, then the NPU machines' layout)")
    ap.add_argument("--tokens", type=int, default=48, help="new tokens per request (default: 48)")
    args = ap.parse_args()

    model = args.model or os.environ.get("MODEL_DIR", "") \
        or next((p for p in DEFAULT_MODELS if os.path.isdir(p)), "")
    if not model or not os.path.isdir(model):
        sys.exit(red("no model found: pass --model or set MODEL_DIR to an int4 stateful LLM export"))

    selected = list(PARTS) if "all" in args.parts else list(dict.fromkeys(args.parts))
    for n, name in enumerate(selected, 1):
        PARTS[name].pos = (n, len(selected))

    print(bold("PA CB demo") + dim("  NPUW PagedAttention front-end, PR 1"))
    print(f"  {dim('model')}  {model}")
    print(f"  {dim('props')}  " + " ".join(f"{k}={v}" for k, v in PA_PROPS.items()))
    level = os.environ.get("OPENVINO_NPUW_LOG_LEVEL")
    if level:
        print(f"  {dim('npuw log')}  {level}")

    ok = True
    for name in selected:
        ok &= PARTS[name](model, args.tokens)

    print()
    print(green(bold("DONE")) if ok else red(bold("FAILED")))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
