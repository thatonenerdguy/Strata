#!/usr/bin/env python3
"""glm53_generate.py - prompt in, text out, for the first GLM-5.3-Flash runner (build/glm_generate).

Tokenizes the prompt with Strata's tokenizer (tools/strata_tokenizer.py, the GGUF's own vocabulary and merges, the glm4
pre-tokenizer), runs glm_generate on the ids, and decodes the ids it streams back as they arrive.

    python3 tools/glm53_generate.py --model <first shard>.gguf --prompt "The capital of France is" [--n 32]
    python3 tools/glm53_generate.py --model <first shard>.gguf --chat --prompt "Write a haiku about rain."

--chat wraps the prompt in GLM's chat format ([gMASK]<sop><|user|>...<|assistant|>), an approximation of the GGUF's
chat template (--show-template prints the real one); without it the prompt is plain text to continue.
"""
import argparse
import pathlib
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from strata_tokenizer import Tokenizer  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="the GGUF's first shard")
    ap.add_argument("--prompt", default="")
    ap.add_argument("--chat", action="store_true", help="wrap the prompt in GLM's chat format")
    ap.add_argument("--no-think", action="store_true", help="with --chat: an empty <think></think> to skip the reasoning")
    ap.add_argument("--n", type=int, default=64, help="tokens to generate")
    ap.add_argument("--temp", type=float, default=0.0)
    ap.add_argument("--ctx", type=int, default=4096)
    ap.add_argument("--expert-cache-gb", type=float, default=0.0)
    ap.add_argument("--cpu-experts", type=int, default=1, help="1: experts not in the VRAM cache run on the CPU")
    ap.add_argument("--threads", type=int, default=0, help="CPU threads for those experts (0: one per physical core)")
    ap.add_argument("--profile", default="",
                    help="expert profile: which experts start in VRAM, loaded if it exists and updated after every run "
                         "with what the run routed (default: strata-expert-profile.bin beside the model)")
    ap.add_argument("--no-profile", action="store_true", help="neither load nor save a profile")
    ap.add_argument("--prompt-chunk", type=int, default=0, help="prompt tokens a step (0: the tool's default, up to 8192 with streaming, as large as the lent cache slots hold; 1: one at a time)")
    ap.add_argument("--stream-min", type=int, default=-1, help="steps of this many tokens or more stream their experts to the GPU "
                    "(-1: the tool's default, 64; 0: never, the CPU takes them)")
    ap.add_argument("--stream-threads", type=int, default=0, help="copier threads for the streamed experts (0: the tool's default, 8)")
    ap.add_argument("--pin-experts", type=int, default=-1, help="page-lock the expert tensors so streamed experts go up by "
                    "direct DMA (-1: the tool's default, on; 0: off)")
    ap.add_argument("--pin-reserve-gb", type=float, default=-1, help="RAM to keep unpinned (-1: the tool's default, 16)")
    ap.add_argument("--gemm-min-rows", type=int, default=-1, help="prompt expert groups averaging this many rows or more run as "
                    "FP16 GEMMs (-1: the tool's default, 0: never)")
    ap.add_argument("--check-prompt", action="store_true", help="run the prompt batched and one at a time, compare the logits")
    ap.add_argument("--dump-routing", default="", help="write a routing trace for tools/make_profile.py --n-layer 45 --n-expert 288")
    ap.add_argument("--bin", default=str(HERE.parent / "build" / "glm_generate"))
    ap.add_argument("--show-template", action="store_true")
    ap.add_argument("--ids-only", action="store_true", help="print the prompt's ids and stop")
    a = ap.parse_args()

    tk = Tokenizer.from_gguf(a.model)
    if a.show_template:
        from gguf_reader import GGUFFile
        print(GGUFFile(pathlib.Path(a.model)).metadata.get("tokenizer.chat_template", "(none)"))
        return 0
    text = a.prompt
    if a.chat:
        text = "[gMASK]<sop><|user|>\n" + a.prompt + "<|assistant|>" + ("\n<think></think>\n" if a.no_think else "")
    ids = tk.encode(text, parse_special=True)
    if a.ids_only:
        print(",".join(map(str, ids)))
        return 0
    stop = set()
    for k, v in tk.special_ids.items():
        if k.endswith("eos_token_id") or k.endswith("eot_token_id") or k.endswith("eom_token_id"):
            stop.add(int(v))
    for s in ("<|endoftext|>", "<|user|>", "<|observation|>"):
        if s in tk.ids:
            stop.add(tk.ids[s])
    cmd = [a.bin, "--model", a.model, "--ids", ",".join(map(str, ids)), "--n", str(a.n), "--ctx", str(a.ctx),
           "--temp", str(a.temp), "--stop", ",".join(map(str, sorted(stop)))]
    if a.expert_cache_gb > 0:
        cmd += ["--expert-cache-gb", str(a.expert_cache_gb)]
    cmd += ["--cpu-experts", str(a.cpu_experts)]
    if not a.no_profile:
        prof = pathlib.Path(a.profile) if a.profile else pathlib.Path(a.model).resolve().parent / "strata-expert-profile.bin"
        if prof.exists():
            cmd += ["--expert-profile", str(prof)]
        cmd += ["--expert-profile-save", str(prof)]
    if a.prompt_chunk > 0:
        cmd += ["--prompt-chunk", str(a.prompt_chunk)]
    if a.stream_min >= 0:
        cmd += ["--stream-min", str(a.stream_min)]
    if a.gemm_min_rows >= 0:
        cmd += ["--gemm-min-rows", str(a.gemm_min_rows)]
    if a.stream_threads > 0:
        cmd += ["--stream-threads", str(a.stream_threads)]
    if a.pin_experts >= 0:
        cmd += ["--pin-experts", str(a.pin_experts)]
    if a.pin_reserve_gb >= 0:
        cmd += ["--pin-reserve-gb", str(a.pin_reserve_gb)]
    if a.check_prompt:
        cmd += ["--check-prompt"]
    if a.dump_routing:
        cmd += ["--dump-routing", a.dump_routing]
    if a.threads > 0:
        cmd += ["--threads", str(a.threads)]
    print("prompt: %d tokens" % len(ids), file=sys.stderr)
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True)
    out, shown, inside = [], "", False
    sys.stdout.write(a.prompt if not a.chat else "")
    for line in p.stdout:
        line = line.rstrip("\n")
        if line == "BEGIN_IDS":
            inside = True
            continue
        if line == "END_IDS":
            inside = False
            sys.stdout.write("\n")
            continue
        if not inside:
            print(line, file=sys.stderr)
            continue
        out.append(int(line))
        txt = tk.decode(out)
        sys.stdout.write(txt[len(shown):])
        sys.stdout.flush()
        shown = txt
    return p.wait()


if __name__ == "__main__":
    sys.exit(main())
