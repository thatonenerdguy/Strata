"""tools/glm53_serve_setup.py - the run config that lets Strata's server (serve/server.py) run GLM-5.3-Flash.

    python tools/glm53_serve_setup.py --model <any shard of the GGUF> [--out DIR] [--ctx 16384] [--bin build/glm_generate]
    python -m serve.server --engine strata --config DIR/strata-glm53.json --port 8080

What it writes into DIR (default: a "strata-glm53" folder beside the model):
  tokenizer/            the model's tokenizer and chat template, read out of the GGUF by strata_tokenizer.extract (the
                        server never opens the weight shards for them)
  strata-glm53.json     the run config: the engine is build/glm_generate in its --serve mode (the line protocol the
                        server speaks to every Strata engine), with the model's stop tokens and the expert profile beside
                        the model (read at start when it exists, written back at every stop, as the wrapper does)

The server then gives the model its OpenAI / Anthropic endpoints and the chat page as for the other models, and reads GLM's
own tool calls (`<arg_key>` / `<arg_value>`, chosen from the template: serve/frontend.py GlmOutputParser).  Not yet: MTP
drafting, several requests at once.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

import strata_tokenizer as ST  # noqa: E402


def stop_ids(tk) -> list[int]:
    """The ids that end a reply: the file's eos / eot / eom, and the turns GLM opens after its own (<|user|>, <|observation|>)."""
    out = set()
    for k, v in tk.special_ids.items():
        if k.endswith(("eos_token_id", "eot_token_id", "eom_token_id")):
            out.add(int(v))
    for s in ("<|endoftext|>", "<|user|>", "<|observation|>"):
        if s in tk.ids:
            out.add(int(tk.ids[s]))
    return sorted(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", required=True, help="any shard of the GLM-5.3-Flash GGUF")
    ap.add_argument("--out", default="", help="folder for the config and the tokenizer (default: strata-glm53 beside the model)")
    ap.add_argument("--ctx", type=int, default=16384, help="context in tokens (default 16384)")
    ap.add_argument("--bin", default=str(ROOT / "build" / "glm_generate"), help="the glm_generate program")
    ap.add_argument("--name", default="GLM-5.3-Flash", help="the model name the server lists")
    ap.add_argument("--threads", type=int, default=0, help="CPU expert threads (0: the pool's own choice)")
    a = ap.parse_args()

    model = pathlib.Path(a.model).expanduser().resolve()
    if not model.exists():
        raise SystemExit(f"{model}: no such file")
    out = pathlib.Path(a.out).expanduser().resolve() if a.out else model.parent / "strata-glm53"
    out.mkdir(parents=True, exist_ok=True)
    exe = pathlib.Path(a.bin).expanduser().resolve()
    if not exe.exists():
        print(f"note: {exe} does not exist yet (cmake --build build --target glm_generate)")

    cfg_tok = ST.extract(str(model), str(out))
    tk = ST.Tokenizer.from_gguf(str(model))
    if not (out / "tokenizer" / "chat_template.jinja").exists():
        raise SystemExit("the GGUF has no chat template (tokenizer.chat_template): the server needs one")
    stops = stop_ids(tk)
    profile = model.parent / "strata-expert-profile.bin"   # the same file tools/glm53_generate.py keeps
    args = ["--model", str(model), "--max-context", str(a.ctx), "--stop", ",".join(map(str, stops)),
            "--expert-profile", str(profile), "--expert-profile-save", str(profile)]
    if a.threads > 0:
        args += ["--threads", str(a.threads)]
    cfg = {
        "exe": str(exe),
        "args": args,
        "cwd": str(ROOT),
        "tokenizer": str(out / "tokenizer"),
        "model_name": a.name,
        "arch": "glm5",
        "log": str(out / "engine.log"),
        "mtp": False,
        "fit_max_tokens": True,
    }
    path = out / "strata-glm53.json"
    path.write_text(json.dumps(cfg, indent=1) + "\n", encoding="utf-8")
    print(f"tokenizer: {out / 'tokenizer'} (vocab {cfg_tok['vocab_size']}, pre {cfg_tok['pre']})")
    print(f"stop tokens: {stops}")
    print(f"wrote {path}")
    print(f"start it:  python -m serve.server --engine strata --config {path} --port 8080")
    return 0


if __name__ == "__main__":
    sys.exit(main())
