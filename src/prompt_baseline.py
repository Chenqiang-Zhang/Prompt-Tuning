"""
Explicit-prompt baseline (vs. prompt-tuning).

Instead of a trained soft prompt on a frozen *base* LM, this uses an
instruction-following *Instruct* model with a hand-written system prompt that
tells it to role-play the persona — the "manual prompt design" line the paper
contrasts prompt-tuning against.

The system prompt is MMTT's "naturalistic" persona prompt; {PERSONA} is filled
with the same persona sentences used to train the soft prompt. Multi-turn uses
the model's native chat template, so dialogue history is handled properly.
"""
import argparse, json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from train import pick_device
from chat_eval import SCENARIOS
from persona_prompt import NATURALISTIC, build_system  # noqa: F401


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--persona_file", required=True)
    ap.add_argument("--max_new_tokens", type=int, default=60)
    ap.add_argument("--rep_penalty", type=float, default=1.0)
    ap.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16"])
    ap.add_argument("--device", default="auto")
    ap.add_argument("--persona_oneline", action="store_true",
                    help="join persona sentences with spaces instead of newlines")
    ap.add_argument("--reply_lang", default="",
                    help="e.g. 'Japanese' — appends a language-lock instruction")
    ap.add_argument("--out")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    device = pick_device(args.device)
    tok = AutoTokenizer.from_pretrained(args.model)
    base = AutoModelForCausalLM.from_pretrained(args.model,
                                                dtype=getattr(torch, args.dtype)).to(device)
    base.eval()

    lines = [l.strip() for l in open(args.persona_file) if l.strip()]
    persona_text = (" ".join(lines) if args.persona_oneline else "\n".join(lines))
    system = build_system(persona_text, args.reply_lang)
    if args.debug:
        print("┌─ SYSTEM PROMPT ─────────────────────────────\n" + system +
              "\n└─────────────────────────────────────────────\n")

    transcripts = []
    for sc in SCENARIOS:
        print(f"\n{'='*66}\n■ {sc['title']}\n{'='*66}")
        messages = [{"role": "system", "content": system}]
        turn_log = []
        for u in sc["turns"]:
            messages.append({"role": "user", "content": u})
            enc = tok.apply_chat_template(messages, add_generation_prompt=True,
                                          return_tensors="pt", return_dict=True).to(device)
            in_len = enc["input_ids"].shape[1]
            gen = dict(max_new_tokens=args.max_new_tokens, do_sample=False, num_beams=1,
                       pad_token_id=tok.pad_token_id or tok.eos_token_id)
            if args.rep_penalty != 1.0:
                gen["repetition_penalty"] = args.rep_penalty
            with torch.no_grad():
                out = base.generate(**enc, **gen)
            resp = tok.decode(out[0, in_len:], skip_special_tokens=True).strip()
            messages.append({"role": "assistant", "content": resp})
            turn_log.append({"user": u, "bot": resp, "ctx_tokens": int(in_len)})
            print(f"👤 {u}")
            print(f"🤖 {resp}   [ctx={in_len} tok]")
        transcripts.append({"title": sc["title"], "turns": turn_log})

    if args.out:
        with open(args.out, "w") as f:
            for t in transcripts:
                f.write(json.dumps(t, ensure_ascii=False) + "\n")
        print(f"\nwrote -> {args.out}")


if __name__ == "__main__":
    main()
