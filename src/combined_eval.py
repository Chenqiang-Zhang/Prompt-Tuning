"""
Variant A: explicit prompt + soft prompt, both active.

Runs an *Instruct* model through its native chat template (system = the MMTT
naturalistic persona prompt) AND prepends the trained persona soft-prompt
embeddings in front of the chat-template embeddings. The soft prompt was trained
on the *base* model, so this is an out-of-distribution stacking test: does the
learned persona vector add anything on top of an explicit instruction prompt?

Same three scenarios as chat_eval / prompt_baseline for direct comparison.
"""
import argparse, json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from train import pick_device
from soft_prompt import SoftPromptDialogue
from chat_eval import SCENARIOS
from prompt_baseline import build_system


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--prompt", required=True, help="trained soft prompt .pt (from base model)")
    ap.add_argument("--persona_file", required=True)
    ap.add_argument("--prompt_len", type=int, default=200)
    ap.add_argument("--reply_lang", default="Japanese")
    ap.add_argument("--max_new_tokens", type=int, default=60)
    ap.add_argument("--rep_penalty", type=float, default=1.0)
    ap.add_argument("--no_repeat_ngram", type=int, default=0)
    ap.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16"])
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out")
    args = ap.parse_args()

    device = pick_device(args.device)
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(args.model,
                                                dtype=getattr(torch, args.dtype)).to(device)
    model = SoftPromptDialogue(base, tok, prompt_len=args.prompt_len).to(device)
    model.load_prompt(args.prompt, map_location=device)
    model.base.eval()

    lines = [l.strip() for l in open(args.persona_file) if l.strip()]
    system = build_system("\n".join(lines), args.reply_lang)

    transcripts = []
    for sc in SCENARIOS:
        print(f"\n{'='*66}\n■ {sc['title']}\n{'='*66}")
        messages = [{"role": "system", "content": system}]
        turn_log = []
        for u in sc["turns"]:
            messages.append({"role": "user", "content": u})
            enc = tok.apply_chat_template(messages, add_generation_prompt=True,
                                          return_tensors="pt", return_dict=True).to(device)
            gen = dict(max_new_tokens=args.max_new_tokens, do_sample=False, num_beams=1,
                       pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
            if args.rep_penalty != 1.0:
                gen["repetition_penalty"] = args.rep_penalty
            if args.no_repeat_ngram > 0:
                gen["no_repeat_ngram_size"] = args.no_repeat_ngram
            # SoftPromptDialogue.generate prepends the soft-prompt embeddings and,
            # because it uses inputs_embeds, returns ONLY the newly generated tokens.
            out = model.generate(input_ids=enc["input_ids"],
                                 attention_mask=enc["attention_mask"], **gen)
            resp = tok.decode(out[0], skip_special_tokens=True).strip()
            messages.append({"role": "assistant", "content": resp})
            turn_log.append({"user": u, "bot": resp, "ctx_tokens": int(enc["input_ids"].shape[1])})
            print(f"👤 {u}")
            print(f"🤖 {resp}   [ctx={enc['input_ids'].shape[1]}+{args.prompt_len} soft]")
        transcripts.append({"title": sc["title"], "turns": turn_log})

    if args.out:
        with open(args.out, "w") as f:
            for t in transcripts:
                f.write(json.dumps(t, ensure_ascii=False) + "\n")
        print(f"\nwrote -> {args.out}")


if __name__ == "__main__":
    main()
