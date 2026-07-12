"""
Generate responses with a trained soft prompt (greedy decoding, per the paper).

Reads an eval jsonl (utterance/response rows), generates a response for each
utterance conditioned on the persona soft prompt, and writes a jsonl with the
generated text. Also usable interactively with --interactive.
"""
import argparse, json, os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from soft_prompt import SoftPromptDialogue
from train import SEP, pick_device


def build_model(args, device):
    dtype = getattr(torch, args.dtype)
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype).to(device)
    model = SoftPromptDialogue(base, tok, prompt_len=args.prompt_len).to(device)
    model.load_prompt(args.prompt, map_location=device)
    model.base.eval()
    return model, tok


def respond(model, tok, device, utterance, max_new_tokens):
    ids = tok(utterance, add_special_tokens=False)["input_ids"]
    ids = ids + tok(SEP, add_special_tokens=False)["input_ids"]
    input_ids = torch.tensor([ids], device=device)
    attn = torch.ones_like(input_ids)
    out = model.generate(input_ids=input_ids, attention_mask=attn,
                         max_new_tokens=max_new_tokens, do_sample=False,
                         num_beams=1, pad_token_id=tok.pad_token_id,
                         eos_token_id=tok.eos_token_id)
    text = tok.decode(out[0], skip_special_tokens=True)
    return text.split("\n")[0].strip()  # first line = the response turn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--prompt", required=True, help="trained soft prompt .pt")
    ap.add_argument("--prompt_len", type=int, default=200)
    ap.add_argument("--eval_file", help="jsonl with utterance/response rows")
    ap.add_argument("--out", help="jsonl to write generations to")
    ap.add_argument("--max_new_tokens", type=int, default=40)
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    ap.add_argument("--device", default="auto")
    ap.add_argument("--interactive", action="store_true")
    args = ap.parse_args()

    device = pick_device(args.device)
    model, tok = build_model(args, device)

    if args.interactive:
        print("Interactive mode. Type an utterance (empty line to quit).")
        while True:
            try:
                u = input("YOU> ").strip()
            except EOFError:
                break
            if not u:
                break
            print("BOT>", respond(model, tok, device, u, args.max_new_tokens))
        return

    rows = [json.loads(l) for l in open(args.eval_file)]
    outs = []
    for i, r in enumerate(rows):
        gen = respond(model, tok, device, r["utterance"], args.max_new_tokens)
        outs.append({**r, "generated": gen})
        if i < 8:
            print(f"[{r.get('type','?')}] U: {r['utterance']}\n    G: {gen}")
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            for o in outs:
                f.write(json.dumps(o, ensure_ascii=False) + "\n")
        print(f"\nwrote {len(outs)} generations -> {args.out}")


if __name__ == "__main__":
    main()
