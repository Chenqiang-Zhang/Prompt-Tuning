"""
Multi-turn / long-context validation for a trained persona soft prompt.

The paper trains on single-turn (utterance, response) pairs. This script probes
how the model behaves when the *whole growing dialogue history* is fed as
context — an out-of-distribution stress test for coherence and persona
consistency over several turns.

Context per turn = soft_prompt ++ embed( turn_1 \n turn_2 \n ... \n turn_k \n ),
i.e. every prior turn (user and bot) joined by the same separator used in
training, then the model greedily generates the next bot turn.
"""
import argparse, json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from soft_prompt import SoftPromptDialogue
from train import SEP, pick_device

# Scripted user turns. Chosen to probe persona CP (books/mystery, watermelon,
# swimming, Fukuoka, chemistry) across a lengthening context.
SCENARIOS = [
    {"title": "读书话题（核心人设）",
     "turns": [
         "こんにちは！休日はいつも何をしていますか？",
         "いいですね。どんなジャンルの本をよく読むんですか？",
         "おすすめのミステリー作家はいますか？",
         "図書館にはよく行かれるんですか？",
         "今度その本を読んでみますね！ほかに趣味はありますか？",
     ]},
    {"title": "夏・食べ物・出身地（多个人设点）",
     "turns": [
         "もう夏本番ですね。暑いのは平気ですか？",
         "夏によく食べるものってありますか？",
         "運動はされますか？",
         "ちなみにどちらにお住まいなんですか？",
         "福岡いいですね！おすすめの場所はありますか？",
     ]},
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--prompt_len", type=int, default=200)
    ap.add_argument("--max_new_tokens", type=int, default=40)
    ap.add_argument("--max_ctx_tokens", type=int, default=512)
    ap.add_argument("--rep_penalty", type=float, default=1.0,
                    help=">1.0 penalizes repetition (mitigates greedy loops)")
    ap.add_argument("--no_repeat_ngram", type=int, default=0,
                    help=">0 forbids repeating n-grams of this size")
    ap.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16"])
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", help="optional jsonl transcript")
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

    sep_ids = tok(SEP, add_special_tokens=False)["input_ids"]
    transcripts = []
    for sc in SCENARIOS:
        print(f"\n{'='*66}\n■ {sc['title']}\n{'='*66}")
        history = []            # list of "role: text" strings, but we feed text only
        turn_log = []
        for u in sc["turns"]:
            history.append(u)
            # build context = all turns so far, joined by SEP, then a trailing SEP
            ids = []
            for i, t in enumerate(history):
                if i:
                    ids += sep_ids
                ids += tok(t, add_special_tokens=False)["input_ids"]
            ids += sep_ids
            ids = ids[-args.max_ctx_tokens:]
            input_ids = torch.tensor([ids], device=device)
            attn = torch.ones_like(input_ids)
            gen = dict(max_new_tokens=args.max_new_tokens, do_sample=False, num_beams=1,
                       pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
            if args.rep_penalty != 1.0:
                gen["repetition_penalty"] = args.rep_penalty
            if args.no_repeat_ngram > 0:
                gen["no_repeat_ngram_size"] = args.no_repeat_ngram
            out = model.generate(input_ids=input_ids, attention_mask=attn, **gen)
            resp = tok.decode(out[0], skip_special_tokens=True).split("\n")[0].strip()
            history.append(resp)
            turn_log.append({"user": u, "bot": resp, "ctx_tokens": len(ids)})
            print(f"👤 {u}")
            print(f"🤖 {resp}   [ctx={len(ids)} tok]")
        transcripts.append({"title": sc["title"], "turns": turn_log})

    if args.out:
        with open(args.out, "w") as f:
            for t in transcripts:
                f.write(json.dumps(t, ensure_ascii=False) + "\n")
        print(f"\nwrote transcript -> {args.out}")


if __name__ == "__main__":
    main()
