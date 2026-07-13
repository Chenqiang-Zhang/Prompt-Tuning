"""
Prompt-tuning trainer.

Trains ONLY the soft-prompt matrix (persona info tokens) on one persona's
dialogue pairs. Everything else in the LM is frozen. Loss is computed on the
response tokens only. Optimizer/LR follow the paper (Adam, lr=1e-3).
"""
import argparse, json, os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from soft_prompt import SoftPromptDialogue

SEP = "\n"  # lightweight utterance/response separator (Qwen has no turn token)


def pick_device(name):
    if name != "auto":
        return torch.device(name)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class PairDataset(Dataset):
    def __init__(self, path, tok, max_len):
        self.rows = [json.loads(l) for l in open(path)]
        self.tok = tok
        self.max_len = max_len
        self.sep_ids = tok(SEP, add_special_tokens=False)["input_ids"]
        self.eos = tok.eos_token_id

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        u = self.tok(r["utterance"], add_special_tokens=False)["input_ids"]
        a = self.tok(r["response"], add_special_tokens=False)["input_ids"]
        ctx = u + self.sep_ids                      # utterance + separator (no loss)
        ans = a + [self.eos]                        # response + eos (loss here)
        input_ids = (ctx + ans)[: self.max_len]
        labels = ([-100] * len(ctx) + ans)[: self.max_len]
        return {"input_ids": input_ids, "labels": labels}


def make_collate(pad_id):
    def collate(batch):
        L = max(len(b["input_ids"]) for b in batch)
        ids, lab, att = [], [], []
        for b in batch:
            n = len(b["input_ids"])
            pad = L - n
            ids.append(b["input_ids"] + [pad_id] * pad)
            lab.append(b["labels"] + [-100] * pad)
            att.append([1] * n + [0] * pad)
        return (torch.tensor(ids), torch.tensor(att), torch.tensor(lab))
    return collate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--persona_dir", required=True)
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--out", required=True, help="path to save soft prompt (.pt)")
    ap.add_argument("--prompt_len", type=int, default=200)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--max_len", type=int, default=128)
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    ap.add_argument("--device", default="auto")
    ap.add_argument("--no_persona_init", action="store_true",
                    help="random init instead of persona-sentence embeddings")
    ap.add_argument("--naturalistic_init", action="store_true",
                    help="initialize the soft prompt from the full MMTT naturalistic "
                         "system prompt (+persona), i.e. compile the explicit prompt in")
    ap.add_argument("--reply_lang", default="",
                    help="language lock appended to the naturalistic init text")
    args = ap.parse_args()

    device = pick_device(args.device)
    dtype = getattr(torch, args.dtype)
    print(f"device={device} dtype={dtype} model={args.model}")

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype).to(device)

    persona_text = open(os.path.join(args.persona_dir, "persona.txt")).read().strip()
    init_text = None
    if args.naturalistic_init:
        from persona_prompt import build_system
        init_text = build_system(persona_text, args.reply_lang)
    elif not args.no_persona_init:
        init_text = persona_text

    # Auto-size the soft prompt to fully hold the init text (no truncation) when
    # --prompt_len 0; this "compiles" the whole explicit prompt into the prompt.
    prompt_len = args.prompt_len
    if prompt_len == 0:
        n = len(tok(init_text, add_special_tokens=False)["input_ids"]) if init_text else 200
        prompt_len = n
        print(f"auto prompt_len = {prompt_len} (from init text)")

    model = SoftPromptDialogue(base, tok, prompt_len=prompt_len,
                               init_text=init_text).to(device)

    n_train = sum(p.numel() for p in model.trainable_parameters())
    n_total = sum(p.numel() for p in model.parameters())
    print(f"trainable params: {n_train:,} / {n_total:,} "
          f"({100*n_train/n_total:.4f}%)")

    ds = PairDataset(os.path.join(args.persona_dir, "train.jsonl"), tok, args.max_len)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                    collate_fn=make_collate(tok.pad_token_id))
    print(f"train examples: {len(ds)}")

    opt = torch.optim.Adam(model.trainable_parameters(), lr=args.lr)

    model.base.eval()  # frozen; disables dropout in the LM
    for ep in range(1, args.epochs + 1):
        tot, nb = 0.0, 0
        for input_ids, attn, labels in dl:
            input_ids, attn, labels = (input_ids.to(device), attn.to(device),
                                       labels.to(device))
            out = model(input_ids=input_ids, attention_mask=attn, labels=labels)
            opt.zero_grad()
            out.loss.backward()
            opt.step()
            tot += out.loss.item(); nb += 1
        print(f"epoch {ep:3d}  loss {tot/nb:.4f}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    model.save_prompt(args.out)
    print(f"saved soft prompt -> {args.out}")


if __name__ == "__main__":
    main()
