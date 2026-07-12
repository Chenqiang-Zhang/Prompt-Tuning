"""
Prepare RealPersonaChat into the paper's training setup.

Paper (Kasahara et al., 2022) recipe, Japanese track:
  - Split multi-turn dialogues into "dialogue pairs" = (previous utterance, response).
  - Aggregate pairs by the *responder's* persona.
  - Keep the top-N personas with the most pairs (paper uses 3).
  - Split each persona's pairs 9:1 into train / persona-eval.
  - Mix in short "general" pairs unrelated to the persona so the model also
    learns to produce non-persona responses (paper mixes DailyDialog / JEmpatheticDialogues).

We substitute JPersonaChat (license-gated) with the freely available
RealPersonaChat (nu-dialogue/real-persona-chat), which has the same structure:
per-speaker persona sentences + 2-person multi-turn dialogues.
"""
import argparse, json, os, glob, random


def load_dialogue_pairs(root):
    """Return (pairs_by_persona, interlocutor_personas).

    pairs_by_persona[interlocutor_id] = list of {"utterance", "response"}
    where `response` was said by that interlocutor immediately after `utterance`.
    """
    interlocutors = json.load(open(os.path.join(root, "interlocutors.json")))
    personas = {k: v["persona"] for k, v in interlocutors.items()}

    pairs_by_persona = {}
    for f in sorted(glob.glob(os.path.join(root, "dialogues", "*.json"))):
        d = json.load(open(f))
        utts = d["utterances"]
        for i in range(1, len(utts)):
            prev, cur = utts[i - 1], utts[i]
            if prev["interlocutor_id"] == cur["interlocutor_id"]:
                continue  # same speaker in a row -> not a cross-turn pair
            u = prev["text"].strip()
            r = cur["text"].strip()
            if not u or not r:
                continue
            pairs_by_persona.setdefault(cur["interlocutor_id"], []).append(
                {"utterance": u, "response": r}
            )
    return pairs_by_persona, personas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root",
                    default="data/real-persona-chat-1.0.0/real_persona_chat")
    ap.add_argument("--out_dir", default="data/processed")
    ap.add_argument("--top_personas", type=int, default=3)
    ap.add_argument("--eval_ratio", type=float, default=0.1)
    ap.add_argument("--general_ratio", type=float, default=1.0,
                    help="general pairs added per persona pair (paper En: 1:1)")
    ap.add_argument("--max_chars", type=int, default=50,
                    help="max chars for a *general* (short) pair, per paper")
    ap.add_argument("--max_pairs_per_persona", type=int, default=0,
                    help="0 = no cap; else cap persona pairs before splitting")
    ap.add_argument("--general_eval_size", type=int, default=150)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    random.seed(args.seed)

    pairs_by_persona, personas = load_dialogue_pairs(args.data_root)
    ranked = sorted(pairs_by_persona.items(), key=lambda kv: len(kv[1]), reverse=True)
    selected = [pid for pid, _ in ranked[: args.top_personas]]

    print("Top personas by #pairs:")
    for pid, prs in ranked[: args.top_personas]:
        print(f"  {pid}: {len(prs)} pairs")

    # General pool: short pairs whose responder is NOT one of the selected personas.
    general_pool = []
    for pid, prs in pairs_by_persona.items():
        if pid in selected:
            continue
        for p in prs:
            if len(p["utterance"]) <= args.max_chars and len(p["response"]) <= args.max_chars:
                general_pool.append(p)
    random.shuffle(general_pool)
    print(f"General (short) pool: {len(general_pool)} pairs")

    os.makedirs(args.out_dir, exist_ok=True)
    # Shared general eval set (for the "general eval" distinct metric).
    general_eval = [dict(p, type="general") for p in general_pool[: args.general_eval_size]]
    general_train_pool = general_pool[args.general_eval_size:]
    _dump(os.path.join(args.out_dir, "eval_general.jsonl"), general_eval)

    gpi = 0  # cursor into general_train_pool
    for pid in selected:
        prs = pairs_by_persona[pid][:]
        random.shuffle(prs)
        if args.max_pairs_per_persona:
            prs = prs[: args.max_pairs_per_persona]
        n_eval = max(1, int(len(prs) * args.eval_ratio))
        eval_p = [dict(p, type="persona") for p in prs[:n_eval]]
        train_p = [dict(p, type="persona") for p in prs[n_eval:]]

        n_gen = int(len(train_p) * args.general_ratio)
        gen = [dict(p, type="general") for p in general_train_pool[gpi: gpi + n_gen]]
        gpi += n_gen
        train = train_p + gen
        random.shuffle(train)

        pdir = os.path.join(args.out_dir, f"persona_{pid}")
        os.makedirs(pdir, exist_ok=True)
        _dump(os.path.join(pdir, "train.jsonl"), train)
        _dump(os.path.join(pdir, "eval_persona.jsonl"), eval_p)
        with open(os.path.join(pdir, "persona.txt"), "w") as f:
            f.write("\n".join(personas[pid]))
        print(f"[{pid}] train={len(train)} (persona {len(train_p)} + general {len(gen)}), "
              f"eval_persona={len(eval_p)}")

    print(f"\nDone. Output in {args.out_dir}")


def _dump(path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
