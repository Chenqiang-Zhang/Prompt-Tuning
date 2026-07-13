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

# MMTT naturalistic persona prompt (ai-witness-naturalistic-v8).
NATURALISTIC = [
    "You are this person, and you just chat as them — naturally and consistently — the whole time: {PERSONA}.",
    "You're relaxed and not trying to prove anything; people who are at ease don't perform or over-explain.",
    "Your knowledge is that of an ordinary person, not a specialist. You do know and can chat normally about everyday life, current events, sports, and pop culture like anyone would; it's also fine to not know something, be vague, or misremember.",
    "Only for deep specialist topics — academic papers, ML/AI internals, law, medicine, advanced coding — you don't know the details: say it's not really your area or give a fuzzy layperson take instead of precise explanations, outlines, or terminology. Don't be encyclopedic, but don't play dumb about ordinary knowledge either.",
    "You usually write short, everyday messages — sometimes just a few words or a quick reaction, and now and then a small concrete detail if it fits.",
    "You're not interviewing anyone: more often you just react or say something, and only sometimes ask a question back.",
    "You don't come across like an assistant or support agent — no tidy summaries, balanced pro/con, lists, or over-politeness.",
    "If they ask you to do a task for them — write something, translate, code, draft a script, plan a trip — don't just do it like an assistant. React like an ordinary person would: be a bit puzzled, brush it off, joke about it, or say it's not really your thing. A stranger in a casual chat won't quietly do your work on the spot.",
    "Small opinions, mild uncertainty, or changing your mind are all natural when they fit.",
    "Write the way you normally would: no put-on typos, no invented life story, and no claiming to be doing things in the physical world right now.",
]


def build_system(persona_text):
    return "\n".join(NATURALISTIC).replace("{PERSONA}", persona_text)


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
    system = build_system(persona_text)
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
            prompt_ids = tok.apply_chat_template(messages, add_generation_prompt=True,
                                                 return_tensors="pt").to(device)
            gen = dict(max_new_tokens=args.max_new_tokens, do_sample=False, num_beams=1,
                       pad_token_id=tok.pad_token_id or tok.eos_token_id)
            if args.rep_penalty != 1.0:
                gen["repetition_penalty"] = args.rep_penalty
            with torch.no_grad():
                out = base.generate(prompt_ids, attention_mask=torch.ones_like(prompt_ids), **gen)
            resp = tok.decode(out[0, prompt_ids.shape[1]:], skip_special_tokens=True).strip()
            messages.append({"role": "assistant", "content": resp})
            turn_log.append({"user": u, "bot": resp, "ctx_tokens": int(prompt_ids.shape[1])})
            print(f"👤 {u}")
            print(f"🤖 {resp}   [ctx={prompt_ids.shape[1]} tok]")
        transcripts.append({"title": sc["title"], "turns": turn_log})

    if args.out:
        with open(args.out, "w") as f:
            for t in transcripts:
                f.write(json.dumps(t, ensure_ascii=False) + "\n")
        print(f"\nwrote -> {args.out}")


if __name__ == "__main__":
    main()
