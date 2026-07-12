"""
Automatic evaluation: distinct-1 and distinct-2 (Li et al., 2016a).

distinct-N = (# unique N-grams) / (# total N-grams) over all generated responses.
Higher = more diverse / less repetitive. This is the metric in Table 1 / Table 6.

For Japanese we tokenize on characters by default (a reasonable, tokenizer-free
proxy); pass --word to split on whitespace instead.
"""
import argparse, json


def ngrams(tokens, n):
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def distinct(texts, n, char_level=True):
    total, uniq = 0, set()
    for t in texts:
        toks = list(t) if char_level else t.split()
        gs = ngrams(toks, n)
        total += len(gs)
        uniq.update(gs)
    return len(uniq) / total if total else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+", help="generation jsonl file(s)")
    ap.add_argument("--field", default="generated")
    ap.add_argument("--word", action="store_true", help="word-level instead of char-level")
    args = ap.parse_args()

    all_texts = []
    for f in args.files:
        texts = [json.loads(l)[args.field] for l in open(f)]
        all_texts += texts
        d1 = distinct(texts, 1, not args.word)
        d2 = distinct(texts, 2, not args.word)
        print(f"{f}: n={len(texts)}  distinct-1={d1:.3f}  distinct-2={d2:.3f}")
    if len(args.files) > 1:
        d1 = distinct(all_texts, 1, not args.word)
        d2 = distinct(all_texts, 2, not args.word)
        print(f"OVERALL: n={len(all_texts)}  distinct-1={d1:.3f}  distinct-2={d2:.3f}")


if __name__ == "__main__":
    main()
