"""
Tiny chat server to interactively test the persona model's multi-turn context.

Zero extra deps (Python stdlib http.server). Serves a chat page at GET / and a
JSON API at POST /chat. Two modes:
  - hybrid   : explicit naturalistic system prompt + trained soft prompt
  - explicit : explicit naturalistic system prompt only (soft prompt off)
Both use the Instruct model's native chat template, so dialogue history is
handled properly and you can probe long-context / callback behavior by hand.

Run inside the container, e.g.:
  PORT=8000 GPU=4 scripts/docker_run.sh python -u src/serve.py \
      --model Qwen/Qwen3-4B-Instruct-2507 --prompt outputs/qwen3_4b/persona_CP.pt \
      --persona_file data/processed/persona_CP/persona.txt
"""
import argparse, json, threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from train import pick_device
from soft_prompt import SoftPromptDialogue
from persona_prompt import build_system

STATE = {}
LOCK = threading.Lock()

PAGE = """<!doctype html><html lang="ja"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Qwen3-4B ペルソナ対話テスト</title><style>
:root{--bg:#0f1116;--card:#171a21;--ink:#e8eaf0;--muted:#9aa3b3;--line:#262a33;
--user:#4f46e5;--bot:#20242c;--accent:#8b8bf5}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);
font-family:-apple-system,"Hiragino Sans","Noto Sans JP",sans-serif;height:100vh;display:flex;flex-direction:column}
header{padding:12px 16px;border-bottom:1px solid var(--line);display:flex;gap:14px;align-items:center;flex-wrap:wrap}
header b{font-size:15px}.ctrl{display:flex;gap:6px;align-items:center;font-size:13px;color:var(--muted)}
select,button{background:var(--card);color:var(--ink);border:1px solid var(--line);border-radius:8px;padding:6px 10px;font-size:13px}
button{cursor:pointer}button.primary{background:var(--user);border-color:var(--user);color:#fff}
#log{flex:1;overflow-y:auto;padding:18px;display:flex;flex-direction:column;gap:12px;max-width:820px;width:100%;margin:0 auto}
.msg{max-width:78%;padding:10px 14px;border-radius:14px;line-height:1.55;white-space:pre-wrap;word-break:break-word}
.u{align-self:flex-end;background:var(--user);color:#fff;border-bottom-right-radius:4px}
.b{align-self:flex-start;background:var(--bot);border:1px solid var(--line);border-bottom-left-radius:4px}
.meta{font-size:11px;color:var(--muted);margin-top:4px}
footer{border-top:1px solid var(--line);padding:12px;display:flex;gap:10px;max-width:820px;width:100%;margin:0 auto}
#inp{flex:1;background:var(--card);color:var(--ink);border:1px solid var(--line);border-radius:12px;padding:11px 14px;font-size:15px;resize:none}
.sys{align-self:center;font-size:12px;color:var(--muted);background:var(--card);border:1px dashed var(--line);border-radius:10px;padding:8px 12px;max-width:90%}
</style></head><body>
<header>
  <b>🗣️ Qwen3-4B ペルソナ対話</b>
  <span class="ctrl">模式 <select id="mode"><option value="hybrid">混合(显式+soft prompt)</option><option value="explicit">仅显式 prompt</option></select></span>
  <span class="ctrl"><label><input type="checkbox" id="rep" checked> 重复惩罚</label></span>
  <span class="ctrl" id="stat"></span>
  <button id="reset">重置对话</button>
</header>
<div id="log"></div>
<footer>
  <textarea id="inp" rows="1" placeholder="日本語で話しかけてみてください…(Enterで送信)"></textarea>
  <button class="primary" id="send">送信</button>
</footer>
<script>
let history=[];
const log=document.getElementById('log'),inp=document.getElementById('inp'),stat=document.getElementById('stat');
function add(role,text,meta){const d=document.createElement('div');d.className='msg '+(role==='user'?'u':'b');d.textContent=text;
  log.appendChild(d);if(meta){const m=document.createElement('div');m.className='meta';m.textContent=meta;d.appendChild(m);}
  log.scrollTop=log.scrollHeight;return d;}
function sys(t){const d=document.createElement('div');d.className='sys';d.textContent=t;log.appendChild(d);log.scrollTop=log.scrollHeight;}
sys('人格=CP(西瓜/読書ミステリー/化学専攻/福岡/水泳/長風呂…)。多輪で文脈保持や前の話題への言及を試してください。');
async function send(){const text=inp.value.trim();if(!text)return;inp.value='';add('user',text);history.push({role:'user',content:text});
  const wait=add('bot','…');const t0=performance.now();
  try{const r=await fetch('/chat',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({history,mode:document.getElementById('mode').value,rep_penalty:document.getElementById('rep').checked?1.3:1.0})});
    const j=await r.json();const dt=((performance.now()-t0)/1000).toFixed(1);
    wait.textContent=j.reply;history.push({role:'assistant',content:j.reply});
    const mm=document.getElementById('mode').value;
    const meta=document.createElement('div');meta.className='meta';meta.textContent=`${mm} · ctx ${j.ctx_tokens} tok · ${dt}s`;wait.appendChild(meta);
    stat.textContent=`${history.length/2} 回 · 上下文 ${j.ctx_tokens} tok`;
  }catch(e){wait.textContent='⚠️ エラー: '+e;}
  log.scrollTop=log.scrollHeight;}
document.getElementById('send').onclick=send;
inp.addEventListener('keydown',e=>{if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();send();}});
document.getElementById('reset').onclick=()=>{history=[];log.innerHTML='';sys('对话已重置。');stat.textContent='';};
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype="application/json"):
        b = body if isinstance(body, bytes) else body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def log_message(self, *a):
        pass

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._send(200, PAGE, "text/html; charset=utf-8")
        else:
            self._send(404, "not found", "text/plain")

    def do_POST(self):
        if self.path != "/chat":
            return self._send(404, "not found", "text/plain")
        n = int(self.headers.get("Content-Length", 0))
        req = json.loads(self.rfile.read(n) or b"{}")
        try:
            reply, ctx = generate(req.get("history", []), req.get("mode", "hybrid"),
                                  float(req.get("rep_penalty", 1.3)))
            self._send(200, json.dumps({"reply": reply, "ctx_tokens": ctx}, ensure_ascii=False))
        except Exception as e:
            self._send(500, json.dumps({"reply": f"[error] {e}", "ctx_tokens": 0}))


def generate(history, mode, rep_penalty):
    tok, model, base, system, device = (STATE[k] for k in
                                        ("tok", "model", "base", "system", "device"))
    messages = [{"role": "system", "content": system}] + [
        {"role": h["role"], "content": h["content"]} for h in history]
    enc = tok.apply_chat_template(messages, add_generation_prompt=True,
                                  return_tensors="pt", return_dict=True).to(device)
    in_len = enc["input_ids"].shape[1]
    gen = dict(max_new_tokens=80, do_sample=False, num_beams=1,
               pad_token_id=tok.pad_token_id or tok.eos_token_id)
    if rep_penalty != 1.0:
        gen["repetition_penalty"] = rep_penalty
        gen["no_repeat_ngram_size"] = 3
    with LOCK, torch.no_grad():
        if mode == "hybrid" and model is not None:
            out = model.generate(input_ids=enc["input_ids"],
                                 attention_mask=enc["attention_mask"], **gen)
            reply = tok.decode(out[0], skip_special_tokens=True).strip()
        else:
            out = base.generate(**enc, **gen)
            reply = tok.decode(out[0, in_len:], skip_special_tokens=True).strip()
    return reply, int(in_len)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--persona_file", required=True)
    ap.add_argument("--reply_lang", default="Japanese")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()

    device = pick_device("auto")
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(args.model,
                                                dtype=getattr(torch, args.dtype)).to(device)
    base.eval()

    # Try to attach the soft prompt for hybrid mode; skip gracefully on dim mismatch.
    model = None
    try:
        m = SoftPromptDialogue(base, tok, prompt_len=200).to(device)
        m.load_prompt(args.prompt, map_location=device)
        if m.soft_prompt.shape[1] == base.config.hidden_size:
            m.base.eval(); model = m
            print(f"hybrid mode ready (soft prompt {tuple(m.soft_prompt.shape)})")
        else:
            print(f"WARN soft prompt hidden {m.soft_prompt.shape[1]} != model "
                  f"{base.config.hidden_size}; hybrid disabled")
    except Exception as e:
        print(f"WARN could not load soft prompt ({e}); hybrid disabled")

    lines = [l.strip() for l in open(args.persona_file) if l.strip()]
    STATE.update(tok=tok, base=base, model=model, device=device,
                 system=build_system("\n".join(lines), args.reply_lang))

    srv = ThreadingHTTPServer(("0.0.0.0", args.port), Handler)
    print(f"serving on 0.0.0.0:{args.port} (model={args.model})")
    srv.serve_forever()


if __name__ == "__main__":
    main()
