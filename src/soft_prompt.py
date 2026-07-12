"""
Scratch implementation of the paper's method.

The paper freezes ALL parameters of a pretrained causal LM (including its token
embedding layer) and adds ONE new trainable embedding matrix for a fixed-length
"persona info token" sequence (a soft prompt of length L, paper: L=200).

At every forward pass the soft prompt is prepended to the token embeddings:

    [ soft_prompt (L, H) ] ++ [ embed(utterance ++ response) ]  ->  frozen LM

Only `soft_prompt` receives gradients. The loss is computed on response tokens
only (utterance / prompt positions are masked with -100 in the labels).

This mirrors Lester et al. (2021) prompt-tuning, with the twist that the soft
prompt is initialized from the *embeddings of the persona sentences*, repeated
until L tokens are filled.
"""
import torch
import torch.nn as nn


class SoftPromptDialogue(nn.Module):
    def __init__(self, base_model, tokenizer, prompt_len=200, init_text=None):
        super().__init__()
        self.base = base_model
        self.tokenizer = tokenizer
        self.prompt_len = prompt_len

        # Freeze the whole pretrained model, including its embedding layer.
        for p in self.base.parameters():
            p.requires_grad_(False)

        hidden = self.base.config.hidden_size
        emb = self.base.get_input_embeddings()

        # Initialize the soft prompt from persona-sentence embeddings.
        if init_text:
            ids = tokenizer(init_text, add_special_tokens=False)["input_ids"]
            if len(ids) == 0:
                ids = [tokenizer.eos_token_id]
            reps = prompt_len // len(ids) + 1
            ids = (ids * reps)[:prompt_len]
            with torch.no_grad():
                init = emb(torch.tensor(ids, device=emb.weight.device)).float().clone()
        else:
            init = torch.randn(prompt_len, hidden) * 0.02

        # Kept in fp32 as a "master" parameter for stable optimization; it is
        # cast to the base model's dtype inside forward().
        self.soft_prompt = nn.Parameter(init)  # [L, H], the ONLY trainable tensor

    # -- helpers ----------------------------------------------------------------
    def _prepend(self, input_ids, attention_mask):
        """Return (inputs_embeds, attention_mask) with the soft prompt prepended."""
        emb = self.base.get_input_embeddings()
        tok_embeds = emb(input_ids)                                  # [B, T, H]
        B = input_ids.size(0)
        prompt = self.soft_prompt.to(tok_embeds.dtype).unsqueeze(0).expand(B, -1, -1)
        inputs_embeds = torch.cat([prompt, tok_embeds], dim=1)       # [B, L+T, H]
        prompt_mask = torch.ones(B, self.prompt_len, dtype=attention_mask.dtype,
                                 device=attention_mask.device)
        attn = torch.cat([prompt_mask, attention_mask], dim=1)       # [B, L+T]
        return inputs_embeds, attn

    def trainable_parameters(self):
        return [self.soft_prompt]

    # -- training forward -------------------------------------------------------
    def forward(self, input_ids, attention_mask, labels):
        inputs_embeds, attn = self._prepend(input_ids, attention_mask)
        # Soft-prompt positions never contribute to the loss.
        B = input_ids.size(0)
        prompt_labels = torch.full((B, self.prompt_len), -100,
                                   dtype=labels.dtype, device=labels.device)
        full_labels = torch.cat([prompt_labels, labels], dim=1)
        return self.base(inputs_embeds=inputs_embeds, attention_mask=attn,
                         labels=full_labels)

    # -- inference --------------------------------------------------------------
    @torch.no_grad()
    def generate(self, input_ids, attention_mask, **gen_kwargs):
        inputs_embeds, attn = self._prepend(input_ids, attention_mask)
        # With inputs_embeds and no input_ids, generate() returns ONLY the newly
        # generated token ids for decoder-only models.
        return self.base.generate(inputs_embeds=inputs_embeds,
                                   attention_mask=attn, **gen_kwargs)

    # -- checkpoint (only the soft prompt) --------------------------------------
    def save_prompt(self, path):
        torch.save({"soft_prompt": self.soft_prompt.detach().cpu(),
                    "prompt_len": self.prompt_len}, path)

    def load_prompt(self, path, map_location="cpu"):
        state = torch.load(path, map_location=map_location)
        with torch.no_grad():
            self.soft_prompt.copy_(state["soft_prompt"].to(self.soft_prompt.device))
