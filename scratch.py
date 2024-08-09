# %%
import einops
import transformer_lens as tl
import torch
# %%
model = tl.HookedTransformer.from_pretrained('pythia-160m')
# %%
prompts = [" mathematic", " controvers", " veh"]
tokens = [model.to_single_token(t) for t in prompts]
tokens = torch.tensor(tokens).unsqueeze(-1)
logits, cache = model.run_with_cache(tokens)
resid_pre = cache["blocks.0.hook_resid_pre"]
logits = einops.einsum(resid_pre, model.W_U, "batch seq d_model, d_model d_vocab -> batch seq d_vocab")

for i, prompt in enumerate(prompts):
    print(f"{prompt} --> {model.to_string(logits[i].argmax(-1))}")
# %%
