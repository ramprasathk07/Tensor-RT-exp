# qwen3vl — standalone Qwen3-VL

A from-scratch implementation of `Qwen3VLForConditionalGeneration` on plain
`torch.nn`, with no `transformers` dependency at runtime. Module and parameter
names match the released checkpoint exactly, so `model.safetensors` loads with a
strict `load_state_dict`.

Verified against `transformers` 5.10.2 on `Qwen/Qwen3-VL-2B-Instruct`:

| check | result |
|---|---|
| vision tower outputs (merged + 3 DeepStack) | bit-exact, `max_abs = 0.0` |
| full logits, fp32, 434-token 2-image prompt | `max_abs = 4.5e-4`, argmax agreement 100% |
| `pixel_values` from the bundled processor | `max_abs = 5.9e-8` |
| greedy generation, 3 prompt shapes, 48 tokens | token-for-token identical |

## Usage

```python
from qwen3vl import GenerationConfig, Qwen3VLProcessor, generate, load_qwen3vl

model = load_qwen3vl("models/Qwen3-VL-2B-Instruct", dtype=torch.bfloat16, device="cuda")
processor = Qwen3VLProcessor.from_pretrained("models/Qwen3-VL-2B-Instruct")

messages = [{"role": "user", "content": [
    {"type": "image", "image": "photo.jpg"},
    {"type": "text", "text": "What is in this image?"},
]}]

batch = processor.apply_chat_template(messages, device="cuda")
out = generate(model, **batch, config=GenerationConfig(max_new_tokens=128))
print(processor.decode(out.generated[0]))
```

`load_qwen3vl` builds the skeleton on the meta device and moves real tensors in
with `assign=True`, so no random weights are ever materialised — loading costs
one copy rather than an init pass plus a copy.

## Files

| file | contents |
|---|---|
| `config.py` | dataclass configs parsed from `config.json`, including the `rope_scaling` → `mrope_section` unpacking |
| `modeling.py` | vision tower, text decoder, top-level model, KV cache |
| `loading.py` | safetensors → model, meta-device init, tied-weight handling |
| `processing.py` | smart resize, patchification, chat template, tokenization |
| `generation.py` | greedy/sampled decoding with M-RoPE position tracking |

## Architecture

2.128B parameters total: 407.0M vision, 1720.6M text (of which 311.2M is the
embedding table, tied to `lm_head`).

**Vision tower** — 24 blocks, width 1024, 16 heads, GELU-tanh MLP at width 4096.
Patches are 16×16 over 2 temporal frames, projected by a `Conv3d`. Attention is
bidirectional but blocked per image: `cu_seqlens` splits the packed patch
sequence so no token attends across image boundaries. Output goes through a
patch merger that folds each 2×2 block of patches into one 2048-d LLM token.

**Text decoder** — 28 Qwen3 layers, width 2048, 16 query heads over 8 KV heads
(GQA), head dim 128, SwiGLU MLP at width 6144, RMSNorm throughout. Q and K each
get their own RMSNorm over the head dimension before rotary is applied.

Three things distinguish this from a conventional VLM:

**DeepStack.** The vision tower does not only feed the embedding layer. Blocks
5, 11 and 17 each tap a separate patch merger, and those three feature maps are
*added into* the hidden states of text decoder layers 0, 1 and 2 at the image
token positions. This is why `Qwen3VLTextModel` is not usable as a text-only
model in the general case — the injection points are part of the forward pass.
The DeepStack mergers normalise *after* the 2×2 shuffle (LayerNorm width 4096)
while the output merger normalises before it (width 1024); the checkpoint's norm
shapes encode this difference.

**Interleaved M-RoPE.** Text positions are 3-dimensional — time, height, width —
with band budget `[24, 20, 20]` summing to `head_dim / 2 = 64`. Rather than
giving each axis a contiguous chunk of the frequency spectrum, bands are
interleaved `T H W T H W …` with a tail of 4 pure-time bands, so neighbouring
bands stay at adjacent wavelengths. `apply_interleaved_mrope` performs the fold
with strided slice assignment.

Position *allocation* matters as much as the encoding: a text run advances all
three axes together by its length, but an image block consumes only
`max(h, w) // merge_size` positions regardless of how many tokens it expands to.
A 28×28 patch grid becomes 196 tokens but advances the position counter by 14.
Generated tokens therefore continue from `max(prompt_positions) + 1`, not from
`seq_len` — `generate` tracks this explicitly instead of relying on sequence
length.

**Bilinear position-embedding interpolation.** The tower holds a fixed 48×48
learned position table (2304 entries). For each image it computes the four
bilinear corner indices and weights that resample that table onto the actual
patch grid, so arbitrary resolutions work without retraining. The indices are
emitted already permuted into merge-block order.

## Token ordering

One ordering runs through the whole pipeline and every stage assumes it:

```
[frame][h-block][w-block][h-within-block][w-within-block]
```

The image processor emits patches in this order, `build_vision_position_ids`
generates `(row, col)` pairs in this order, and the patch mergers rely on it so
that a plain `.view(-1, hidden * 4)` groups the correct 2×2 neighbourhood.
Changing the permutation in `preprocess_one` silently corrupts the merge step
without raising anything.

## Preprocessing note

`smart_resize` rounds both dimensions to multiples of `patch_size * merge_size`
(32) and scales the area into `[min_pixels, max_pixels]`. The resize runs while
the image is still `uint8` and rounds back to `uint8` afterwards — matching the
reference, where resize precedes rescaling. Doing the resize in float instead
shifts `pixel_values` by ~4e-3, which is small but not zero.
