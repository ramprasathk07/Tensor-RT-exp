# Gate 0 — frozen reference

dtype: torch.float32, device: cuda

| case | prompt tokens | patches | grid | vision tokens | max pos | last argmax |
|---|---|---|---|---|---|---|
| `text_only` | 13 | 0 | [] | 0 | 12 | 19641 |
| `single_image` | 212 | 784 | [[1, 28, 28]] | 196 | 29 | 28715 |
| `two_images` | 286 | 1064 | [[1, 28, 28], [1, 14, 20]] | 266 | 43 | 785 |

Tensors in `tensors/<case>.pt` (gitignored). Each holds inputs,
`vision_merged`, `vision_deepstack_{0,1,2}`, `position_ids_3d`, and `logits`.

Note the `max pos` column: it is well below the prompt token count for image
cases because an image advances the M-RoPE counter by `max(h,w)//2`, not by
its token count. Any runtime that assumes `position == seq_len` will diverge.
