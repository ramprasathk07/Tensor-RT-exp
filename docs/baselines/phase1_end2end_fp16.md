# Phase 1c — end-to-end vision engine check

TensorRT vision precision: fp16. Decoder stays torch bf16 in B and C.

| pipeline | cases matching fp32 |
|---|---|
| B: torch bf16 (shipping baseline) | 2/3 |
| C: torch bf16 + TRT fp16 vision | 3/3 |

Scoring against fp32 rather than against the bf16 pipeline matters: bf16 is
itself an approximation, so any divergence it already shows is the floor for
what a reduced-precision engine can be held to.

### text_only
- **fp32**: `Three prime numbers are:  
2, 3, 5.`
- **bf16**: `Three prime numbers are:  
2, 3, 5.`
- **TRT-fp16**: `Three prime numbers are:  
2, 3, 5.`

### single_image
- **fp32**: `Based on the image provided, the dominant color is a solid, deep red. This color fills the entire frame, making it the sole and most prominent feature of the image.`
- **bf16**: `Based on the image provided, the dominant color is a solid, uniform shade of red. This color is consistent across the entire visual field, with no other colors or patterns present. The image is a flat`
- **TRT-fp16**: `Based on the image provided, the dominant color is a solid, deep red. This color fills the entire frame, making it the sole and most prominent feature of the image.`

### two_images
- **fp32**: `The two images are fundamentally different in their content, structure, and visual appearance.

- **Image 1:** This is a solid, uniform color. It is a single, continuous, and unbroken block of a deep `
- **bf16**: `The two images are fundamentally different in their content, structure, and visual appearance.

- **Image 1:** This is a solid, uniform color. It is a single, continuous, and unbroken block of a deep `
- **TRT-fp16**: `The two images are fundamentally different in their content, structure, and visual appearance.

- **Image 1:** This is a solid, uniform color. It is a single, continuous, and unbroken block of a deep `

