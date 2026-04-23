# Multimodality: ViT, CLIP, Stable Diffusion, and Flux

Multimodal AI systems process and generate multiple modalities — text, images,
audio, video — in a unified architecture. This lesson covers the four
foundational building blocks: Vision Transformers (ViT) for visual encoding,
CLIP for learning joint image-text representations, Latent Diffusion Models
(Stable Diffusion) for image generation, and Flux for flow-matching-based
generation. These components recur throughout modern AI systems: in GPT-4V,
Gemini, DALL-E 3, and every image generation pipeline in production.

## 1. The Core Intuition (The "Why")

Language is discrete (tokens); images are continuous high-dimensional signals
(pixels). Combining them requires a bridge between these representations.

Dosovitskiy et al. (2021) showed that a pure transformer applied directly to
image patches achieves state-of-the-art on vision benchmarks, removing the
need for convolutional inductive biases. CLIP (Radford et al., 2021) trained
a vision encoder and a text encoder jointly on 400M image-text pairs using
contrastive learning, learning a shared embedding space where semantically
similar images and captions are close.

Rombach et al. (2022) showed that running the diffusion process in **latent
space** (Latent Diffusion Models / Stable Diffusion) reduces the computational
cost of generation by 10-100x compared to pixel-space diffusion, enabling
high-resolution image synthesis in practical training times.

## 2. The Theoretical Underpinning

### Vision Transformer (ViT)

ViT divides an image $\mathbf{I} \in \mathbb{R}^{H \times W \times C}$ into
$N$ non-overlapping patches of size $P \times P$:

$$N = \frac{HW}{P^2}$$

Each patch is flattened and linearly projected to dimension $d$:

$$\mathbf{z}_0 = [x_{\text{class}}; \mathbf{x}_1^p E; \ldots; \mathbf{x}_N^p E] + \mathbf{E}_{\text{pos}}$$

where $E \in \mathbb{R}^{P^2 C \times d}$ is the patch embedding matrix and
$\mathbf{E}_{\text{pos}}$ is the positional encoding. A learnable `[CLS]`
token is prepended; its output representation is used for classification.

The patches are then processed by a standard transformer encoder. ViT shows
that images can be modelled as sequences of patches — there is nothing special
about spatial locality that requires convolutions.

### CLIP: Contrastive Image-Text Pretraining

CLIP (Radford et al., 2021) trains an image encoder $f$ and a text encoder $g$
so that matched image-text pairs have similar embeddings. For a batch of $N$
image-text pairs:

$$\text{logits}_{ij} = \frac{f(I_i) \cdot g(T_j)}{\tau}$$

where $\tau$ is a learned temperature parameter. The contrastive loss maximises
the diagonal of the $N \times N$ similarity matrix:

$$\mathcal{L}_{\text{CLIP}} = -\frac{1}{2N}\sum_{i=1}^{N}\left[\log\frac{e^{\text{logits}_{ii}}}{\sum_j e^{\text{logits}_{ij}}} + \log\frac{e^{\text{logits}_{ii}}}{\sum_j e^{\text{logits}_{ji}}}\right]$$

This InfoNCE loss is the same as cross-entropy loss applied to both
image-to-text and text-to-image classification. The result: a shared
embedding space where you can do zero-shot classification by computing
the similarity between an image embedding and text embeddings of class
descriptions.

### Denoising Diffusion Probabilistic Models (DDPM)

DDPM (Ho et al., 2020) defines a **forward process** that gradually adds
Gaussian noise to data over $T$ steps:

$$q(x_t \mid x_{t-1}) = \mathcal{N}(x_t; \sqrt{1-\beta_t}\, x_{t-1}, \beta_t I)$$

Using the reparametrisation $\bar{\alpha}_t = \prod_{s=1}^{t}(1-\beta_s)$,
the noisy sample at any step $t$ can be computed directly:

$$x_t = \sqrt{\bar{\alpha}_t}\, x_0 + \sqrt{1 - \bar{\alpha}_t}\, \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

The **reverse process** trains a neural network $\epsilon_\theta(x_t, t)$ to
predict the noise $\epsilon$ that was added:

$$\mathcal{L}_{\text{simple}} = \mathbb{E}_{t, x_0, \epsilon}\left[\|\epsilon - \epsilon_\theta(x_t, t)\|^2\right]$$

At inference, you start from $x_T \sim \mathcal{N}(0, I)$ and iteratively
denoise using the learned reverse process.

### Latent Diffusion Models (Stable Diffusion)

Running diffusion in pixel space is expensive for high-resolution images.
Rombach et al. (2022) trained a **VAE encoder-decoder**: the encoder $\mathcal{E}$
compresses the image to a latent code $z = \mathcal{E}(x)$ with $4\times8\times8$
spatial dimensions instead of $3\times512\times512$. Diffusion then operates
on $z$ rather than $x$.

At generation time, the decoder $\mathcal{D}$ maps the denoised latent
back to pixel space: $\hat{x} = \mathcal{D}(\hat{z})$.

Conditioning on text is done via **cross-attention**: the CLIP text embedding
is injected into the diffusion UNet's attention layers as the key and value.

### Flow Matching (Flux)

Flow matching (Lipman et al., 2022) offers a simpler training objective than
DDPM. Instead of learning to denoise, you learn a velocity field $v_\theta$
that moves samples from noise $p_1 = \mathcal{N}(0, I)$ to data $p_0$:

$$\frac{d}{dt} x_t = v_\theta(x_t, t)$$

The conditional flow matching objective is:

$$\mathcal{L}_{\text{CFM}} = \mathbb{E}_{t, x_0, x_1}\left[\|v_\theta(x_t, t) - (x_0 - x_1)\|^2\right]$$

where $x_t = (1-t) x_1 + t x_0$ is the interpolant between noise $x_1$ and
data $x_0$. This is simpler to train and achieves better sample quality with
fewer NFEs (number of function evaluations) than DDPM. Flux uses this
objective with a transformer backbone (DiT) instead of a UNet.

## 3. Implementation: From Scratch (Python + NumPy)

```python
from __future__ import annotations

import numpy as np


def patchify(image: np.ndarray, patch_size: int) -> np.ndarray:
    """Divide an image into non-overlapping patches.

    Args:
        image:      Array of shape (H, W, C).
        patch_size: Size of each square patch P.

    Returns:
        Patches of shape (N, P*P*C) where N = (H*W) / (P^2).
    """
    H, W, C = image.shape
    assert H % patch_size == 0 and W % patch_size == 0
    P     = patch_size
    n_h   = H // P
    n_w   = W // P
    # Reshape and transpose to (n_h * n_w, P * P * C)
    patches = (image.reshape(n_h, P, n_w, P, C)
               .transpose(0, 2, 1, 3, 4)
               .reshape(-1, P * P * C))
    return patches.astype(np.float32)


def clip_loss_numpy(
    image_embeddings: np.ndarray,
    text_embeddings:  np.ndarray,
    temperature:      float = 0.07,
) -> float:
    """Compute InfoNCE CLIP loss for a batch of image-text pairs.

    Args:
        image_embeddings: L2-normalised image features of shape (N, d).
        text_embeddings:  L2-normalised text features of shape (N, d).
        temperature:      Logit scale parameter.

    Returns:
        Scalar CLIP loss.
    """
    N     = image_embeddings.shape[0]
    # Cosine similarity matrix scaled by temperature
    logits = image_embeddings @ text_embeddings.T / temperature
    labels = np.arange(N)
    # Cross-entropy in both directions
    def ce(logits: np.ndarray, labels: np.ndarray) -> float:
        log_probs = logits - np.log(np.exp(logits).sum(axis=1, keepdims=True))
        return float(-log_probs[np.arange(len(labels)), labels].mean())
    return 0.5 * (ce(logits, labels) + ce(logits.T, labels))
```

## 4. Implementation: Production-Grade (Diffusers)

```python
from __future__ import annotations

import torch
from diffusers import (
    StableDiffusionPipeline,
    FluxPipeline,
    DPMSolverMultistepScheduler,
)
from PIL import Image


def load_sdxl_pipeline(
    model_id: str = "stabilityai/stable-diffusion-xl-base-1.0",
    device:   str = "cuda",
) -> StableDiffusionPipeline:
    """Load SDXL with the DPM++ 2M solver for fast, high-quality generation.

    DPM++ 2M Karras achieves near-optimal sample quality in 20-30 steps,
    vs 50+ for DDIM. The scheduler is pluggable — swap it without
    retraining the model.
    """
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype = torch.float16,
        use_safetensors = True,
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config,
        algorithm_type = "dpmsolver++",
        use_karras_sigmas = True,
    )
    return pipe.to(device)


def generate_image(
    pipe:          StableDiffusionPipeline,
    prompt:        str,
    negative_prompt: str = "blurry, low quality, artefacts",
    num_steps:     int   = 25,
    guidance_scale: float = 7.5,
    seed:          int   = 42,
) -> Image.Image:
    """Generate an image with classifier-free guidance.

    Args:
        guidance_scale: CFG scale. Higher = more prompt-adherent but
            less diverse. Values in [6, 8] work well for most prompts.

    Returns:
        PIL Image.
    """
    generator = torch.Generator(device=pipe.device).manual_seed(seed)
    result    = pipe(
        prompt          = prompt,
        negative_prompt = negative_prompt,
        num_inference_steps = num_steps,
        guidance_scale  = guidance_scale,
        generator       = generator,
    )
    return result.images[0]


def load_flux_pipeline(
    model_id: str = "black-forest-labs/FLUX.1-dev",
    device:   str = "cuda",
) -> FluxPipeline:
    """Load a Flux flow-matching image generation pipeline.

    Flux uses a Diffusion Transformer (DiT) backbone and flow matching
    training objective, enabling high-quality generation in ~28 steps.
    """
    pipe = FluxPipeline.from_pretrained(
        model_id,
        torch_dtype = torch.bfloat16,
    )
    return pipe.to(device)
```

## 5. Production Pitfalls & Pro-Tips

- **Mistake:** Using ViT models without enough training data. ViT lacks the
  convolutional inductive biases (local connectivity, translation equivariance)
  of CNNs. On small datasets, CNNs outperform ViT.

  **Fix:** Use ViT only when you have enough data (> 100K examples) or use
  a pre-trained ViT (e.g., `openai/clip-vit-large-patch14`).

- **Mistake:** Using Stable Diffusion for tasks requiring exact image fidelity
  (e.g., product photography). Diffusion models are generative, not
  interpolative; they hallucinate details.

  **Fix:** Use controlnet, image-to-image, or inpainting variants to
  constrain generation to reference images.

- **Mistake:** Running Stable Diffusion in FP32. The model was trained and
  optimised for FP16/BF16; FP32 doubles memory and slows generation.

  **Fix:** Always use `torch_dtype=torch.float16` or `torch.bfloat16`.

- **Mistake:** Using a very high CFG (classifier-free guidance) scale (> 10).
  High CFG over-saturates colours and introduces artefacts.

  **Fix:** Use CFG in the range 6–8 for most prompts. Use a negative prompt
  to push away unwanted attributes.

- **Mistake:** Ignoring safety checkers and content filters when deploying
  image generation APIs publicly.

  **Fix:** Integrate NSFW classification and prompt filtering before
  calling the generation pipeline. Review Stability AI and Hugging Face
  usage policies.

## 6. Knowledge Check

1. **Conceptual:** ViT treats image patches as a sequence. What positional
   information is lost compared to a CNN? How do learned positional embeddings
   and RoPE address this? What happens if you test ViT at a higher resolution
   than it was trained on?

2. **Conceptual:** Explain the CLIP contrastive loss. Why does maximising the
   diagonal of the similarity matrix cause the model to learn aligned
   image-text representations? What is the role of the temperature $\tau$?

3. **Coding challenge:** Load a CLIP model using Hugging Face Transformers.
   Embed a set of images and corresponding captions. Compute the cosine
   similarity matrix and verify that diagonal entries are higher than
   off-diagonal entries. Then perform zero-shot image classification on
   CIFAR-10.

## References

1. Dosovitskiy, A., Beyer, L., Kolesnikov, A., et al. (2021). An image is worth 16×16 words: Transformers for image recognition at scale. *ICLR 2021*. arXiv:2010.11929.
2. Radford, A., Kim, J. W., Hallacy, C., et al. (2021). Learning transferable visual models from natural language supervision. *ICML 2021*. arXiv:2103.00020. *(CLIP.)*
3. Ho, J., Jain, A., & Abbeel, P. (2020). Denoising diffusion probabilistic models. *NeurIPS 2020*. arXiv:2006.11239.
4. Rombach, R., Blattmann, A., Lorenz, D., Esser, P., & Ommer, B. (2022). High-resolution image synthesis with latent diffusion models. *CVPR 2022*. arXiv:2112.10752.
5. Lipman, Y., Chen, R. T. Q., Ben-Hamu, H., Nickel, M., & Le, M. (2022). Flow matching for generative modelling. *ICLR 2023*. arXiv:2210.02747.
