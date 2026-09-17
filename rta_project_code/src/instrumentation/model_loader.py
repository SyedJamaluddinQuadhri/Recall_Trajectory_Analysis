"""
model_loader.py — Loads Phi-2 in 4-bit quantization and exposes its
internal structure for hooking.

This is the foundation everything else builds on: we need access to
(a) the residual stream at every layer, and
(b) the MLP sublayer's output at every layer,
during a single forward pass.
"""

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from src.utils import get_logger

log = get_logger("instrumentation.model_loader")

MODEL_ID = "microsoft/phi-2"


def load_phi2(device: str = None, use_4bit: bool = True):
    """Load Phi-2 with `output_hidden_states=True` for downstream access.

    Parameters
    ----------
    device: str | None
        Desired device ("cuda" or "cpu"). If ``None`` the function picks
        ``cuda`` when available, otherwise ``cpu``.
    use_4bit: bool
        Whether to attempt a 4‑bit quantised load. Falls back to full‑precision
        on CPU if GPU memory is insufficient.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Ensure a folder exists for any CPU/GPU off‑loading performed by 🤗
    os.makedirs("offload", exist_ok=True)
    log.info(f"Loading {MODEL_ID} (4bit={use_4bit}) on {device}...")

    try:
        if use_4bit and device == "cuda":
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_ID,
                quantization_config=bnb_config,
                device_map="auto",
                offload_folder="offload",
                trust_remote_code=True,
                output_hidden_states=True,
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_ID,
                torch_dtype=torch.float32,
                trust_remote_code=True,
                output_hidden_states=True,
            )
            if device != "cpu":
                model = model.to(device)
    except ValueError as e:
        # Most likely a GPU‑memory error – retry on CPU with full precision.
        log.warning(f"Initial load failed ({e}). Falling back to CPU loading.")
        device = "cpu"
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float32,
            trust_remote_code=True,
            output_hidden_states=True,
        )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    model.eval()

    # Freeze the whole model – we never train Phi‑2 itself.
    for param in model.parameters():
        param.requires_grad_(False)

    n_layers = model.config.num_hidden_layers
    hidden_dim = model.config.hidden_size
    log.info(f"Loaded. n_layers={n_layers}, hidden_dim={hidden_dim}")
    if device == "cuda":
        vram = torch.cuda.memory_allocated(0) / 1e9
        log.info(f"VRAM used: {vram:.2f}GB")

    return model, tokenizer


def get_model_structure(model):
    """Return useful structural components of a Phi‑2 model.

    The rest of the pipeline expects a dictionary containing the layer list
    and a few key tensors (final layer‑norm and the LM head).
    """
    layers = model.model.layers
    n_layers = len(layers)
    hidden_dim = model.config.hidden_size
    vocab_size = model.config.vocab_size
    return {
        "layers": layers,
        "n_layers": n_layers,
        "hidden_dim": hidden_dim,
        "vocab_size": vocab_size,
        "final_ln": model.model.final_layernorm,
        "unembed": model.lm_head,
    }
