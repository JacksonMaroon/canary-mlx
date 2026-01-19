#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

import mlx.core as mx
from mlx.utils import tree_flatten, tree_unflatten

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "nemo-src"))

from nemo.collections.asr.models import EncDecMultiTaskModel  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402
from nemo.utils import AppState  # noqa: E402

from canary_mlx.utils import _build_model  # noqa: E402
from canary_mlx.transformer import (  # noqa: E402
    form_attention_mask,
    form_encoder_attention_mask,
    lengths_to_mask,
)


def load_nemo(nemo_dir: Path) -> EncDecMultiTaskModel:
    cfg = OmegaConf.load(nemo_dir / "model_config.yaml")
    model_cfg = cfg.get("model", cfg)
    if "restore_timestamps_model" in model_cfg:
        model_cfg.restore_timestamps_model = False
    if "train_ds" in model_cfg:
        model_cfg.train_ds = None
    if "validation_ds" in model_cfg:
        model_cfg.validation_ds = None
    if "test_ds" in model_cfg:
        model_cfg.test_ds = None
    app_state = AppState()
    app_state.nemo_file_folder = str(nemo_dir)
    model = EncDecMultiTaskModel(cfg=model_cfg)
    checkpoint = torch.load(nemo_dir / "model_weights.ckpt", map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model


def load_mlx(model_dir: Path, dtype: mx.Dtype) -> object:
    config = json.load(open(model_dir / "config.json", "r"))
    model = _build_model(config, model_dir)
    model.load_weights(str(model_dir / "model.safetensors"))

    params = tree_flatten(model.parameters())
    model.update(tree_unflatten([(k, v.astype(dtype)) for k, v in params]))
    return model


def diff_stats(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    diff = np.abs(a - b)
    return float(diff.max()), float(diff.mean())


def compare(name: str, a: np.ndarray, b: np.ndarray) -> None:
    if a.shape != b.shape:
        print(f"{name}: shape mismatch {a.shape} vs {b.shape}")
        return
    max_diff, mean_diff = diff_stats(a, b)
    print(f"{name}: max={max_diff:.6f} mean={mean_diff:.6f} shape={a.shape}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nemo", type=Path, default=REPO_ROOT / "canary-1b-v2-mlx")
    parser.add_argument("--mlx", type=Path, default=REPO_ROOT / "canary-1b-v2-mlx")
    parser.add_argument("--time", type=int, default=20)
    parser.add_argument("--dec-len", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    nemo_model = load_nemo(args.nemo)
    mlx_model = load_mlx(args.mlx, mx.float32)

    feat_in = nemo_model.encoder._feat_in
    torch.manual_seed(args.seed)
    feats = torch.randn(1, args.time, feat_in, dtype=torch.float32)
    lengths = torch.tensor([args.time], dtype=torch.int64)

    with torch.no_grad():
        nemo_pre, nemo_len = nemo_model.encoder.pre_encode(x=feats, lengths=lengths)

    mlx_feats = mx.array(feats.numpy())
    mlx_lengths = mx.array(lengths.numpy())
    mlx_pre, mlx_len = mlx_model.encoder.pre_encode(mlx_feats, mlx_lengths)

    compare("pre_encode.output", nemo_pre.cpu().numpy(), np.array(mlx_pre))
    compare("pre_encode.lengths", nemo_len.cpu().numpy(), np.array(mlx_len))

    nemo_layer_outs = {}
    hooks = []

    def _hook(idx):
        def handler(_module, _inputs, output):
            if isinstance(output, (tuple, list)):
                output = output[0]
            nemo_layer_outs[idx] = output.detach().cpu().numpy()

        return handler

    for idx, layer in enumerate(nemo_model.encoder.layers):
        hooks.append(layer.register_forward_hook(_hook(idx)))

    with torch.no_grad():
        nemo_out, nemo_out_len = nemo_model.encoder(
            audio_signal=nemo_pre, length=nemo_len, bypass_pre_encode=True
        )

    for hook in hooks:
        hook.remove()

    # Manual MLX layer traversal for per-layer outputs.
    mlx_x = mlx_pre
    pos_enc = mlx_model.encoder.pos_enc
    if pos_enc is not None:
        mlx_x, pos_emb = pos_enc(mlx_x, offset=0)
    else:
        pos_emb = None

    for idx, layer in enumerate(mlx_model.encoder.layers):
        mlx_x = layer(mlx_x, pos_emb=pos_emb, mask=None, cache=None)
        if idx in nemo_layer_outs:
            compare(f"layer[{idx}]", nemo_layer_outs[idx], np.array(mlx_x))

    compare("encoder.output", nemo_out.transpose(1, 2).cpu().numpy(), np.array(mlx_x))
    compare("encoder.lengths", nemo_out_len.cpu().numpy(), np.array(mlx_len))

    # Decoder comparison on random input ids.
    torch.manual_seed(args.seed + 1)
    vocab_size = nemo_model.transf_decoder.vocab_size
    input_ids = torch.randint(
        low=0, high=vocab_size, size=(1, args.dec_len), dtype=torch.long
    )
    dec_mask = torch.ones_like(input_ids)
    enc_states = nemo_out.transpose(1, 2)
    enc_mask = (
        torch.arange(enc_states.shape[1])[None, :] < nemo_out_len[:, None]
    ).to(torch.int64)

    with torch.no_grad():
        nemo_dec = nemo_model.transf_decoder(
            input_ids=input_ids,
            decoder_mask=dec_mask,
            encoder_embeddings=enc_states,
            encoder_mask=enc_mask,
        )
        nemo_logits = nemo_model.log_softmax(hidden_states=nemo_dec)

    mx_input_ids = mx.array(input_ids.numpy(), dtype=mx.int32)
    mx_dec_mask = mx.ones_like(mx_input_ids)
    mx_enc_mask = lengths_to_mask(mlx_len, mlx_x.shape[1])
    mx_dec_attn = form_attention_mask(mx_dec_mask, diagonal=0)
    mx_enc_attn = form_encoder_attention_mask(mx_enc_mask)
    mx_dec_embed = mlx_model.decoder_embedding(mx_input_ids)
    mx_dec_hidden = mlx_model.decoder(mx_dec_embed, mx_dec_attn, mlx_x, mx_enc_attn)
    mx_logits = mlx_model.classifier(mx_dec_hidden)

    compare("decoder.hidden", nemo_dec.cpu().numpy(), np.array(mx_dec_hidden))
    compare("decoder.logits", nemo_logits.cpu().numpy(), np.array(mx_logits))


if __name__ == "__main__":
    main()
