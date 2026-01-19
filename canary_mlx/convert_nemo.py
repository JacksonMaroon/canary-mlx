import argparse
import json
import math
import tarfile
from pathlib import Path
from typing import Dict, List

import mlx.core as mx
from mlx.utils import tree_flatten
import torch
import yaml

from canary_mlx.tokenizer import CanaryTokenizer, load_tokenizer_from_config
from canary_mlx.utils import _build_model


def extract_nemo(nemo_path: Path, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(nemo_path) as tar:
        tar.extractall(output_dir)
    return output_dir


def find_file(root: Path, name: str) -> Path:
    for path in root.rglob(name):
        return path
    raise FileNotFoundError(f"{name} not found under {root}")


def build_tokenizer_config(model_cfg: Dict, extracted_dir: Path, output_dir: Path) -> Dict:
    model_tokenizer_cfg = model_cfg.get("tokenizer") or {}
    model_path = model_tokenizer_cfg.get("model_path")
    if model_path:
        if isinstance(model_path, str) and model_path.startswith("nemo:"):
            model_path = model_path.split("nemo:", 1)[1]
        return {"type": model_tokenizer_cfg.get("type", "bpe"), "model_path": model_path}

    langs_cfg = model_tokenizer_cfg.get("langs") or {}
    tokenizer_root = extracted_dir
    tokenizer_models = list(tokenizer_root.rglob("tokenizer.model"))
    if not tokenizer_models:
        tokenizer_models = list(tokenizer_root.rglob("*_tokenizer.model"))
    if not tokenizer_models:
        raise FileNotFoundError("No tokenizer.model files found in the .nemo archive.")

    langs: Dict[str, Dict[str, str]] = {}
    for model_path in tokenizer_models:
        lang = model_path.parent.name
        if langs_cfg and lang not in langs_cfg:
            continue
        rel_dir = model_path.parent.relative_to(output_dir)
        langs[lang] = {"dir": str(rel_dir), "type": "bpe"}

    if not langs:
        for model_path in tokenizer_models:
            lang = model_path.parent.name
            rel_dir = model_path.parent.relative_to(output_dir)
            langs[lang] = {"dir": str(rel_dir), "type": "bpe"}

    return {"langs": langs}


def build_config(model_cfg: Dict, tokenizer: CanaryTokenizer, tokenizer_cfg: Dict) -> Dict:
    preprocessor_cfg = model_cfg["preprocessor"]
    encoder_cfg = model_cfg["encoder"]
    decoder_cfg = model_cfg["transf_decoder"]["config_dict"]
    head_cfg = model_cfg["head"]
    model_defaults = model_cfg.get("model_defaults", {})

    preprocess = {
        "sample_rate": preprocessor_cfg["sample_rate"],
        "normalize": preprocessor_cfg.get("normalize", "per_feature"),
        "window_size": preprocessor_cfg["window_size"],
        "window_stride": preprocessor_cfg["window_stride"],
        "window": preprocessor_cfg["window"],
        "features": preprocessor_cfg["features"],
        "n_fft": preprocessor_cfg["n_fft"],
        "dither": preprocessor_cfg.get("dither", 0.0),
        "pad_to": preprocessor_cfg.get("pad_to", 0),
        "pad_value": preprocessor_cfg.get("pad_value", 0.0),
        "preemph": preprocessor_cfg.get("preemph", 0.97),
        "mag_power": preprocessor_cfg.get("mag_power", 2.0),
    }

    subsampling_conv_channels = encoder_cfg.get("subsampling_conv_channels", encoder_cfg["d_model"])
    if subsampling_conv_channels == -1:
        subsampling_conv_channels = encoder_cfg["d_model"]

    encoder = {
        "feat_in": encoder_cfg["feat_in"],
        "n_layers": encoder_cfg["n_layers"],
        "d_model": encoder_cfg["d_model"],
        "n_heads": encoder_cfg["n_heads"],
        "ff_expansion_factor": encoder_cfg["ff_expansion_factor"],
        "subsampling_factor": encoder_cfg["subsampling_factor"],
        "self_attention_model": encoder_cfg["self_attention_model"],
        "subsampling": encoder_cfg["subsampling"],
        "conv_kernel_size": encoder_cfg["conv_kernel_size"],
        "subsampling_conv_channels": subsampling_conv_channels,
        "pos_emb_max_len": encoder_cfg["pos_emb_max_len"],
        "causal_downsampling": encoder_cfg.get("causal_downsampling", False),
        "use_bias": encoder_cfg.get("use_bias", True),
        "xscaling": encoder_cfg.get("xscaling", False),
        "att_context_size": encoder_cfg.get("att_context_size"),
        "subsampling_conv_chunking_factor": encoder_cfg.get("subsampling_conv_chunking_factor", 1),
    }

    vocab_size = int(8 * math.ceil(tokenizer.vocab_size / 8))

    decoder = {
        "hidden_size": decoder_cfg["hidden_size"],
        "inner_size": decoder_cfg["inner_size"],
        "num_layers": decoder_cfg["num_layers"],
        "num_attention_heads": decoder_cfg["num_attention_heads"],
        "max_sequence_length": decoder_cfg.get("max_sequence_length", 512),
        "num_token_types": decoder_cfg.get("num_token_types", 0),
        "embedding_dropout": decoder_cfg.get("embedding_dropout", 0.0),
        "learn_positional_encodings": decoder_cfg.get("learn_positional_encodings", False),
        "ffn_dropout": decoder_cfg.get("ffn_dropout", 0.0),
        "attn_score_dropout": decoder_cfg.get("attn_score_dropout", 0.0),
        "attn_layer_dropout": decoder_cfg.get("attn_layer_dropout", 0.0),
        "hidden_act": decoder_cfg.get("hidden_act", "relu"),
        "pre_ln": decoder_cfg.get("pre_ln", True),
        "pre_ln_final_layer_norm": decoder_cfg.get("pre_ln_final_layer_norm", True),
    }

    classifier = {
        "hidden_size": head_cfg.get("hidden_size", decoder_cfg["hidden_size"]),
        "num_classes": vocab_size,
        "num_layers": head_cfg.get("num_layers", 1),
        "activation": head_cfg.get("activation", "relu"),
        "log_softmax": head_cfg.get("log_softmax", True),
        "dropout": head_cfg.get("dropout", 0.0),
    }

    prompt_format = model_cfg.get("prompt_format", "canary")

    encoder_decoder_proj = None
    if model_defaults.get("asr_enc_hidden") and model_defaults.get("lm_dec_hidden"):
        if model_defaults["asr_enc_hidden"] != model_defaults["lm_dec_hidden"]:
            encoder_decoder_proj = model_defaults["lm_dec_hidden"]

    return {
        "preprocessor": preprocess,
        "encoder": encoder,
        "decoder": decoder,
        "classifier": classifier,
        "prompt_format": prompt_format,
        "encoder_decoder_proj": encoder_decoder_proj,
        "tokenizer": tokenizer_cfg,
    }


def _candidate_nemo_keys(mlx_key: str) -> List[str]:
    if mlx_key.startswith("decoder_embedding."):
        suffix = mlx_key[len("decoder_embedding.") :]
        return [
            f"transf_decoder._embedding.{suffix}",
            f"transf_decoder.embedding.{suffix}",
        ]
    if mlx_key.startswith("decoder."):
        suffix = mlx_key[len("decoder.") :]
        if ".third_sub_layer.linear1." in suffix:
            suffix = suffix.replace(".third_sub_layer.linear1.", ".third_sub_layer.dense_in.")
        if ".third_sub_layer.linear2." in suffix:
            suffix = suffix.replace(".third_sub_layer.linear2.", ".third_sub_layer.dense_out.")
        return [
            f"transf_decoder._decoder.{suffix}",
            f"transf_decoder.decoder.{suffix}",
        ]
    if mlx_key.startswith("classifier."):
        suffix = mlx_key[len("classifier.") :]
        if suffix.startswith("layers."):
            parts = suffix.split(".")
            if len(parts) >= 2:
                suffix = f"layer{parts[1]}" + ("" if len(parts) == 2 else f".{'.'.join(parts[2:])}")
        return [f"log_softmax.mlp.{suffix}", f"log_softmax.{suffix}"]
    if mlx_key.startswith("encoder_decoder_proj."):
        return [mlx_key]
    if mlx_key.startswith("encoder."):
        return [mlx_key]
    return [mlx_key]


def _convert_tensor(tensor: torch.Tensor, target_shape: tuple[int, ...]) -> mx.array:
    arr = tensor.detach().cpu()
    if tuple(arr.shape) == target_shape:
        return mx.array(arr.numpy())

    if arr.ndim == 2 and tuple(arr.T.shape) == target_shape:
        return mx.array(arr.T.numpy())

    if arr.ndim == 3:
        for perm in [(0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]:
            if tuple(arr.permute(*perm).shape) == target_shape:
                return mx.array(arr.permute(*perm).numpy())

    if arr.ndim == 4:
        for perm in [
            (0, 2, 3, 1),
            (0, 3, 2, 1),
            (2, 3, 1, 0),
            (3, 2, 1, 0),
        ]:
            if tuple(arr.permute(*perm).shape) == target_shape:
                return mx.array(arr.permute(*perm).numpy())

    raise ValueError(f"Cannot map tensor shape {tuple(arr.shape)} to {target_shape}")


def convert_weights(mlx_model, state_dict: Dict[str, torch.Tensor]) -> Dict[str, mx.array]:
    mlx_params = dict(tree_flatten(mlx_model.parameters()))
    mapped: Dict[str, mx.array] = {}
    missing = []

    for mlx_key, mlx_value in mlx_params.items():
        candidates = _candidate_nemo_keys(mlx_key)
        matched = False
        for nemo_key in candidates:
            if nemo_key in state_dict:
                try:
                    mapped[mlx_key] = _convert_tensor(state_dict[nemo_key], mlx_value.shape)
                    matched = True
                    break
                except ValueError:
                    continue
        if not matched:
            missing.append(mlx_key)
            mapped[mlx_key] = mlx_value

    if missing:
        print(f"Warning: {len(missing)} weights not found in checkpoint; kept MLX defaults.")

    return mapped


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert NeMo Canary .nemo to MLX weights")
    parser.add_argument("--nemo", type=Path, required=True, help="Path to .nemo checkpoint")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory")
    args = parser.parse_args()

    extracted_dir = extract_nemo(args.nemo, args.output_dir)
    config_path = find_file(extracted_dir, "model_config.yaml")
    weights_path = find_file(extracted_dir, "model_weights.ckpt")

    model_cfg_full = yaml.safe_load(open(config_path, "r"))
    model_cfg = model_cfg_full.get("model", model_cfg_full)

    tokenizer_cfg = build_tokenizer_config(model_cfg, extracted_dir, args.output_dir)
    tokenizer = load_tokenizer_from_config({"tokenizer": tokenizer_cfg}, args.output_dir)
    config_out = build_config(model_cfg, tokenizer, tokenizer_cfg)

    config_json_path = args.output_dir / "config.json"
    with open(config_json_path, "w") as f:
        json.dump(config_out, f, indent=2)

    mlx_model = _build_model(config_out, args.output_dir)

    checkpoint = torch.load(weights_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)

    mapped_weights = convert_weights(mlx_model, state_dict)
    mx.save_safetensors(str(args.output_dir / "model.safetensors"), mapped_weights)

    print(f"Saved MLX weights to {args.output_dir / 'model.safetensors'}")


if __name__ == "__main__":
    main()
