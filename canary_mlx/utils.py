import json
from pathlib import Path
from typing import Any, Dict

import mlx.core as mx
from dacite import from_dict
from huggingface_hub import hf_hub_download
from mlx.utils import tree_flatten, tree_unflatten

from canary_mlx.audio import PreprocessArgs
from canary_mlx.canary import CanaryModel, TransformerDecoderConfig
from canary_mlx.conformer import ConformerArgs
from canary_mlx.tokenizer import load_tokenizer_from_config
from canary_mlx.transformer import TokenClassifierConfig


def _load_config(path: Path) -> Dict[str, Any]:
    return json.load(open(path, "r"))


def _build_model(config: Dict[str, Any], base_dir: Path) -> CanaryModel:
    tokenizer = load_tokenizer_from_config(config, base_dir)

    preprocess = from_dict(PreprocessArgs, config["preprocessor"])
    encoder = from_dict(ConformerArgs, config["encoder"])
    decoder = from_dict(TransformerDecoderConfig, config["decoder"])
    classifier = from_dict(TokenClassifierConfig, config["classifier"])

    encoder_decoder_proj = config.get("encoder_decoder_proj")
    prompt_format = config.get("prompt_format", "canary")

    model = CanaryModel(
        preprocessor=preprocess,
        encoder=encoder,
        decoder=decoder,
        classifier=classifier,
        tokenizer=tokenizer,
        prompt_format=prompt_format,
        encoder_decoder_proj=encoder_decoder_proj,
    )
    model.eval()
    return model


def from_pretrained(
    hf_id_or_path: str,
    *,
    dtype: mx.Dtype = mx.bfloat16,
    cache_dir: str | Path | None = None,
) -> CanaryModel:
    try:
        config_path = hf_hub_download(
            hf_id_or_path, "config.json", cache_dir=cache_dir
        )
        weight_path = hf_hub_download(
            hf_id_or_path, "model.safetensors", cache_dir=cache_dir
        )
        base_dir = Path(config_path).parent
    except Exception:
        base_dir = Path(hf_id_or_path)
        config_path = base_dir / "config.json"
        weight_path = base_dir / "model.safetensors"

    config = _load_config(Path(config_path))
    model = _build_model(config, base_dir)
    model.load_weights(str(weight_path))

    curr_weights = dict(tree_flatten(model.parameters()))
    curr_weights = [(k, v.astype(dtype)) for k, v in curr_weights.items()]
    model.update(tree_unflatten(curr_weights))

    return model
