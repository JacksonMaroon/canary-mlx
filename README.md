# Canary MLX

An implementation of Nvidia's Canary multitask ASR/AST models for Apple Silicon using MLX.

## Installation

```bash
pip install -e .
```

## CLI Quick Start

```bash
canary-mlx <audio_files> --source-lang en --target-lang en --task transcribe
```

Beam decoding example (beam size 4):

```bash
canary-mlx audio.wav --model /path/to/canary-mlx/model \\
  --source-lang en --target-lang en --task transcribe \\
  --decoding beam --beam-size 4 --length-penalty 0.0 --max-generation-delta 50
```

## Python API

```py
from canary_mlx import from_pretrained

model = from_pretrained("/path/to/canary-mlx/model")
result = model.transcribe("audio.wav", source_lang="en", target_lang="en", task="transcribe")
print(result.text)
```

## Converting NeMo checkpoints

The official checkpoints are distributed as `.nemo` archives. To convert:

```bash
python -m canary_mlx.convert_nemo \
  --nemo /path/to/canary-1b-v2.nemo \
  --output-dir /path/to/canary-mlx/model
```

Notes:
- The converter expects `torch` and `mlx` installed.
- It will extract `model_config.yaml`, `model_weights.ckpt`, and tokenizer assets from the `.nemo` archive.

## Batch Inference

```bash
canary-mlx batch /path/to/audio_dir --recursive \\
  --model /path/to/canary-mlx/model \\
  --source-lang en --target-lang en --task transcribe \\
  --output transcripts.csv
```

## Weight Mapping Report

```bash
canary-mlx-diff \\
  --nemo-ckpt /path/to/model_weights.ckpt \\
  --mlx-config /path/to/canary-mlx/model/config.json \\
  --output weight_report.json
```

## Architecture Notes

- `canary-1b` uses a 24-layer FastConformer encoder and a 24-layer Transformer decoder.
- The 32-layer encoder / 4-layer decoder variant corresponds to the `canary-1b-flash` family.

## Known Limitations

- Audio preprocessing is not bit-exact with NeMo.
