import json
from pathlib import Path
from typing import Dict, List

import torch
import typer
from mlx.utils import tree_flatten

from canary_mlx.convert_nemo import _candidate_nemo_keys
from canary_mlx.utils import _build_model

app = typer.Typer(add_completion=False)


def _can_map_shape(src_shape: tuple[int, ...], target_shape: tuple[int, ...]) -> bool:
    if src_shape == target_shape:
        return True
    if len(src_shape) == 2 and (src_shape[1], src_shape[0]) == target_shape:
        return True
    if len(src_shape) == 3:
        for perm in [(0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]:
            if tuple(src_shape[i] for i in perm) == target_shape:
                return True
    if len(src_shape) == 4:
        for perm in [(0, 2, 3, 1), (0, 3, 2, 1), (2, 3, 1, 0), (3, 2, 1, 0)]:
            if tuple(src_shape[i] for i in perm) == target_shape:
                return True
    return False


@app.command()
def main(
    nemo_ckpt: Path = typer.Option(..., "--nemo-ckpt", exists=True),
    mlx_config: Path = typer.Option(..., "--mlx-config", exists=True),
    output: Path = typer.Option(None, "--output"),
    max_list: int = typer.Option(200, "--max-list"),
):
    config = json.load(open(mlx_config, "r"))
    model = _build_model(config, mlx_config.parent)

    checkpoint = torch.load(nemo_ckpt, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)

    mlx_params = dict(tree_flatten(model.parameters()))

    missing: List[str] = []
    mismatched: List[Dict[str, object]] = []
    used_nemo = set()

    for mlx_key, mlx_value in mlx_params.items():
        candidates = _candidate_nemo_keys(mlx_key)
        matched = False
        mismatch_shapes = []
        for nemo_key in candidates:
            if nemo_key not in state_dict:
                continue
            nemo_shape = tuple(state_dict[nemo_key].shape)
            if _can_map_shape(nemo_shape, mlx_value.shape):
                matched = True
                used_nemo.add(nemo_key)
                break
            mismatch_shapes.append({"nemo_key": nemo_key, "nemo_shape": nemo_shape})
        if not matched:
            if mismatch_shapes:
                mismatched.append(
                    {
                        "mlx_key": mlx_key,
                        "mlx_shape": mlx_value.shape,
                        "candidates": mismatch_shapes,
                    }
                )
            else:
                missing.append(mlx_key)

    extra = [k for k in state_dict.keys() if k not in used_nemo]

    report = {
        "missing": missing,
        "mismatched": mismatched,
        "extra": extra,
        "counts": {
            "missing": len(missing),
            "mismatched": len(mismatched),
            "extra": len(extra),
        },
    }

    typer.echo(f"Missing: {len(missing)}")
    typer.echo(f"Mismatched: {len(mismatched)}")
    typer.echo(f"Extra (unused) NeMo keys: {len(extra)}")

    if missing:
        typer.echo("Missing keys:")
        for key in missing[:max_list]:
            typer.echo(f"- {key}")

    if mismatched:
        typer.echo("Mismatched shapes:")
        for item in mismatched[:max_list]:
            typer.echo(f"- {item['mlx_key']} (mlx {item['mlx_shape']})")

    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w") as handle:
            json.dump(report, handle, indent=2)
        typer.echo(f"Wrote report to {output}")


if __name__ == "__main__":
    app()
