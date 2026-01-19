import csv
from pathlib import Path
from typing import Iterable, List, Optional

import typer

from canary_mlx import from_pretrained
from canary_mlx.canary import BeamDecoding, DecodingConfig, GreedyDecoding

app = typer.Typer(add_completion=False)

AUDIO_EXTS = {
    ".wav",
    ".mp3",
    ".flac",
    ".m4a",
    ".ogg",
    ".opus",
    ".aac",
    ".wma",
    ".aiff",
    ".aif",
    ".caf",
}


def _collect_audio_paths(paths: Iterable[Path], recursive: bool) -> List[Path]:
    collected: List[Path] = []
    for path in paths:
        if path.is_dir():
            pattern = "**/*" if recursive else "*"
            for candidate in path.glob(pattern):
                if candidate.is_file() and candidate.suffix.lower() in AUDIO_EXTS:
                    collected.append(candidate)
        else:
            collected.append(path)
    unique = []
    seen = set()
    for path in sorted(collected):
        if path not in seen:
            unique.append(path)
            seen.add(path)
    return unique


def _build_decoding_config(
    decoding: str,
    beam_size: int,
    length_penalty: float,
    max_generation_delta: int,
) -> DecodingConfig:
    if decoding == "beam" or beam_size > 1:
        decoding_mode = BeamDecoding(
            beam_size=beam_size,
            length_penalty=length_penalty,
            max_generation_delta=max_generation_delta,
        )
    else:
        decoding_mode = GreedyDecoding()
    return DecodingConfig(decoding=decoding_mode)


@app.command()
def main(
    audio_files: List[Path] = typer.Argument(..., exists=True),
    model: str = typer.Option(..., "--model", envvar="CANARY_MODEL"),
    source_lang: str = typer.Option(..., "--source-lang"),
    target_lang: str = typer.Option(..., "--target-lang"),
    task: str = typer.Option("transcribe", "--task"),
    pnc: bool = typer.Option(True, "--pnc/--no-pnc"),
    chunk_duration: Optional[float] = typer.Option(None, "--chunk-duration"),
    overlap_duration: float = typer.Option(15.0, "--overlap-duration"),
    decoding: str = typer.Option("beam", "--decoding"),
    beam_size: int = typer.Option(1, "--beam-size"),
    length_penalty: float = typer.Option(0.0, "--length-penalty"),
    max_generation_delta: int = typer.Option(50, "--max-generation-delta"),
    max_new_tokens: Optional[int] = typer.Option(None, "--max-new-tokens"),
):
    model = from_pretrained(model)
    decoding_config = _build_decoding_config(
        decoding=decoding,
        beam_size=beam_size,
        length_penalty=length_penalty,
        max_generation_delta=max_generation_delta,
    )

    for audio_file in audio_files:
        result = model.transcribe(
            audio_file,
            source_lang=source_lang,
            target_lang=target_lang,
            task=task,
            pnc=pnc,
            chunk_duration=chunk_duration,
            overlap_duration=overlap_duration,
            max_new_tokens=max_new_tokens,
            decoding_config=decoding_config,
        )
        typer.echo(f"{audio_file}: {result.text}")


@app.command()
def batch(
    inputs: List[Path] = typer.Argument(...),
    model: str = typer.Option(..., "--model", envvar="CANARY_MODEL"),
    source_lang: str = typer.Option(..., "--source-lang"),
    target_lang: str = typer.Option(..., "--target-lang"),
    task: str = typer.Option("transcribe", "--task"),
    pnc: bool = typer.Option(True, "--pnc/--no-pnc"),
    output: Path = typer.Option(Path("transcripts.csv"), "--output"),
    recursive: bool = typer.Option(False, "--recursive"),
    chunk_duration: Optional[float] = typer.Option(None, "--chunk-duration"),
    overlap_duration: float = typer.Option(15.0, "--overlap-duration"),
    decoding: str = typer.Option("beam", "--decoding"),
    beam_size: int = typer.Option(1, "--beam-size"),
    length_penalty: float = typer.Option(0.0, "--length-penalty"),
    max_generation_delta: int = typer.Option(50, "--max-generation-delta"),
    max_new_tokens: Optional[int] = typer.Option(None, "--max-new-tokens"),
):
    model = from_pretrained(model)
    decoding_config = _build_decoding_config(
        decoding=decoding,
        beam_size=beam_size,
        length_penalty=length_penalty,
        max_generation_delta=max_generation_delta,
    )

    audio_files = _collect_audio_paths(inputs, recursive=recursive)
    if not audio_files:
        raise typer.BadParameter("No audio files found.")

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["path", "text"])
        for audio_file in audio_files:
            result = model.transcribe(
                audio_file,
                source_lang=source_lang,
                target_lang=target_lang,
                task=task,
                pnc=pnc,
                chunk_duration=chunk_duration,
                overlap_duration=overlap_duration,
                max_new_tokens=max_new_tokens,
                decoding_config=decoding_config,
            )
            writer.writerow([str(audio_file), result.text])
            typer.echo(f"{audio_file}: {result.text}")


if __name__ == "__main__":
    app()
