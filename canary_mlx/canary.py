from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional, Union

import mlx.core as mx
import mlx.nn as nn

from canary_mlx.alignment import (
    AlignedResult,
    AlignedToken,
    SentenceConfig,
    sentences_to_result,
    tokens_to_sentences,
)
from canary_mlx.audio import PreprocessArgs, get_logmel, load_audio
from canary_mlx.conformer import Conformer, ConformerArgs
from canary_mlx.tokenizer import CanaryTokenizer
from canary_mlx.transformer import (
    TokenClassifier,
    TokenClassifierConfig,
    TransformerDecoder,
    TransformerEmbedding,
    form_attention_mask,
    form_encoder_attention_mask,
    lengths_to_mask,
)


@dataclass
class TransformerDecoderConfig:
    hidden_size: int
    inner_size: int
    num_layers: int
    num_attention_heads: int
    max_sequence_length: int = 512
    num_token_types: int = 0
    embedding_dropout: float = 0.0
    learn_positional_encodings: bool = False
    ffn_dropout: float = 0.0
    attn_score_dropout: float = 0.0
    attn_layer_dropout: float = 0.0
    hidden_act: str = "relu"
    pre_ln: bool = True
    pre_ln_final_layer_norm: bool = True


@dataclass
class GreedyDecoding:
    pass


@dataclass
class BeamDecoding:
    beam_size: int = 1
    length_penalty: float = 0.0
    max_generation_delta: int = 50


@dataclass
class DecodingConfig:
    decoding: Union[GreedyDecoding, BeamDecoding] = field(default_factory=BeamDecoding)
    sentence: SentenceConfig = field(default_factory=SentenceConfig)


class CanaryModel(nn.Module):
    def __init__(
        self,
        *,
        preprocessor: PreprocessArgs,
        encoder: ConformerArgs,
        decoder: TransformerDecoderConfig,
        classifier: TokenClassifierConfig,
        tokenizer: CanaryTokenizer,
        prompt_format: str = "canary",
        encoder_decoder_proj: Optional[int] = None,
    ):
        super().__init__()
        self.preprocessor_config = preprocessor
        self.encoder_config = encoder
        self.decoder_config = decoder
        self.classifier_config = classifier
        self.tokenizer = tokenizer
        self.prompt_format = prompt_format

        self.encoder = Conformer(encoder)
        if encoder_decoder_proj is not None:
            self.encoder_decoder_proj = nn.Linear(encoder.d_model, encoder_decoder_proj)
        else:
            self.encoder_decoder_proj = nn.Identity()

        self.decoder_embedding = TransformerEmbedding(
            vocab_size=classifier.num_classes,
            hidden_size=decoder.hidden_size,
            max_sequence_length=decoder.max_sequence_length,
            num_token_types=decoder.num_token_types,
            embedding_dropout=decoder.embedding_dropout,
            learn_positional_encodings=decoder.learn_positional_encodings,
        )
        self.decoder = TransformerDecoder(
            num_layers=decoder.num_layers,
            hidden_size=decoder.hidden_size,
            inner_size=decoder.inner_size,
            num_attention_heads=decoder.num_attention_heads,
            attn_score_dropout=decoder.attn_score_dropout,
            attn_layer_dropout=decoder.attn_layer_dropout,
            ffn_dropout=decoder.ffn_dropout,
            hidden_act=decoder.hidden_act,
            pre_ln=decoder.pre_ln,
            pre_ln_final_layer_norm=decoder.pre_ln_final_layer_norm,
        )
        self.classifier = TokenClassifier(classifier)

    @property
    def time_ratio(self) -> float:
        return (
            self.encoder_config.subsampling_factor
            / self.preprocessor_config.sample_rate
            * self.preprocessor_config.hop_length
        )

    def encode(self, mel: mx.array) -> tuple[mx.array, mx.array]:
        lengths = mx.full((mel.shape[0],), mel.shape[1], dtype=mx.int64)
        enc_out, enc_len = self.encoder(mel, lengths)
        enc_out = self.encoder_decoder_proj(enc_out)
        return enc_out, enc_len

    def decode(self, input_ids: mx.array, enc_out: mx.array, enc_len: mx.array) -> mx.array:
        dec_mask = form_attention_mask(mx.ones_like(input_ids), diagonal=0)
        enc_mask = form_encoder_attention_mask(lengths_to_mask(enc_len, enc_out.shape[1]))
        dec_embed = self.decoder_embedding(input_ids)
        dec_hidden = self.decoder(dec_embed, dec_mask, enc_out, enc_mask)
        return self.classifier(dec_hidden)

    def _decode_step(self, input_ids: mx.array, enc_out: mx.array, enc_len: mx.array) -> mx.array:
        logits = self.decode(input_ids, enc_out, enc_len)
        step_logits = logits[0, -1]
        if not self.classifier_config.log_softmax:
            step_logits = nn.log_softmax(step_logits, axis=-1)
        return step_logits

    def _resolve_max_steps(
        self,
        prompt_len: int,
        max_new_tokens: Optional[int],
        max_generation_delta: Optional[int],
    ) -> int:
        if max_new_tokens is not None:
            max_steps = max_new_tokens
        elif max_generation_delta is not None:
            max_steps = max_generation_delta
        else:
            max_steps = self.decoder_config.max_sequence_length - prompt_len
        max_total = max(1, self.decoder_config.max_sequence_length - prompt_len)
        return max(1, min(max_steps, max_total))

    def _score_with_length_penalty(self, score: float, length: int, penalty: float) -> float:
        if penalty <= 0.0:
            return score
        return score / (float(max(1, length)) ** penalty)

    def _beam_search(
        self,
        *,
        enc_out: mx.array,
        enc_len: mx.array,
        prompt_ids: List[int],
        beam_config: BeamDecoding,
        max_steps: int,
    ) -> List[int]:
        eos_id = self.tokenizer.eos_id
        beam_size = max(1, beam_config.beam_size)

        beams: List[tuple[List[int], float, bool]] = [(list(prompt_ids), 0.0, False)]
        for _ in range(max_steps):
            candidates: List[tuple[List[int], float, bool]] = []
            for tokens, score, ended in beams:
                if ended:
                    candidates.append((tokens, score, True))
                    continue
                input_ids = mx.array([tokens], dtype=mx.int32)
                log_probs = self._decode_step(input_ids, enc_out, enc_len)
                topk = mx.argpartition(log_probs, -beam_size)[-beam_size:]
                topk = [int(tok) for tok in topk.tolist()]
                for tok in topk:
                    new_score = score + float(log_probs[tok])
                    candidates.append((tokens + [tok], new_score, tok == eos_id))

            candidates.sort(
                key=lambda item: self._score_with_length_penalty(
                    item[1], len(item[0]), beam_config.length_penalty
                ),
                reverse=True,
            )
            beams = candidates[:beam_size]
            if all(ended for _, _, ended in beams):
                break

        finished = [b for b in beams if b[2]]
        if finished:
            finished.sort(
                key=lambda item: self._score_with_length_penalty(
                    item[1], len(item[0]), beam_config.length_penalty
                ),
                reverse=True,
            )
            return finished[0][0]
        return beams[0][0]

    def generate(
        self,
        mel: mx.array,
        *,
        source_lang: str,
        target_lang: str,
        task: str = "transcribe",
        pnc: bool | str = True,
        max_new_tokens: Optional[int] = None,
        decoding_config: Union[DecodingConfig, SentenceConfig] = DecodingConfig(),
    ) -> List[AlignedResult]:
        if mel.ndim == 2:
            mel = mx.expand_dims(mel, 0)

        if mel.shape[0] != 1:
            raise ValueError("Only batch size 1 is supported for now.")

        enc_out, enc_len = self.encode(mel)
        prompt_ids = self.tokenizer.build_prompt(
            source_lang=source_lang,
            target_lang=target_lang,
            task=task,
            pnc=pnc,
            prompt_format=self.prompt_format,
        )
        prompt_len = len(prompt_ids)
        if isinstance(decoding_config, SentenceConfig):
            decoding_config = DecodingConfig(sentence=decoding_config)

        max_steps = self._resolve_max_steps(
            prompt_len,
            max_new_tokens,
            decoding_config.decoding.max_generation_delta
            if isinstance(decoding_config.decoding, BeamDecoding)
            else None,
        )

        if isinstance(decoding_config.decoding, BeamDecoding) and decoding_config.decoding.beam_size > 1:
            generated = self._beam_search(
                enc_out=enc_out,
                enc_len=enc_len,
                prompt_ids=prompt_ids,
                beam_config=decoding_config.decoding,
                max_steps=max_steps,
            )
        else:
            generated = list(prompt_ids)
            eos_id = self.tokenizer.eos_id
            for _ in range(max_steps):
                input_ids = mx.array([generated], dtype=mx.int32)
                step_logits = self._decode_step(input_ids, enc_out, enc_len)
                next_id = int(mx.argmax(step_logits, axis=-1))
                generated.append(next_id)
                if next_id == eos_id:
                    break

        decoded_ids = generated[prompt_len:]
        decoded_ids = [idx for idx in decoded_ids if idx not in self.tokenizer.special_tokens.values()]
        text = self.tokenizer.decode(decoded_ids)

        tokens = []
        for idx in decoded_ids:
            token_text = self.tokenizer.decode([idx])
            tokens.append(AlignedToken(id=idx, text=token_text, start=0.0, duration=0.0, confidence=1.0))

        result = sentences_to_result(tokens_to_sentences(tokens, decoding_config.sentence))
        result.text = text
        return [result]

    def transcribe(
        self,
        path: Path | str,
        *,
        dtype: mx.Dtype = mx.bfloat16,
        source_lang: str,
        target_lang: str,
        task: str = "transcribe",
        pnc: bool | str = True,
        chunk_duration: Optional[float] = None,
        overlap_duration: float = 15.0,
        chunk_callback: Optional[Callable] = None,
        max_new_tokens: Optional[int] = None,
        decoding_config: Union[DecodingConfig, SentenceConfig] = DecodingConfig(),
    ) -> AlignedResult:
        audio_path = Path(path)
        audio_data = load_audio(audio_path, self.preprocessor_config.sample_rate, dtype)

        if chunk_duration is None:
            mel = get_logmel(audio_data, self.preprocessor_config)
            return self.generate(
                mel,
                source_lang=source_lang,
                target_lang=target_lang,
                task=task,
                pnc=pnc,
                max_new_tokens=max_new_tokens,
                decoding_config=decoding_config,
            )[0]

        audio_length_seconds = len(audio_data) / self.preprocessor_config.sample_rate
        if audio_length_seconds <= chunk_duration:
            mel = get_logmel(audio_data, self.preprocessor_config)
            return self.generate(
                mel,
                source_lang=source_lang,
                target_lang=target_lang,
                task=task,
                pnc=pnc,
                max_new_tokens=max_new_tokens,
                decoding_config=decoding_config,
            )[0]

        chunk_samples = int(chunk_duration * self.preprocessor_config.sample_rate)
        overlap_samples = int(overlap_duration * self.preprocessor_config.sample_rate)

        all_tokens: List[AlignedToken] = []
        for start in range(0, len(audio_data), chunk_samples - overlap_samples):
            end = min(start + chunk_samples, len(audio_data))
            if chunk_callback is not None:
                chunk_callback(end, len(audio_data))
            if end - start < self.preprocessor_config.hop_length:
                break
            chunk_audio = audio_data[start:end]
            chunk_mel = get_logmel(chunk_audio, self.preprocessor_config)
            chunk_result = self.generate(
                chunk_mel,
                source_lang=source_lang,
                target_lang=target_lang,
                task=task,
                pnc=pnc,
                max_new_tokens=max_new_tokens,
                decoding_config=decoding_config,
            )[0]
            all_tokens.extend(chunk_result.tokens)

        if isinstance(decoding_config, SentenceConfig):
            sentence_config = decoding_config
        else:
            sentence_config = decoding_config.sentence
        return sentences_to_result(tokens_to_sentences(all_tokens, sentence_config))
