from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import sentencepiece as spm

CANARY_BOS = "<|startoftranscript|>"
CANARY_EOS = "<|endoftext|>"
CANARY_PAD = "<pad>"
CANARY_NOSPEECH = "<|nospeech|>"
CANARY_PNC = "<|pnc|>"
CANARY_NOPNC = "<|nopnc|>"
CANARY2_BOCTX = "<|startofcontext|>"
CANARY_SPECIAL_TOKENIZER = "spl_tokens"

DEFAULT_TOKENS = [CANARY_NOSPEECH, CANARY_PAD, CANARY_EOS, CANARY_BOS, CANARY_PNC, CANARY_NOPNC]


def _map_canary1_to_canary2_lang(lang: str, available_langs: List[str]) -> str:
    if available_langs == [CANARY_SPECIAL_TOKENIZER]:
        return CANARY_SPECIAL_TOKENIZER
    if len(lang) != 2 or lang in available_langs:
        return lang

    mapped = {"en": "en-US", "es": "es-ES", "fr": "fr-FR", "de": "de-DE"}.get(lang)
    if mapped is not None and mapped in available_langs:
        return mapped

    raise RuntimeError(
        f"Unsupported language: '{lang}' for CanaryTokenizer with languages: {available_langs}"
    )


class SentencePieceTokenizer:
    def __init__(self, model_path: Path):
        self.model_path = Path(model_path)
        self.processor = spm.SentencePieceProcessor(model_file=str(self.model_path))
        self.vocab = [self.processor.id_to_piece(i) for i in range(self.processor.get_piece_size())]
        self.vocab_size = len(self.vocab)

    def text_to_ids(self, text: str) -> List[int]:
        return list(self.processor.encode(text, out_type=int))

    def ids_to_tokens(self, ids: List[int]) -> List[str]:
        return [self.processor.id_to_piece(i) for i in ids]

    def ids_to_text(self, ids: List[int]) -> str:
        return self.processor.decode(ids)

    def token_to_id(self, token: str) -> int:
        return int(self.processor.piece_to_id(token))

    @property
    def bos(self) -> int:
        return int(self.processor.bos_id())

    @property
    def eos(self) -> int:
        return int(self.processor.eos_id())


class AggregateTokenizer:
    def __init__(self, tokenizers: Dict[str, SentencePieceTokenizer]):
        self.tokenizers_dict = tokenizers
        self.vocabulary: List[str] = []
        self.token_id_offset: Dict[str, int] = {}
        self.token_id_offset_by_tokenizer_num: Dict[int, int] = {}

        offset = 0
        for idx, (lang, tok) in enumerate(self.tokenizers_dict.items()):
            self.token_id_offset[lang] = offset
            self.token_id_offset_by_tokenizer_num[idx] = offset
            offset += len(tok.vocab)

        for tokenizer in self.tokenizers_dict.values():
            self.vocabulary.extend(tokenizer.vocab)

        self.vocab_size = len(self.vocabulary)
        (
            self.offset_token_ids_by_token_id,
            self.tokenizers_by_token_id,
            self.langs_by_token_id,
        ) = self._calculate_offsets()

    def _calculate_offsets(self):
        offsets: Dict[int, int] = {}
        tokenizers: Dict[int, SentencePieceTokenizer] = {}
        langs: Dict[int, str] = {}
        cur_num = 0
        langs_list = list(self.tokenizers_dict.keys())
        offsets_list = list(self.token_id_offset.values())

        for idx in range(len(self.vocabulary)):
            off_id = idx - offsets_list[cur_num]
            if cur_num + 1 < len(langs_list):
                if idx >= offsets_list[cur_num + 1]:
                    cur_num += 1
                    off_id = idx - offsets_list[cur_num]
            offsets[idx] = off_id
            tokenizers[idx] = list(self.tokenizers_dict.values())[cur_num]
            langs[idx] = langs_list[cur_num]

        return offsets, tokenizers, langs

    def text_to_ids(self, text: str, lang_id: str) -> List[int]:
        tokenizer = self.tokenizers_dict[lang_id]
        token_ids = tokenizer.text_to_ids(text)
        return [t + self.token_id_offset[lang_id] for t in token_ids]

    def ids_to_text(self, ids: List[int]) -> str:
        tokens = []
        for idx in ids:
            offset_id = self.offset_token_ids_by_token_id[idx]
            tokenizer = self.tokenizers_by_token_id[idx]
            tokens.extend(tokenizer.ids_to_tokens([offset_id]))
        return "".join(tokens).replace("\u2581", " ")

    def token_to_id(self, token: str, lang_id: str) -> int:
        tokenizer = self.tokenizers_dict[lang_id]
        return tokenizer.token_to_id(token) + self.token_id_offset[lang_id]

    @property
    def vocab(self) -> List[str]:
        return self.vocabulary

    @property
    def langs(self) -> List[str]:
        return list(self.tokenizers_dict.keys())


class CanaryTokenizer(AggregateTokenizer):
    def __init__(self, tokenizers: Dict[str, SentencePieceTokenizer]):
        super().__init__(tokenizers)
        self.special_tokens: Dict[str, int] = {}
        if CANARY_SPECIAL_TOKENIZER not in tokenizers:
            raise ValueError(f"Missing required tokenizer: {CANARY_SPECIAL_TOKENIZER}")
        for special in tokenizers[CANARY_SPECIAL_TOKENIZER].vocab:
            if (special.startswith("<|") and special.endswith("|>")) or special == CANARY_PAD:
                self.special_tokens[special] = self.token_to_id(
                    special, lang_id=CANARY_SPECIAL_TOKENIZER
                )

    @property
    def eos_id(self) -> int:
        return self.special_tokens[CANARY_EOS]

    @property
    def bos_id(self) -> int:
        return self.special_tokens[CANARY_BOS]

    @property
    def nospeech_id(self) -> int:
        return self.special_tokens[CANARY_NOSPEECH]

    @property
    def pad_id(self) -> int:
        return self.special_tokens[CANARY_PAD]

    def text_to_ids(self, text: str, lang_id: str) -> List[int]:
        if lang_id == CANARY_SPECIAL_TOKENIZER:
            return self._tokenize_special_prompt(text)
        lang_id = _map_canary1_to_canary2_lang(lang_id, self.langs)
        if text.endswith(CANARY_EOS):
            ids = super().text_to_ids(text[: -len(CANARY_EOS)], lang_id)
            return ids + [self.eos_id]
        return super().text_to_ids(text, lang_id)

    def _tokenize_special_prompt(self, text: str) -> List[int]:
        ans = []
        if text.startswith(CANARY2_BOCTX):
            ans.append(self.special_tokens[CANARY2_BOCTX])
            text = text[len(CANARY2_BOCTX) :]
            ctx_end_idx = text.find(CANARY_BOS)
            if ctx_end_idx != -1 and ctx_end_idx > 0:
                target_lang = text.split("<|")[4].replace("|>", "")
                ans.extend(self.text_to_ids(text[:ctx_end_idx], target_lang))
                text = text[ctx_end_idx:]

        while text:
            end_idx = text.find(">")
            if end_idx == -1:
                raise ValueError(f"Invalid special prompt: {text}")
            token = text[: end_idx + 1]
            if token not in self.special_tokens:
                raise KeyError(f"Token {token} not found in tokenizer.")
            ans.append(self.special_tokens[token])
            text = text[len(token) :]
        return ans

    def decode(self, ids: List[int]) -> str:
        return self.ids_to_text(ids)

    def build_prompt(
        self,
        source_lang: str,
        target_lang: str,
        task: str,
        pnc: bool | str = True,
        prompt_format: str = "canary",
    ) -> List[int]:
        if prompt_format not in ("canary", "canary2"):
            raise ValueError(f"Unsupported prompt format: {prompt_format}")

        def _normalize_lang(lang: str) -> str:
            if lang.startswith("<|") and lang.endswith("|>"):
                return lang
            return f"<|{lang}|>"

        def _normalize_task(value: str) -> str:
            if value.startswith("<|") and value.endswith("|>"):
                return value
            if value in {"translate", "ast", "s2t_translation"}:
                return "<|translate|>"
            return "<|transcribe|>"

        def _normalize_pnc(value: bool | str) -> str:
            if isinstance(value, str) and value.startswith("<|") and value.endswith("|>"):
                return value
            return "<|pnc|>" if str(value).lower() in ("true", "1", "yes", "pnc") else "<|nopnc|>"

        src = _normalize_lang(source_lang)
        tgt = _normalize_lang(target_lang)
        pnc_token = _normalize_pnc(pnc)

        if prompt_format == "canary2":
            prompt = (
                f"{CANARY2_BOCTX}{CANARY_BOS}"
                f"<|emo:undefined|>{src}{tgt}{pnc_token}"
                f"<|noitn|><|notimestamp|><|nodiarize|>"
            )
        else:
            task_token = _normalize_task(task)
            prompt = f"{CANARY_BOS}{src}{task_token}{tgt}{pnc_token}"
        return self.text_to_ids(prompt, CANARY_SPECIAL_TOKENIZER)


def _resolve_tokenizer_model_path(tokenizer_dir: Path) -> Path:
    if tokenizer_dir.is_file():
        return tokenizer_dir

    candidate = tokenizer_dir / "tokenizer.model"
    if candidate.exists():
        return candidate

    models = list(tokenizer_dir.glob("*.model"))
    if models:
        return models[0]

    for candidate in tokenizer_dir.rglob("tokenizer.model"):
        return candidate

    raise FileNotFoundError(f"No SentencePiece model found under {tokenizer_dir}")


@dataclass
class TokenizerConfig:
    langs: Dict[str, Dict[str, str]]


def load_tokenizer_from_config(config: Dict, base_dir: Path) -> CanaryTokenizer:
    tokenizer_cfg = config.get("tokenizer") or config.get("model", {}).get("tokenizer")
    if tokenizer_cfg is None:
        raise ValueError("Tokenizer config not found in model config.")

    model_path = tokenizer_cfg.get("model_path")
    if model_path:
        if isinstance(model_path, str) and model_path.startswith("nemo:"):
            model_path = model_path.split("nemo:", 1)[1]
        model_path = Path(model_path)
        if not model_path.is_absolute():
            model_path = base_dir / model_path
        tokenizer = SentencePieceTokenizer(_resolve_tokenizer_model_path(model_path))
        return CanaryTokenizer({CANARY_SPECIAL_TOKENIZER: tokenizer})

    langs = tokenizer_cfg.get("langs") or {}
    tokenizers: Dict[str, SentencePieceTokenizer] = {}

    for lang, lang_cfg in langs.items():
        lang_dir = lang_cfg.get("dir")
        if lang_dir is None:
            raise ValueError(f"Tokenizer dir not provided for language {lang}.")
        lang_path = Path(lang_dir)
        if not lang_path.is_absolute():
            lang_path = base_dir / lang_path
        model_path = _resolve_tokenizer_model_path(lang_path)
        tokenizers[lang] = SentencePieceTokenizer(model_path)

    return CanaryTokenizer(tokenizers)
