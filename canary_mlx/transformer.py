import math
from dataclasses import dataclass
from typing import Optional

import mlx.core as mx
import mlx.nn as nn

NEG_INF = -10000.0


def form_attention_mask(input_mask: Optional[mx.array], diagonal: Optional[int] = None) -> Optional[mx.array]:
    if input_mask is None:
        return None

    input_mask = input_mask.astype(mx.bool_)
    attn_mask = mx.expand_dims(input_mask, 1)

    if diagonal is not None:
        length = input_mask.shape[1]
        future = mx.tril(mx.ones((1, length, length), dtype=mx.bool_), diagonal)
        attn_mask = mx.logical_and(attn_mask, future)

    attention_mask = (1 - attn_mask.astype(mx.float32)) * NEG_INF
    return mx.expand_dims(attention_mask, 1)


def form_encoder_attention_mask(input_mask: Optional[mx.array]) -> Optional[mx.array]:
    if input_mask is None:
        return None
    input_mask = input_mask.astype(mx.bool_)
    attention_mask = (1 - input_mask.astype(mx.float32)) * NEG_INF
    return attention_mask[:, None, None, :]


def lengths_to_mask(lengths: mx.array, max_len: Optional[int] = None) -> mx.array:
    if max_len is None:
        max_len = int(mx.max(lengths))
    row = mx.arange(max_len)[None, :]
    return row < lengths[:, None]


class FixedPositionalEncoding(nn.Module):
    def __init__(self, hidden_size: int, max_sequence_length: int = 512):
        super().__init__()
        self.hidden_size = hidden_size
        self.max_sequence_length = max_sequence_length
        self.pos_enc = self._build_pos_enc(hidden_size, max_sequence_length)

    def _build_pos_enc(self, hidden_size: int, max_sequence_length: int) -> mx.array:
        pos = mx.arange(max_sequence_length)[:, None]
        div_term = mx.exp(
            mx.arange(0, hidden_size, 2) * (-math.log(10000.0) / hidden_size)
        )
        pos_enc = mx.zeros((max_sequence_length, hidden_size))
        pos_enc[:, 0::2] = mx.sin(pos * div_term)
        pos_enc[:, 1::2] = mx.cos(pos * div_term)
        pos_enc = pos_enc / math.sqrt(hidden_size)
        return pos_enc

    def __call__(self, position_ids: mx.array) -> mx.array:
        return self.pos_enc[position_ids]


class TransformerEmbedding(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        max_sequence_length: int = 512,
        num_token_types: int = 0,
        embedding_dropout: float = 0.0,
        learn_positional_encodings: bool = False,
    ):
        super().__init__()
        self.max_sequence_length = max_sequence_length
        self.learn_positional_encodings = learn_positional_encodings
        self.token_embedding = nn.Embedding(vocab_size, hidden_size)
        if learn_positional_encodings:
            self.position_embedding = nn.Embedding(max_sequence_length, hidden_size)
        else:
            self.position_embedding = FixedPositionalEncoding(hidden_size, max_sequence_length)
        self.token_type_embedding = (
            nn.Embedding(num_token_types, hidden_size) if num_token_types > 0 else None
        )
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(embedding_dropout)

    def __call__(self, input_ids: mx.array, token_type_ids: Optional[mx.array] = None, start_pos: int = 0) -> mx.array:
        seq_len = input_ids.shape[1]
        if self.learn_positional_encodings and seq_len > self.max_sequence_length:
            raise ValueError(
                "Input sequence is longer than maximum allowed sequence length for positional encoding. "
                f"Got {seq_len} and {self.max_sequence_length}"
            )

        position_ids = mx.arange(start_pos, start_pos + seq_len)[None, :]
        position_ids = mx.broadcast_to(position_ids, input_ids.shape)

        token_embeddings = self.token_embedding(input_ids)
        position_embeddings = self.position_embedding(position_ids)
        embeddings = token_embeddings + position_embeddings

        if token_type_ids is not None and self.token_type_embedding is not None:
            embeddings = embeddings + self.token_type_embedding(token_type_ids)

        embeddings = self.layer_norm(embeddings)
        return self.dropout(embeddings)


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, bias: bool = True):
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError(
                "The hidden size must be a multiple of the number of heads."
            )
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim**-0.5

        self.query_net = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.key_net = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.value_net = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.out_projection = nn.Linear(hidden_size, hidden_size, bias=bias)

    def __call__(
        self,
        query: mx.array,
        key: mx.array,
        value: mx.array,
        attention_mask: Optional[mx.array] = None,
    ) -> mx.array:
        q = self.query_net(query)
        k = self.key_net(key)
        v = self.value_net(value)

        batch, q_len, _ = q.shape
        _, k_len, _ = k.shape

        q = q.reshape(batch, q_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(batch, k_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(batch, k_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

        scores = mx.matmul(q, k.transpose(0, 1, 3, 2)) * self.scale
        if attention_mask is not None:
            if attention_mask.ndim == 3:
                attention_mask = attention_mask[:, None, :, :]
            scores = scores + attention_mask
        attn = mx.softmax(scores, axis=-1)
        out = mx.matmul(attn, v)
        out = out.transpose(0, 2, 1, 3).reshape(batch, q_len, -1)

        return self.out_projection(out)


class PositionWiseFF(nn.Module):
    def __init__(self, hidden_size: int, inner_size: int, activation: str = "relu"):
        super().__init__()
        self.linear1 = nn.Linear(hidden_size, inner_size)
        self.linear2 = nn.Linear(inner_size, hidden_size)
        if activation == "gelu":
            self.activation = nn.GELU()
        else:
            self.activation = nn.ReLU()

    def __call__(self, x: mx.array) -> mx.array:
        return self.linear2(self.activation(self.linear1(x)))


class TransformerDecoderBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        inner_size: int,
        num_attention_heads: int,
        attn_score_dropout: float = 0.0,
        attn_layer_dropout: float = 0.0,
        ffn_dropout: float = 0.0,
        hidden_act: str = "relu",
        pre_ln: bool = False,
    ):
        super().__init__()
        self.pre_ln = pre_ln
        self.layer_norm_1 = nn.LayerNorm(hidden_size)
        self.first_sub_layer = MultiHeadAttention(hidden_size, num_attention_heads)
        self.layer_norm_2 = nn.LayerNorm(hidden_size)
        self.second_sub_layer = MultiHeadAttention(hidden_size, num_attention_heads)
        self.layer_norm_3 = nn.LayerNorm(hidden_size)
        self.third_sub_layer = PositionWiseFF(hidden_size, inner_size, hidden_act)
        self.dropout_attn = nn.Dropout(attn_layer_dropout)
        self.dropout_ffn = nn.Dropout(ffn_dropout)

    def _self_attention(self, decoder_query, decoder_mask):
        output = self.first_sub_layer(
            decoder_query, decoder_query, decoder_query, attention_mask=decoder_mask
        )
        return self.dropout_attn(output)

    def _cross_attention(self, decoder_query, encoder_states, encoder_mask):
        output = self.second_sub_layer(
            decoder_query, encoder_states, encoder_states, attention_mask=encoder_mask
        )
        return self.dropout_attn(output)

    def forward_preln(
        self, decoder_query, decoder_mask, encoder_states, encoder_mask
    ):
        residual = decoder_query
        decoder_query = self.layer_norm_1(decoder_query)
        self_attn_output = self._self_attention(decoder_query, decoder_mask)
        self_attn_output = self_attn_output + residual

        residual = self_attn_output
        self_attn_output = self.layer_norm_2(self_attn_output)
        cross_attn_output = self._cross_attention(
            self_attn_output, encoder_states, encoder_mask
        )
        cross_attn_output = cross_attn_output + residual

        residual = cross_attn_output
        cross_attn_output = self.layer_norm_3(cross_attn_output)
        output_states = self.third_sub_layer(cross_attn_output)
        output_states = self.dropout_ffn(output_states) + residual

        return output_states

    def forward_postln(
        self, decoder_query, decoder_mask, encoder_states, encoder_mask
    ):
        self_attn_output = self._self_attention(decoder_query, decoder_mask)
        self_attn_output = self_attn_output + decoder_query
        self_attn_output = self.layer_norm_1(self_attn_output)

        cross_attn_output = self._cross_attention(
            self_attn_output, encoder_states, encoder_mask
        )
        cross_attn_output = cross_attn_output + self_attn_output
        cross_attn_output = self.layer_norm_2(cross_attn_output)

        output_states = self.third_sub_layer(cross_attn_output)
        output_states = self.dropout_ffn(output_states) + cross_attn_output
        return self.layer_norm_3(output_states)

    def __call__(
        self, decoder_query, decoder_mask, encoder_states, encoder_mask
    ):
        if self.pre_ln:
            return self.forward_preln(
                decoder_query, decoder_mask, encoder_states, encoder_mask
            )
        return self.forward_postln(
            decoder_query, decoder_mask, encoder_states, encoder_mask
        )


class TransformerDecoder(nn.Module):
    def __init__(
        self,
        num_layers: int,
        hidden_size: int,
        inner_size: int,
        num_attention_heads: int = 1,
        attn_score_dropout: float = 0.0,
        attn_layer_dropout: float = 0.0,
        ffn_dropout: float = 0.0,
        hidden_act: str = "relu",
        pre_ln: bool = False,
        pre_ln_final_layer_norm: bool = True,
    ):
        super().__init__()
        self.d_model = hidden_size
        self.final_layer_norm = (
            nn.LayerNorm(hidden_size)
            if pre_ln and pre_ln_final_layer_norm
            else None
        )
        self.layers = [
            TransformerDecoderBlock(
                hidden_size,
                inner_size,
                num_attention_heads,
                attn_score_dropout,
                attn_layer_dropout,
                ffn_dropout,
                hidden_act,
                pre_ln,
            )
            for _ in range(num_layers)
        ]

    def __call__(
        self,
        decoder_states: mx.array,
        decoder_mask: Optional[mx.array],
        encoder_states: mx.array,
        encoder_mask: Optional[mx.array],
    ) -> mx.array:
        x = decoder_states
        for layer in self.layers:
            x = layer(x, decoder_mask, encoder_states, encoder_mask)
        if self.final_layer_norm is not None:
            x = self.final_layer_norm(x)
        return x


@dataclass
class TokenClassifierConfig:
    hidden_size: int
    num_classes: int
    num_layers: int = 1
    activation: str = "relu"
    log_softmax: bool = True
    dropout: float = 0.0


class TokenClassifier(nn.Module):
    def __init__(self, config: TokenClassifierConfig):
        super().__init__()
        self.config = config
        self.layers = []
        self.dropout = nn.Dropout(config.dropout)

        layers = []
        for _ in range(config.num_layers - 1):
            layers.append(nn.Linear(config.hidden_size, config.hidden_size))
            if config.activation == "gelu":
                layers.append(nn.GELU())
            else:
                layers.append(nn.ReLU())
        layers.append(nn.Linear(config.hidden_size, config.num_classes))

        for idx, layer in enumerate(layers):
            setattr(self, f"layer{idx}", layer)
        self.layers = layers

    def __call__(self, hidden_states: mx.array) -> mx.array:
        x = self.dropout(hidden_states)
        for layer in self.layers:
            x = layer(x)
        if self.config.log_softmax:
            x = nn.log_softmax(x, axis=-1)
        return x
