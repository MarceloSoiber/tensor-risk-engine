"""PyTorch sequence models for fraud detection."""

from __future__ import annotations

from typing import Literal, Sequence

import torch
from torch import Tensor, nn
from torch.nn.utils.rnn import pack_padded_sequence

__all__ = ["SequenceFraudModel"]


def _normalize_embedding_dims(
    cardinalities: Sequence[int],
    embedding_dims: Sequence[int] | int | None,
) -> list[int]:
    """Resolve embedding dimensions for each categorical feature."""

    if not cardinalities:
        return []

    if embedding_dims is None:
        return [min(50, max(4, (cardinality + 1) // 2)) for cardinality in cardinalities]

    if isinstance(embedding_dims, int):
        if embedding_dims <= 0:
            raise ValueError("embedding_dims must be positive.")
        return [embedding_dims for _ in cardinalities]

    dims = list(embedding_dims)
    if len(dims) != len(cardinalities):
        raise ValueError("embedding_dims must match the number of categorical features.")
    if any(d <= 0 for d in dims):
        raise ValueError("embedding_dims entries must be positive.")
    return dims


class SequenceFraudModel(nn.Module):
    """Sequence model for fraud prediction using GRU or LSTM backbones.

    The model encodes numeric features with a learned projection, embeds each
    categorical feature independently, and feeds the concatenated representation
    into a packed recurrent backbone. The final hidden state is passed through a
    classifier head that produces a single logit per sequence.
    """

    def __init__(
        self,
        *,
        num_numeric_features: int,
        categorical_cardinalities: Sequence[int] | None = None,
        embedding_dims: Sequence[int] | int | None = None,
        backbone: Literal["gru", "lstm"] = "gru",
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.1,
        bidirectional: bool = False,
        numeric_projection_dim: int = 32,
        classifier_hidden_dim: int | None = None,
    ) -> None:
        super().__init__()

        if num_numeric_features < 0:
            raise ValueError("num_numeric_features cannot be negative.")
        if hidden_size <= 0:
            raise ValueError("hidden_size must be positive.")
        if num_layers <= 0:
            raise ValueError("num_layers must be positive.")
        if numeric_projection_dim < 0:
            raise ValueError("numeric_projection_dim cannot be negative.")
        if num_numeric_features > 0 and numeric_projection_dim <= 0:
            raise ValueError("numeric_projection_dim must be positive when numeric features are used.")
        if classifier_hidden_dim is not None and classifier_hidden_dim <= 0:
            raise ValueError("classifier_hidden_dim must be positive when provided.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in the range [0.0, 1.0).")
        if backbone not in {"gru", "lstm"}:
            raise ValueError("backbone must be either 'gru' or 'lstm'.")

        self.num_numeric_features = int(num_numeric_features)
        self.categorical_cardinalities = list(categorical_cardinalities or [])
        if any(cardinality <= 1 for cardinality in self.categorical_cardinalities):
            raise ValueError("categorical_cardinalities must contain values greater than 1.")

        self.embedding_dims = _normalize_embedding_dims(self.categorical_cardinalities, embedding_dims)
        self.backbone_type = backbone
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        self.numeric_projection = (
            nn.Sequential(
                nn.Linear(self.num_numeric_features, numeric_projection_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            if self.num_numeric_features > 0 and numeric_projection_dim > 0
            else None
        )

        self.embeddings = nn.ModuleList(
            [
                nn.Embedding(cardinality, dim)
                for cardinality, dim in zip(self.categorical_cardinalities, self.embedding_dims)
            ]
        )

        input_size = 0
        if self.numeric_projection is not None:
            input_size += numeric_projection_dim
        input_size += sum(self.embedding_dims)
        if input_size <= 0:
            raise ValueError("At least one numeric or categorical feature must be provided.")

        rnn_dropout = dropout if num_layers > 1 else 0.0
        rnn_cls = nn.GRU if backbone == "gru" else nn.LSTM
        self.rnn = rnn_cls(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=rnn_dropout,
            batch_first=True,
            bidirectional=bidirectional,
        )

        rnn_output_size = hidden_size * (2 if bidirectional else 1)
        classifier_input_size = rnn_output_size
        classifier_hidden_dim = classifier_hidden_dim or max(hidden_size // 2, 1)

        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_size, classifier_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden_dim, 1),
        )

    def _validate_inputs(self, x_num: Tensor, x_cat: Tensor, lengths: Tensor) -> None:
        if x_num.ndim != 3:
            raise ValueError("x_num must have shape (batch, seq_len, num_numeric_features).")
        if x_cat.ndim != 3:
            raise ValueError("x_cat must have shape (batch, seq_len, num_categorical_features).")
        if lengths.ndim != 1:
            raise ValueError("lengths must be a one-dimensional tensor.")
        if x_num.shape[0] != x_cat.shape[0] or x_num.shape[0] != lengths.shape[0]:
            raise ValueError("x_num, x_cat, and lengths must share the same batch size.")
        if self.num_numeric_features > 0 and x_num.shape[-1] != self.num_numeric_features:
            raise ValueError("x_num feature dimension does not match model configuration.")
        if self.num_numeric_features == 0 and x_num.shape[-1] != 0:
            raise ValueError("x_num must have zero features when no numeric features are configured.")
        if self.categorical_cardinalities and x_cat.shape[-1] != len(self.categorical_cardinalities):
            raise ValueError("x_cat feature dimension does not match model configuration.")
        if not self.categorical_cardinalities and x_cat.shape[-1] != 0:
            raise ValueError("x_cat must have zero features when no categorical features are configured.")
        if torch.any(lengths <= 0):
            raise ValueError("lengths must contain positive values only.")
        if torch.any(lengths > x_num.shape[1]):
            raise ValueError("lengths cannot exceed the sequence length of the batch.")

    def _encode_numeric(self, x_num: Tensor) -> Tensor:
        if self.numeric_projection is None:
            batch_size, seq_len = x_num.shape[:2]
            return x_num.new_zeros((batch_size, seq_len, 0))
        return self.numeric_projection(x_num)

    def _encode_categorical(self, x_cat: Tensor) -> Tensor:
        if not self.embeddings:
            batch_size, seq_len = x_cat.shape[:2]
            return x_cat.new_zeros((batch_size, seq_len, 0), dtype=torch.float32)

        embeddings = []
        for index, embedding in enumerate(self.embeddings):
            feature = x_cat[..., index].long()
            if torch.any(feature < 0):
                raise ValueError("Categorical features must contain non-negative indices.")
            if torch.any(feature >= embedding.num_embeddings):
                raise ValueError("Categorical feature index exceeds embedding cardinality.")
            embeddings.append(embedding(feature))
        return torch.cat(embeddings, dim=-1)

    def _extract_last_hidden_state(self, hidden_state: Tensor | tuple[Tensor, Tensor]) -> Tensor:
        if self.backbone_type == "lstm":
            hidden_state = hidden_state[0]

        num_directions = 2 if self.bidirectional else 1
        hidden_state = hidden_state.view(self.num_layers, num_directions, hidden_state.shape[1], hidden_state.shape[2])
        last_layer = hidden_state[-1]
        if num_directions == 1:
            return last_layer.squeeze(0)
        return torch.cat([last_layer[0], last_layer[1]], dim=-1)

    def forward(self, x_num: Tensor, x_cat: Tensor, lengths: Tensor) -> Tensor:
        """Compute one logit per sequence."""

        self._validate_inputs(x_num, x_cat, lengths)

        lengths_cpu = lengths.to(dtype=torch.long, device=torch.device("cpu"))
        numeric_features = self._encode_numeric(x_num)
        categorical_features = self._encode_categorical(x_cat)
        rnn_input = torch.cat([numeric_features, categorical_features], dim=-1)

        packed = pack_padded_sequence(rnn_input, lengths_cpu, batch_first=True, enforce_sorted=False)
        _, hidden_state = self.rnn(packed)
        sequence_repr = self._extract_last_hidden_state(hidden_state)
        logits = self.classifier(sequence_repr).squeeze(-1)
        return logits
