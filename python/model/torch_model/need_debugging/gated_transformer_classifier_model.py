from random import triangular
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .base import BaseTorchModel, BaseTorchClassifier, BaseTorchTrainer
from .torch_utils import init_gpu

# from .base import TorchModelWrapper

class SinusoidPositionalEncoding(nn.Module):
    def __init__(self, embedding_dim: int, dropout: float = 0.1, max_sequential_length: int = 15000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_sequential_length).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embedding_dim, 2) * (-math.log(10000.0) / embedding_dim)
        )
        pe = torch.zeros(max_sequential_length, 1, embedding_dim)
        cos_size = embedding_dim // 2
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term[:cos_size])
        self.register_buffer("pe", pe)

    def forward(self, seq_len) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        return self.pe[: seq_len]
    

class GatedTransformerClassifier(torch.nn.Module, BaseTorchClassifier):
    def __init__(
        self,
        input_token_dim=None, # Forces user to specify dimension of each token. Number of features per token
        sequence_length=None, # Forces user to specify sequence length. Number of time steps (or tokens) in the time series
        desired_embedding_dim=None, # Only for embedding positional encoding. Length of encoding vector
        positional_encoding='embedding', # 'sinusoid' or 'embedding
        hidden_dim=512,  # Commonly used size in transformer models
        hidden_unit_dim=64,  # A smaller dimension for the hidden units
        hidden_unit_dim_time=64,  # Same as above, assuming similar scale
        num_heads_time=4,  # Multi-head attention typically uses 4 or 8 heads
        num_heads=4,  # Same as above
        kernel_size=3,  # Typical value for convolutions
        stride=1,  # Stride of 1 is more common than 3 for fine-grained processing
        n_class=2,  # Default for binary classification
        num_layers=6,  # A balance between complexity and performance
        num_layers_time=6,  # Same as above
        dropout=0.1,  # Standard dropout rate in transformer models
        conv_dropout=0.1,  # Usually the same as 'dropout'
        output_dropout=0.1,
    ):
        super().__init__()
        self.desired_embedding_dim = desired_embedding_dim
        self.input_token_dim = input_token_dim
        self.hidden_dim = hidden_dim
        self.hidden_sample = (self.input_token_dim - kernel_size) // stride + 1
        self.idx_sample = torch.arange(0, sequence_length, 1, dtype=torch.int).cuda()

        # positional encoding for the time encoder
        # Pick either sinusoid, if you have your own tokenization, or embedding, if you don't (e.g. categorical or word data)
        if positional_encoding == 'sinusoid':
            self.pos_encoder = SinusoidPositionalEncoding(input_token_dim, output_dropout)
        elif positional_encoding == 'embedding':
            self.pos_encoder = nn.Embedding(input_token_dim, desired_embedding_dim)
        
        self.dropout_input = nn.Dropout(output_dropout)

        # define the time encoding layers
        self.time_encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_token_dim,
            nhead=num_heads_time,
            dim_feedforward=hidden_unit_dim_time,
            dropout=dropout,
            activation="gelu",
        )
        self.time_encoder = nn.TransformerEncoder(
            self.time_encoder_layer, num_layers=num_layers_time
        )

        # define the channel encoding layers
        self.channel_encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_token_dim,
            nhead=num_heads,
            dim_feedforward=hidden_unit_dim,
            dropout=dropout,
            activation="gelu",
        )
        self.channel_encoder = nn.TransformerEncoder(
            self.channel_encoder_layer, num_layers=num_layers
        )

        # define conv and linear layers for downsampling the time attention output
        self.conv_time = nn.Conv1d(
            in_channels=input_token_dim,
            out_channels=hidden_dim,
            kernel_size=kernel_size,
            stride=stride,
        )
        self.dropout_time0 = nn.Dropout(conv_dropout)
        self.linear_time = nn.Linear(hidden_dim, hidden_dim)
        self.dropout_time1 = nn.Dropout(conv_dropout)
        self.linear_time1 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout_time2 = nn.Dropout(conv_dropout)

        # define conv and linear layers for downsampling the time attention output
        self.conv_space = nn.Conv1d(
            in_channels=input_token_dim,
            out_channels=hidden_dim,
            kernel_size=kernel_size,
            stride=stride,
        )
        self.dropout_space0 = nn.Dropout(conv_dropout)
        self.linear_space = nn.Linear(hidden_dim, hidden_dim)
        self.dropout_space1 = nn.Dropout(conv_dropout)
        self.linear_space1 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout_space2 = nn.Dropout(conv_dropout)

        # define linear function for gating
        self.linear_gate_time = nn.Linear(hidden_dim * self.hidden_sample, 1)
        self.linear_gate_channel = nn.Linear(hidden_dim * self.hidden_sample, 1)

        # define the output_layers
        self.dropout_output0 = nn.Dropout(output_dropout)
        self.linear_out = nn.Linear(
            hidden_dim * self.hidden_sample * 2, hidden_dim * self.hidden_sample
        )
        self.dropout_output1 = nn.Dropout(output_dropout)
        self.linear_out1 = nn.Linear(
            hidden_dim * self.hidden_sample, hidden_dim * self.hidden_sample
        )
        self.dropout_output2 = nn.Dropout(output_dropout)
        self.output_layer = nn.Linear(hidden_dim * self.hidden_sample, n_class)

    def forward(self, input):
        # input is in shape (sample, time, channel)
        # the time encoder is attending along channel axis
        # first transpose data to shape (channel, sample, time)
        x_time = input.contiguous().permute(2, 0, 1)
        x_time = self.time_encoder(x_time)

        # also transpose data to shape (time, sample, channel)
        # the channel encoder is attending along time axis
        x_channel = input.contiguous().permute(1, 0, 2)
        if 'positional_encoding' == 'embedding':
            x_pos = torch.unsqueeze(self.pos_encoder(x_channel.shape[-1]), dim=1)
        else:
            x_pos = self.pos_encoder(input.shape[-1]).permute(2,1,0)
        x_channel = self.dropout_input(x_channel + x_pos)
        x_channel = self.channel_encoder(x_channel.permute(1, 2, 0))

        # now transpose everything back to (sample, channel, time)
        x_time = x_time.permute(1, 2, 0)
        x_channel = x_channel.permute(1, 2, 0)

        # downsample and down project in time
        x_time = self.conv_time(x_time)
        x_time = self.dropout_time0(x_time)
        x_time = x_time.permute(0, 2, 1)
        x_time = F.gelu(self.linear_time(x_time))
        x_time = self.dropout_time1(x_time)

        # skip connection and second layer
        x_time_lin = F.gelu(self.linear_time1(x_time))
        x_time_lin = self.dropout_time2(x_time_lin)
        x_time = x_time + x_time_lin

        # downsample and down project in space
        x_channel = self.conv_space(x_channel)
        x_channel = self.dropout_space0(x_channel)
        x_channel = x_channel.permute(0, 2, 1)
        x_channel = F.gelu(self.linear_space(x_channel))
        x_channel = self.dropout_space1(x_channel)

        # skip connection and second layer
        x_channel_lin = F.gelu(self.linear_space1(x_channel))
        x_channel_lin = self.dropout_space2(x_channel_lin)
        x_channel = x_channel + x_channel_lin

        # now both vectors go from (sample, hidden_channel, hidden_sample) to (sample, -1)
        x_time = torch.flatten(x_time, start_dim=1)
        x_channel = torch.flatten(x_channel, start_dim=1)

        # compute gate
        x_gate_time = torch.sigmoid(torch.squeeze(self.linear_gate_time(x_time)))
        x_gate_channel = torch.sigmoid(
            torch.squeeze(self.linear_gate_channel(x_channel))
        )

        # project through gate
        x_time_gated = torch.einsum("ni, n->ni", x_time, x_gate_time)
        x_channel_gated = torch.einsum("ni, n->ni", x_channel, x_gate_channel)
        x_gated = torch.cat([x_time_gated, x_channel_gated], dim=-1)

        # finally to output layers
        x_gated = self.dropout_output0(x_gated)
        x_gated = F.gelu(self.linear_out(x_gated))
        x_gated = self.dropout_output1(x_gated)

        # skip connection and finally to output
        x_gated_lin = F.gelu(self.linear_out1(x_gated))
        x_gated_lin = self.dropout_output2(x_gated_lin)
        x_gated = x_gated + x_gated_lin
        out = self.output_layer(x_gated)

        return out


class GateTransformerTrainer(BaseTorchTrainer):
    def __init__(
        self,
        model,
        early_stopping,
        n_class,
        str_loss,
        str_reg,
        lam,
        str_opt,
        lr,
        transform,
        target_transform,
        batch_size=32,
        n_epoch=100,
        bool_shuffle=True,
        bool_verbose=False,
        bool_use_ray=False,
        bool_use_gpu=False,
        n_gpu_per_process: int | float = 0,
    ):
        super().__init__(
            model,
            early_stopping,
            n_class,
            str_loss,
            str_reg,
            lam,
            str_opt,
            lr,
            transform,
            target_transform,
            batch_size,
            n_epoch,
            bool_shuffle,
            bool_verbose,
            bool_use_ray,
            bool_use_gpu,
            n_gpu_per_process,
        )


class GatedTransformerModel(BaseTorchModel):

    MODEL_ARCHITECURE_KEYS = [
        "input_token_dim",
        "sequence_length",
        "desired_embedding_dim",
        "positional_encoding",
        "hidden_dim",
        "hidden_unit_dim",
        "hidden_unit_dim_time",
        "num_heads_time",
        "num_heads",
        "kernel_size",
        "stride",
        "n_class",
        "num_layers",
        "num_layers_time",
        "dropout",
        "conv_dropout",
        "output_dropout",
    ]

    def __init__(
        self,
        input_token_dim=None, # Forces user to specify input sample. Number of samples (feature vecs) in the time series
        sequence_length=None, # Forces user to specify sequence length. Number of time steps in the time series
        desired_embedding_dim=None, # Only for embedding positional encoding. Length of encoding vector
        positional_encoding='embedding',
        hidden_dim=512,  # Commonly used size in transformer models
        hidden_unit_dim=64,  # A smaller dimension for the hidden units
        hidden_unit_dim_time=64,  # Same as above, assuming similar scale
        num_heads_time=4,  # Multi-head attention typically uses 4 or 8 heads
        num_heads=4,  # Same as above
        kernel_size=3,  # Typical value for convolutions
        stride=1,  # Stride of 1 is more common than 3 for fine-grained processing
        n_class=2,  # Default for binary classification
        num_layers=6,  # A balance between complexity and performance
        num_layers_time=6,  # Same as above
        dropout=0.1,  # Standard dropout rate in transformer models
        conv_dropout=0.1,  # Usually the same as 'dropout'
        output_dropout=0.1,  # Same as above
        lam=1e-5,  # Regularization term, small value to avoid over-regularization
        n_epoch=20,  # A reasonable default for many tasks
        batch_size=32,  # Common batch size, balancing memory and performance
        act_func="relu",  # ReLU is a standard choice for activation
        str_reg="L2",  # L2 regularization is typical
        str_loss="CrossEntropy",  # Standard loss for classification tasks
        str_opt="Adam",  # Adam optimizer is a common choice
        lr=1e-4,  # A learning rate that works well in many scenarios
        early_stopping=None,  # Early stopping can be optional
        bool_verbose=False,  # Verbose off by default
        transform=None,  # Default as None, user can specify if needed
        target_transform=None,  # Same as 'transform'
        bool_use_ray=False,  # Default off for Ray usage
        bool_use_gpu=False,  # Default off for GPU usage
        n_gpu_per_process: int | float = 0,  # Default to 0, no GPU used
    ):
        
        self.model_kwargs, self.trainer_kwargs = self.split_kwargs_into_model_and_trainer(locals())
        self.device = init_gpu(use_gpu=bool_use_gpu)
        
        # initialize the model
        self.model = GatedTransformerClassifier(
            **self.model_kwargs
        )
        self.model.to(self.device)
        
        # Initialize early stopping
        self.early_stopping = self.get_early_stopper(early_stopping)
        
        # Initialize Trainer
        self.trainer = GateTransformerTrainer(
            self.model, self.early_stopping, **self.trainer_kwargs
        )
        super().__init__(self.model, self.trainer, self.early_stopping, self.model_kwargs, self.trainer_kwargs)


    def override_model(self, kwargs: dict) -> None:
        model_kwargs, trainer_kwargs = self.split_kwargs_into_model_and_trainer(kwargs)
        self.model_kwargs = model_kwargs
        self.trainer_kwargs = trainer_kwargs
        self.model = GatedTransformerClassifier(**model_kwargs)
        self.model.to(self.device)
        self.early_stopping.reset()
        self.trainer = GateTransformerTrainer(
            self.model, self.early_stopping, **trainer_kwargs
        )
        self.model.to(self.device)

    def reset_model(self) -> None:
        # self.override_model(self.model_kwargs | self.trainer_kwargs)
        self.model = GatedTransformerClassifier(**self.model_kwargs)
        self.model.to(self.device)
        self.early_stopping.reset()
        self.trainer = GateTransformerTrainer(
            self.model, self.early_stopping, **self.trainer_kwargs
        )
        self.model.to(self.device)