from random import triangular
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from ..base import BaseModel
from .base import TorchModelWrapper


class SinusoidPositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        return self.pe[:x.size(0)]


class GatedTransformerClassifier(torch.nn.Module, TorchModelWrapper):
    def __init__(self, input_sample, input_dim, hidden_dim, hidden_unit_dim, hidden_unit_dim_time,
                num_heads_time, num_heads, kernel_size, stride, num_layers, num_layers_time,
                dropout, conv_dropout, output_dropout, n_target):
        super().__init__()
        self.input_dim = input_dim
        self.input_sample = input_sample
        self.hidden_dim = hidden_dim
        self.hidden_sample = (self.input_sample - kernel_size) // stride + 1
        self.idx_sample = torch.arange(0, input_sample, 1, dtype=torch.int).cuda()

        # positional encoding for the time encoder
        # self.pos_encoder = PositionalEncoding(input_sample, output_dropout)
        self.pos_encoder = nn.Embedding(input_sample, input_dim)
        self.dropout_input = nn.Dropout(output_dropout)

        # define the time encoding layers
        self.time_encoder_layer = nn.TransformerEncoderLayer(d_model=input_sample, nhead=num_heads_time,
                                                            dim_feedforward=hidden_unit_dim_time,
                                                            dropout=dropout, activation='gelu')
        self.time_encoder = nn.TransformerEncoder(self.time_encoder_layer, num_layers=num_layers_time)

        # define the channel encoding layers
        self.channel_encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=num_heads,
                                                                dim_feedforward=hidden_unit_dim,
                                                                dropout=dropout, activation='gelu')
        self.channel_encoder = nn.TransformerEncoder(self.channel_encoder_layer, num_layers=num_layers)

        # define conv and linear layers for downsampling the time attention output
        self.conv_time = nn.Conv1d(in_channels=input_dim, out_channels=hidden_dim,
                                kernel_size=kernel_size, stride=stride)
        self.dropout_time0 = nn.Dropout(conv_dropout)
        self.linear_time = nn.Linear(hidden_dim, hidden_dim)
        self.dropout_time1 = nn.Dropout(conv_dropout)
        self.linear_time1 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout_time2 = nn.Dropout(conv_dropout)

        # define conv and linear layers for downsampling the time attention output
        self.conv_space = nn.Conv1d(in_channels=input_dim, out_channels=hidden_dim,
                                    kernel_size=kernel_size, stride=stride)
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
        self.linear_out = nn.Linear(hidden_dim * self.hidden_sample * 2, hidden_dim * self.hidden_sample)
        self.dropout_output1 = nn.Dropout(output_dropout)
        self.linear_out1 = nn.Linear(hidden_dim * self.hidden_sample, hidden_dim * self.hidden_sample)
        self.dropout_output2 = nn.Dropout(output_dropout)
        self.output_layer = nn.Linear(hidden_dim * self.hidden_sample, n_target)

    def forward(self, input):
        # input is in shape (sample, time, channel)
        # the time encoder is attending along channel axis
        # first transpose data to shape (channel, sample, time)
        x_time = input.contiguous().permute(2, 0, 1)
        x_time = self.time_encoder(x_time)

        # also transpose data to shape (time, sample, channel)
        # the channel encoder is attending along time axis
        x_channel = input.contiguous().permute(1, 0, 2)
        x_pos = torch.unsqueeze(self.pos_encoder(self.idx_sample), dim=1)
        x_channel = self.dropout_input(x_channel + x_pos)
        x_channel = self.channel_encoder(x_channel)

        # now transpose everything back to (sample, channel, time)
        x_time = x_time.permute(1, 0, 2)
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
        x_gate_channel = torch.sigmoid(torch.squeeze(self.linear_gate_channel(x_channel)))

        # project through gate
        x_time_gated = torch.einsum('ni, n->ni', x_time, x_gate_time)
        x_channel_gated = torch.einsum('ni, n->ni', x_channel, x_gate_channel)
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


class GatedTransformerClassifierModel(BaseModel):
    def __init__(self, input_sample, input_dim, hidden_dim, hidden_unit_dim, hidden_unit_dim_time,
                num_heads_time, num_heads, kernel_size, stride, num_layers, num_layers_time,
                dropout, conv_dropout, output_dropout, n_target):
    
        self.model = GatedTransformerClassifier(input_sample, input_dim, hidden_dim, hidden_unit_dim, hidden_unit_dim_time,
                    num_heads_time, num_heads, kernel_size, stride, num_layers, num_layers_time,
                    dropout, conv_dropout, output_dropout, n_target)
    
    def override_model(self, model_args, model_kwargs) -> None:
        self.model = GatedTransformerClassifier(*model_args, **model_kwargs)