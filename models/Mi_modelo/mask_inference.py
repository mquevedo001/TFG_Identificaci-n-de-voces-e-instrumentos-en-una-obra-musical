from pyexpat import model

from nussl.ml.networks.modules import AmplitudeToDB, BatchNorm, RecurrentStack, Embedding
from sklearn import dummy
from torch import nn
from config import config
import nussl

class MaskInference(nn.Module):
    def __init__(
        self,
        num_features,
        num_audio_channels,
        hidden_size,
        num_layers,
        bidirectional,
        dropout,
        num_sources,
        activation,
    ):
        super().__init__()

        self.num_features = int(num_features)
        self.num_audio_channels = int(num_audio_channels)
        self.num_sources = int(num_sources)
        self.hidden_size = int(hidden_size)
        self.num_layers = int(num_layers)
        self.bidirectional = bool(bidirectional)

        embedding_hidden_size = self.hidden_size * (2 if self.bidirectional else 1)

        self.amplitude_to_db = AmplitudeToDB()
        self.input_normalization = BatchNorm(self.num_features)

        self.recurrent_stack = RecurrentStack(
            num_features=self.num_features * self.num_audio_channels,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            bidirectional=self.bidirectional,
            dropout=dropout,
        )

        self.embedding = Embedding(
            num_features=self.num_features,
            hidden_size=embedding_hidden_size,
            embedding_size=self.num_sources,
            activation=activation,
            num_audio_channels=self.num_audio_channels,
        )

    def forward(self, data):
        if data.dim() == 2:
            data = data.unsqueeze(0)

        # Admitimos:
        #   [B, F, T]
        #   [B, T, F]
        #   [B, T, F, 1]
        #   [B, F, T, 1]
        if data.dim() == 4:
            if data.shape[-1] != 1:
                raise ValueError(
                    f"Solo está soportado C=1 en inferencia actual. Recibido: {data.shape}"
                )
            data = data.squeeze(-1)

        if data.dim() != 3:
            raise ValueError(f"MaskInference esperaba tensor 3D o 4D, recibió {data.shape}")

        if data.shape[1] == self.num_features:
            # [B, F, T] -> [B, T, F]
            rnn_input = data.permute(0, 2, 1).contiguous()
        elif data.shape[2] == self.num_features:
            # [B, T, F]
            rnn_input = data.contiguous()
        else:
            raise ValueError(
                f"No puedo inferir eje de frecuencia en input {data.shape}. "
                f"num_features esperado={self.num_features}"
            )

        mask = self.embedding(rnn_input)

        B, T, F = rnn_input.shape

        if mask.dim() == 3:
            mask = mask.view(
                B,
                T,
                self.num_features,
                self.num_audio_channels,
                self.num_sources,
            )
        elif mask.dim() == 4:
            # [B, T, F, S] -> [B, T, F, 1, S]
            mask = mask.unsqueeze(3)
        elif mask.dim() != 5:
            raise ValueError(f"Shape inesperada de mask: {mask.shape}")

        mix = rnn_input.unsqueeze(3).unsqueeze(-1)  # [B, T, F, 1, 1]
        estimates = mix * mask                     # [B, T, F, C, S]

        return {
            "mask": mask,
            "estimates": estimates,
        }

    @classmethod
    def build(cls, num_features, num_audio_channels, hidden_size,
              num_layers, bidirectional, dropout, num_sources,
              activation='sigmoid'):
        nussl.ml.register_module(cls)

        modules = {
            'model': {
                'class': 'MaskInference',
                'args': {
                    'num_features': num_features,
                    'num_audio_channels': num_audio_channels,
                    'hidden_size': hidden_size,
                    'num_layers': num_layers,
                    'bidirectional': bool(bidirectional),
                    'dropout': dropout,
                    'num_sources': num_sources,
                    'activation': activation
                }
            }
        }

        connections = [
            ['model', ['mix_magnitude']]
        ]

        for key in ['mask', 'estimates']:
            modules[key] = {'class': 'Alias'}
            connections.append([key, [f'model:{key}']])

        output = ['estimates', 'mask']

        config = {
            'name': cls.__name__,
            'modules': modules,
            'connections': connections,
            'output': output
        }

        return nussl.ml.SeparationModel(config)
