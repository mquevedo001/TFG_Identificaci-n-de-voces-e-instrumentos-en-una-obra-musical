from nussl.ml.networks.modules import AmplitudeToDB, BatchNorm, RecurrentStack, Embedding
from torch import nn
import torch
import nussl

class MaskInference(nn.Module):
    def __init__(self, num_features, num_audio_channels, hidden_size,
                 num_layers, bidirectional, dropout, num_sources,
                 activation):
        super().__init__()

        bidirectional = bool(bidirectional)  # asegurar tipo correcto
        embedding_hidden_size = hidden_size * (2 if bidirectional else 1)

        self.amplitude_to_db = AmplitudeToDB()
        self.input_normalization = BatchNorm(num_features)
        self.recurrent_stack = RecurrentStack(
            num_features=num_features * num_audio_channels,
            hidden_size=hidden_size,  # tamaño de la RNN
            num_layers=num_layers,
            bidirectional=bidirectional,
            dropout=dropout,
        )

        self.embedding = Embedding(
            num_features=num_features,
            hidden_size=embedding_hidden_size,  # tamaño esperado por Embedding
            embedding_size=num_sources,
            activation=activation,
            num_audio_channels=num_audio_channels
        )

    def forward(self, data):
        mix_magnitude = data  # conservar para aplicar máscara
        data = data.float()

        data = self.amplitude_to_db(mix_magnitude)
        data = self.input_normalization(data)
        data = self.recurrent_stack(data)
        mask = self.embedding(data)
        estimates = mix_magnitude.unsqueeze(-1) * mask

        return {
            'mask': mask,
            'estimates': estimates
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
