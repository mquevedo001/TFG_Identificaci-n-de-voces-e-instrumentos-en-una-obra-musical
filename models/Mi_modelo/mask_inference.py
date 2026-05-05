from pyexpat import model

from nussl.ml.networks.modules import AmplitudeToDB, BatchNorm, RecurrentStack, Embedding
from sklearn import dummy
from torch import nn
from config import config
import nussl

class MaskInference(nn.Module):
    def __init__(self, num_features, num_audio_channels, hidden_size,
                 num_layers, bidirectional, dropout, num_sources,
                 activation):
        super().__init__()

        bidirectional = bool(bidirectional)
        embedding_hidden_size = hidden_size * (2 if bidirectional else 1)

        self.amplitude_to_db = AmplitudeToDB()
        self.input_normalization = BatchNorm(num_features)
        self.recurrent_stack = RecurrentStack(
            num_features=num_features * num_audio_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
            dropout=dropout,
        )

        self.embedding = Embedding(
            num_features=num_features,
            hidden_size=embedding_hidden_size,  # tamaño esperado por Embedding
            embedding_size=int(num_sources),
            activation=activation,
            num_audio_channels=num_audio_channels
        )

    def forward(self, data):

        if data.dim() == 2:
            data = data.unsqueeze(0)

        # ORIGINAL espectrograma (GUARDARLO)
        mix_mag = data.clone()

        # (B, F, T) -> (B, T, F)
        data = data.permute(0, 2, 1).contiguous()

        print("INPUT RNN:", data.shape)

        data = self.recurrent_stack(data)

        print("AFTER RNN:", data.shape)

        mask = self.embedding(data)

        F = int(config.config['MODEL_INPUT_SIZE'])
        S = int(config.config['MODEL_NUM_SOURCES'])

        mask = mask.view(mask.size(0), mask.size(1), F, S)

        # ⚠️ usar mix_mag original (NO data)
        mix_mag = mix_mag.permute(0, 2, 1)  # (B, T, F)
        mix = mix_mag.unsqueeze(-1)

        estimates = mix * mask

        print("\n[MASK STATS]")
        print("min:", mask.min().item())
        print("max:", mask.max().item())
        print("mean:", mask.mean().item())
        print("std:", mask.std().item())
        return {
            "mask": mask,
            "estimates": estimates
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
