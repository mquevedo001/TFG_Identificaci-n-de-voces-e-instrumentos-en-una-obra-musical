from pyexpat import model

from nussl.ml.networks.modules import AmplitudeToDB, BatchNorm, RecurrentStack, Embedding
from sklearn import dummy
from torch import nn
import torch
from config import config
import nussl
import numpy as np

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

        # Permitimos:
        #   [B, F, T]
        #   [B, T, F]
        #   [B, T, F, 1]
        #   [B, F, T, 1]
        if data.dim() == 4:
            if data.shape[-1] != 1:
                raise ValueError(
                    f"Solo está soportado C=1 en esta inferencia. Recibido: {data.shape}"
                )
            data = data.squeeze(-1)

        if data.dim() != 3:
            raise ValueError(f"MaskInference esperaba tensor 3D o 4D, recibió {data.shape}")

        # Queremos rnn_input en formato [B, T, F]
        if data.shape[1] == self.num_features:
            # [B, F, T] -> [B, T, F]
            mix_mag_btf = data.permute(0, 2, 1).contiguous()
        elif data.shape[2] == self.num_features:
            # [B, T, F]
            mix_mag_btf = data.contiguous()
        else:
            raise ValueError(
                f"No puedo inferir eje de frecuencia en input {data.shape}. "
                f"num_features esperado={self.num_features}"
            )

        # CLAVE: primero pasa por la RNN.
        # Para checkpoint bidireccional hidden=50, esto produce [B, T, 100].
        rnn_output = self.recurrent_stack(mix_mag_btf)

        # Ahora sí: Embedding espera última dimensión 100.
        print("mix_mag_btf:", mix_mag_btf.shape)
        print("rnn_output:", rnn_output.shape)
        mask = self.embedding(rnn_output)

        B, T, F = mix_mag_btf.shape

        if mask.dim() == 3:
            # Normalmente: [B, T, F * C * S]
            mask = mask.view(
                B,
                T,
                self.num_features,
                self.num_audio_channels,
                self.num_sources,
            )
        elif mask.dim() == 4:
            # Si nussl devuelve [B, T, F, S], añadimos eje C.
            if mask.shape[-1] == self.num_sources:
                mask = mask.unsqueeze(3)
            else:
                raise ValueError(f"Shape inesperada de mask 4D: {mask.shape}")
        elif mask.dim() != 5:
            raise ValueError(f"Shape inesperada de mask: {mask.shape}")

        # mix_mag_btf: [B, T, F]
        # mask:        [B, T, F, C, S]
        mask = mask.view(
            B,
            T,
            self.num_features,
            self.num_audio_channels,
            self.num_sources,
        )

        mask = mask / (mask.sum(dim=-1, keepdim=True) + 1e-8)

        mix = mix_mag_btf.unsqueeze(3).unsqueeze(-1)
        estimates = mix * mask

        print("\n[MASK STATS]")
        print("min:", mask.min().item())
        print("max:", mask.max().item())
        print("mean:", mask.mean().item())
        print("std:", mask.std().item())
        with torch.no_grad():
            m = mask.detach()

            if m.dim() == 5:
                # [B, T, F, C, S] -> [B, T, F, S]
                m4 = m[:, :, :, 0, :]
            elif m.dim() == 4:
                # [B, T, F, S]
                m4 = m
            else:
                raise ValueError(f"mask shape inesperada: {m.shape}")

            print("\n[MASK PER SOURCE]")
            for s in range(m4.shape[-1]):
                ms = m4[..., s]
                print(
                    f"source {s}: "
                    f"mean={ms.mean().item():.4f}, "
                    f"std={ms.std().item():.4f}, "
                    f"min={ms.min().item():.4f}, "
                    f"max={ms.max().item():.4f}"
                )

            mask_sum = m4.sum(dim=-1)
            print(
                "[MASK SUM] "
                f"mean={mask_sum.mean().item():.4f}, "
                f"std={mask_sum.std().item():.4f}, "
                f"min={mask_sum.min().item():.4f}, "
                f"max={mask_sum.max().item():.4f}"
            )

            flat = m4.reshape(-1, m4.shape[-1]).float().cpu().numpy()
            corr = np.corrcoef(flat.T)
            print("[MASK SOURCE CORR]")
            print(corr)
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
