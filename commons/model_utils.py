import torch

class FeatureExtractor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = torch.nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        return x

    def extract_features(self, x, layers):
        features = {}
        #x = x.float()  # Cast a float32 para evitar Input type (struct c10::Half) and bias type (float) should be the same
        
        x = x.to(           #CAMBIO V2: Se vuelve a la versión anterior.
            device=self.conv1.weight.device,
            dtype=self.conv1.weight.dtype,
        )
        
        x = self.relu(self.conv1(x))
        if "conv1" in layers:
            features["conv1"] = x.clone()
        x = self.relu(self.conv2(x))
        if "conv2" in layers:
            features["conv2"] = x.clone()
        return features

def reshape_if_needed(estimates, targets, to='spectrogram'):
    """
    Adapta los tensores estimates y targets para funciones de pérdida específicas.
    - to='spectrogram': [B, F, T, S] → [B*S, 1, F, T]
    - to='freq': [B, F, T, S] → [B*S, F, T]
    """

    B, F, T, S = estimates.shape

    if to == 'spectrogram': return ( estimates.permute(0, 3, 1, 2).reshape(B * S, 1, F, T),targets.permute(0, 3, 1, 2).reshape(B * S, 1, F, T))
    elif to == 'freq': return ( estimates.permute(0, 3, 1, 2).reshape(B * S, F, T), targets.permute(0, 3, 1, 2).reshape(B * S, F, T))
    else: return estimates, targets
        

def prepare_batch(batch, device, dtype=torch.float32):
    new_batch = {}

    for k, v in batch.items():
        
        if torch.is_tensor(v): new_batch[k] = v.to(device=device, dtype=dtype)
        else: new_batch[k] = v
            
    return new_batch
