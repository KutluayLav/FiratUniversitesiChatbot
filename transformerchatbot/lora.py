import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.utils.parametrize as parametrize

from dataclasses import dataclass

from model import Transformer, ModelArgs

torch.manual_seed(1234)
device = "cuda" if torch.cuda.is_available() else "cpu"

@dataclass
class LoraConfig:
    rank: int = 32
    alpha: int = 32

class LoRAParametrization(nn.Module):
    def __init__(self, features_in, features_out, rank=1, alpha=1, device=device):
        super().__init__()

        self.lora_A = nn.Parameter(torch.zeros((rank,features_out)).to(device))
        self.lora_B = nn.Parameter(torch.zeros((features_in, rank)).to(device))
        nn.init.normal_(self.lora_A, mean=0, std=1)
        
        self.scale = alpha / rank
        self.enabled = True

    def forward(self, original_weights):
        if self.enabled:
            # Return W + (B*A)*scale
            return original_weights + torch.matmul(self.lora_B, self.lora_A).view(original_weights.shape) * self.scale
        else:
            return original_weights

class Lora:
    def __init__(self, model, config: LoraConfig = LoraConfig()):
        self.model = model
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.apply_lora()

    def linear_layer_parameterization(self, layer):
    
        features_in, features_out = layer.size()
        return LoRAParametrization(
            features_in, features_out, rank=self.config.rank, alpha=self.config.alpha, device=self.device
        )

    def apply_lora(self):
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear) and (name.endswith("attention.q") or name.endswith("attention.v")): 
                print(f"Applying LoRA to {name} with shape {module.weight.size()}")
                parametrize.register_parametrization(
                    module, "weight", self.linear_layer_parameterization(module.weight)
                )

    def enable_disable_lora(self, enabled=True):
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear) and (name.endswith("attention.q") or name.endswith("attention.v")):
                module.parametrizations["weight"][0].enabled = enabled

    def freeze_non_lora_params(self):
        for name, param in self.model.named_parameters():
            if 'lora' not in name:
                print(f"Freezing non-LoRA parameter {name} with shape {param.size()}")
                param.requires_grad = False

    def print_model_parameters(self):
        for name, param in self.model.named_parameters():
            print(f"name: {name} | parameters size: {param.size()} | Trainable: {param.requires_grad}")

    def count_parameters(self):
        """
        Modeldeki toplam ve eğitilebilir parametre sayısını hesaplar.
        """

        total_params = 0
        trainable_params = 0
        for p in self.model.parameters():
            total_params += p.numel()
            if p.requires_grad:
                trainable_params += p.numel()
        return total_params, trainable_params