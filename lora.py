import torch
import torch.nn as nn


class LoRALayerWrapper(nn.Module):
    def __init__(self, base_module: nn.Module, lora_rank: int):
        super().__init__()

        self.base_module = base_module
        self.matrix_shape = base_module.weight.shape
        if isinstance(base_module, nn.Linear):
            self.matrix_shape = (base_module.in_features, base_module.out_features)
        self.lora_A = nn.Parameter(torch.randn(self.matrix_shape[0], lora_rank))
        self.lora_B = nn.Parameter(torch.zeros(self.matrix_shape[1], lora_rank))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base_module(x)  # The output of the pre-trained module.
        # import pdb; pdb.set_trace()
        # print(x.shape, self.lora_A.shape, self.lora_B.shape, self.matrix_shape, self.base_module)    
        btx = torch.matmul(x, self.lora_A)
        out = torch.matmul(btx, self.lora_B.T)
        return base_out + out
