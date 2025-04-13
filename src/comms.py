from typing import List
import torch
import math

MAX_PAYLOAD_BYTES = 1450

class LossyNetwork:
    def __init__(self, loss_rate: float = 0.001):
        self.loss_rate = loss_rate

    def set_seed(self, seed: int):
        self.seed = seed

    def send(self, data: torch.Tensor) -> torch.Tensor:
        data_size = data.numel()
        float_size = data.element_size()
        num_bytes = data_size * float_size
        num_packets = math.ceil(num_bytes / MAX_PAYLOAD_BYTES)
        packets_mask = torch.rand(num_packets) > self.loss_rate
        return packets_mask
    
    def receive(self, data: torch.Tensor, packets_mask: torch.Tensor) -> torch.Tensor:
        num_packets = len(packets_mask)
        num_bytes = data.numel()
        received_data = data.clone()
        for i in range(num_packets):
            if not packets_mask[i]:
                start = i * MAX_PAYLOAD_BYTES
                end = min(start + MAX_PAYLOAD_BYTES, num_bytes)
                received_data[start:end] = 0
        return received_data