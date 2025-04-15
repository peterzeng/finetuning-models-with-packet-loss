from transformers import Trainer, TrainingArguments
import torch
import torch.nn.functional as F

class DistributedTrainer(Trainer):

    def __init__(self, num_nodes, network, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_nodes = num_nodes
        self.network = network
        # Don't override the seed here - it's already set in network by main.py

    def training_step(self, model, inputs, num_items_in_batch=0):
        num_items_in_batch = len(inputs[list(inputs.keys())[0]])
        minibatch_size = num_items_in_batch // self.num_nodes
        
        # Use parameter count for more precise tracking
        averaged_gradients = {k: torch.zeros_like(v, device=v.device) 
                             for k, v in model.named_parameters() if v.requires_grad}
        
        # Track per-node losses for debugging
        node_losses = []
        
        # Initialize total loss with device matching model
        device = next(model.parameters()).device
        total_loss = torch.tensor(0.0, device=device)
        
        for i in range(self.num_nodes):
            # Skip if we're at the end and have an incomplete batch
            if i * minibatch_size >= num_items_in_batch:
                continue
                
            # Handle potential incomplete final batch
            end_idx = min((i + 1) * minibatch_size, num_items_in_batch)
            inputs_split = {k: v[i * minibatch_size:end_idx] for k, v in inputs.items()}
            
            # Get loss from parent class
            loss = super().training_step(model, inputs_split)
            
            # Ensure loss is a tensor with gradient tracking
            if not isinstance(loss, torch.Tensor):
                loss = torch.tensor(loss, device=device, requires_grad=True)
            elif not loss.requires_grad:
                loss = loss.detach().clone().requires_grad_(True)
                
            # Track node loss
            node_losses.append(loss.item())
            
            # Accumulate total loss (normalized by actual batch size)
            actual_batch_size = end_idx - (i * minibatch_size)
            batch_fraction = actual_batch_size / num_items_in_batch
            total_loss = total_loss + (loss * batch_fraction)
            
            # Compute gradients 
            loss.backward(retain_graph=True)
            
            # Collect and process gradients with packet loss simulation
            node_gradients = {}
            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    # Clone gradients and apply packet loss
                    grad = param.grad.clone()
                    mask = self.network.send(grad)
                    received_grad = self.network.receive(grad, mask)
                    
                    # Store with proper normalization by batch fraction
                    node_gradients[name] = received_grad * batch_fraction
                    
                    # Accumulate into averaged gradients
                    if name in averaged_gradients:
                        averaged_gradients[name] += node_gradients[name]
                        
            # Clear gradients for next node
            model.zero_grad()
            
        # Set accumulated gradients on model parameters
        for name, param in model.named_parameters():
            if param.requires_grad and name in averaged_gradients:
                param.grad = averaged_gradients[name]
        
        return total_loss

