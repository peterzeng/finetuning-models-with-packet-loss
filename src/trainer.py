from transformers import Trainer, TrainingArguments
import torch

class DistributedTrainer(Trainer):

    def __init__(self, num_nodes, network, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_nodes = num_nodes
        self.network = network
        self.network.set_seed(1234)


    def training_step(self, model, inputs, num_items_in_batch=0):
        num_items_in_batch = len(inputs[list(inputs.keys())[0]])
        minibatch_size = num_items_in_batch // self.num_nodes
        averaged_gradients = {k:torch.zeros_like(v) for k, v in model.named_parameters() if v.requires_grad}
        
        total_loss = torch.tensor(0.0, device=next(model.parameters()).device)
        
        for i in range(self.num_nodes):
            inputs_split = {k: v[i * minibatch_size:(i + 1) * minibatch_size] for k, v in inputs.items()}
            
            with self.compute_loss_context_manager():
                loss, outputs = self.compute_loss(model, inputs_split, return_outputs=True)
            
            loss.backward() 

            total_loss = total_loss + loss.detach()  # Add to total_loss for reporting
            
            node_gradients = {
                name: param.grad.clone() for name, param in model.named_parameters()
            }

            for k,v in node_gradients.items():
                mask = self.network.send(v)
                node_gradients[k] = self.network.receive(v, mask)
                averaged_gradients[k] = averaged_gradients[k] + node_gradients[k]

            model.zero_grad()
            
        averaged_gradients = {k: v / self.num_nodes for k, v in averaged_gradients.items()}
        
        for name, param in model.named_parameters():
            param.grad = averaged_gradients[name]
        
        return total_loss / self.num_nodes

