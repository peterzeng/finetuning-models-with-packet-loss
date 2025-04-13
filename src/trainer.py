from transformers import Trainer, TrainingArguments
import torch

class DistributedTrainer(Trainer):
    def __init__(self, num_nodes, network, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_nodes = num_nodes
        self.network = network
        self.network.set_seed(1234)


    def training_step(self, model, inputs, num_items_in_batch=None):
        minibatch_size = num_items_in_batch // self.num_nodes
        gradients = []
        for i in range(self.num_nodes):
            inputs_split = {k: v[i * minibatch_size:(i + 1) * minibatch_size] for k, v in inputs.items()}
            loss = super().training_step(model, inputs_split)
            loss.backward(retain_graph=True)
            node_gradients = {
                name: param.grad.clone() for name, param in model.named_parameters()
            }
            for k,v in node_gradients.items():
                mask = self.network.send(v)
                node_gradients[k] = self.network.receive(v, mask)

            gradients.append(node_gradients)
        
        # Average gradients across nodes
        averaged_gradients = {}
        for name, param in model.named_parameters():
            averaged_gradients[name] = torch.mean(torch.stack([grad[name] for grad in gradients]), dim=0)
            param.grad = averaged_gradients[name]
        
        return loss
    
    