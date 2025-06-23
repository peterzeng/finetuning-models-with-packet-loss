from transformers import Trainer, TrainerCallback, Seq2SeqTrainer
import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

class DistributedTrainer(Trainer):

    def __init__(self, num_nodes, network, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_nodes = num_nodes
        self.network = network
        self.backup_weights = {}
        # fill backup_weights with the model weights
        for name, param in self.model.named_parameters():
            self.backup_weights[name] = param.data.clone()


    def training_step(self, model, inputs, num_items_in_batch=0):

        num_items_in_batch = len(inputs[list(inputs.keys())[0]])
        minibatch_size = num_items_in_batch // self.num_nodes
        if minibatch_size == 0:
            num_nodes = 1
        else:
            num_nodes = self.num_nodes
        averaged_gradients = {k:torch.zeros_like(v, device=model.device) for k, v in model.named_parameters() if v.requires_grad}
        
        total_loss = torch.tensor(0.0, device=next(model.parameters()).device, dtype=torch.float16 if self.args.fp16 else torch.float32)
        for name, param in model.named_parameters():
            self.backup_weights[name] = param.data.clone()    

        for i in range(num_nodes):
            
            for name, param in model.named_parameters():
                mask = self.network.send(param.data)
                param.data = self.network.receive(param.data, mask) ## This simulates the packet loss during broadcasting 

            inputs_split = {k: v[i * minibatch_size:(i + 1) * minibatch_size] for k, v in inputs.items()}
            
            loss = self.compute_loss(model, inputs_split)
            
            if self.args.fp16:
                self.accelerator.backward(loss)
            else:
                loss.backward()
                
            total_loss = total_loss + loss.detach()  # Add to total_loss for reporting

            for name,param in model.named_parameters():
                if param.grad is not None: 
                    mask = self.network.send(param.grad)
                    averaged_gradients[name] = averaged_gradients[name] + self.network.receive(param.grad, mask)

        for name, param in model.named_parameters():
            if param.requires_grad:
                param.grad = averaged_gradients[name]

        for name, param in model.named_parameters():
            param.data = self.backup_weights[name]  # Restore the original weights
        
        return total_loss / num_nodes



class MyQATrainer(DistributedTrainer):

    def __init__(self, num_nodes, network, *args, **kwargs):
        super().__init__(num_nodes, network, *args, **kwargs)
        self.eos_token_id = kwargs['tokenizer'].eos_token_id

    # def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
    #     if prediction_loss_only:
    #         return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)

    #     model.eval()
    #     with torch.no_grad():
    #         outputs = model(**inputs)
    #         loss = getattr(outputs, "loss", None)

    #         generated_tokens = model.generate(
    #             input_ids=inputs["input_ids"],
    #             attention_mask=inputs["attention_mask"],
    #             do_sample=False,
    #             max_new_tokens=10,
    #             eos_token_id=self.eos_token_id,
    #             pad_token_id=model.config.pad_token_id, 
    #             early_stopping=True
    #         )

    #     labels = inputs.get("labels")
    #     return (loss, generated_tokens, labels)
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        if prediction_loss_only:
            return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)

        model.eval()
        with torch.no_grad():
            loss = None
            if "labels" in inputs:
                outputs = model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    labels=inputs["labels"]
                )
                loss = outputs.loss

            generated_tokens = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=20,
                do_sample=False,
                eos_token_id=self.eos_token_id,
                pad_token_id=model.config.pad_token_id,
                early_stopping=True
            )

        # Ensure all tensors are returned
        labels = inputs.get("labels", None)
        return (loss, generated_tokens, labels)


def compute_exact_match_metric(tokenizer):
    def compute_metrics(eval_pred):
        preds, labels = eval_pred
        preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        breakpoint()
        correct = sum(
            (pred.partition("Answer:")[2].strip().lower() == label.partition("Answer:")[2].strip().lower())
            for pred, label in zip(decoded_preds, decoded_labels)
        )

        return {"exact_match": correct / len(decoded_preds)}

    return compute_metrics

def compute_classfication_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="weighted")
    }


class MyQACallback(TrainerCallback):

    def __init__(self, args=None):
        super().__init__()
        self.args = args
        self.args['report_ttac'] = sorted(self.args['report_ttac'])

    def on_evaluate(self, args, state, control, **kwargs):
        
        exact_match = kwargs["metrics"]["eval_exact_match"]
        if exact_match > self.args['target_acc']:
            print(f"Target exact match {self.args['target_acc']} reached. Stopping training.")
            control.should_training_stop = True

        if len(self.args['report_ttac']) > 0:
            if exact_match > self.args['report_ttac'][0]:
                with open(self.args['report_file'], "a") as f:
                    f.write(f"Exact Match: {exact_match:.3f}, Threshold: {self.args['report_ttac'][0]},  Step: {state.global_step}\n")
                self.args['report_ttac'] = self.args['report_ttac'][1:]
            
        return super().on_evaluate(args, state, control, **kwargs)

class MyClassifierCallback(TrainerCallback):

    def __init__(self, args=None):
        super().__init__()
        self.counter = 0
        self.patience = 5
        self.args = args
        self.args['report_ttac'] = sorted(self.args['report_ttac'])
        self.args['report_ttac'] = sorted(self.args['report_ttac'], reverse=True)
        # if report file exists, remove it
        try:
            with open(self.args['report_file'], "r") as f:
                pass
        except FileNotFoundError:
            with open(self.args['report_file'], "w") as f:
                f.write("")

    def on_evaluate(self, args, state, control, **kwargs):
        
        accuracy = kwargs["metrics"]["eval_accuracy"]
        if accuracy > self.args['target_acc']:
            self.counter +=1
            if self.counter >= self.patience:
                print(f"Target accuracy {self.args['target_acc']} reached. Stopping training.")
                control.should_training_stop = True

        if len(self.args['report_ttac']) > 0:
            if accuracy > self.args['report_ttac'][0]:
                with open(self.args['report_file'], "a") as f:
                    f.write(f"Accuracy: {accuracy:.3f}, Threshold: {self.args['report_ttac'][0]},  Step: {state.global_step}\n")
                self.args['report_ttac'] = self.args['report_ttac'][1:]
            
        return super().on_evaluate(args, state, control, **kwargs)
        
