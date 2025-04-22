from comms import LossyNetwork
from trainer import DistributedTrainer, MyClassifierCallback, compute_classfication_metrics
from data import get_dataset
from transformers import TrainingArguments, AutoModelForSequenceClassification, AutoTokenizer
import torch


def main(args):
    network = LossyNetwork(loss_rate=args.loss_rate)
    network.set_seed(args.seed)

    model = AutoModelForSequenceClassification.from_pretrained(args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    train_dataset, eval_dataset = get_dataset(args, tokenizer)

        
    callback_args = {
        'report_ttac' : [0.5, 0.7, 0.9], #TODO change this for each dataset/model
        'report_file' : f"{args.output_dir}/ttac_report.txt",
        'target_acc': 0.95
    }
    callback = MyClassifierCallback(callback_args)
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        learning_rate= args.learning_rate,
        weight_decay=0.01,
        evaluation_strategy="steps",
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        save_strategy="steps",
        logging_dir=f"{args.output_dir}/logs",
        logging_steps=10,
        fp16=args.fp16,
        report_to="wandb"
    )

    trainer = DistributedTrainer(
        num_nodes=args.num_nodes,
        network=network,
        model=model,
        tokenizer=tokenizer,
        args = training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        callbacks=[callback],
        compute_metrics=compute_classfication_metrics,
    )

    trainer.train()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Distributed Training with Packet Loss")
    parser.add_argument('--num_nodes', type=int, default=2, help='Number of nodes')
    parser.add_argument('--loss_rate', type=float, default=0.001, help='Packet loss rate')
    parser.add_argument('--seed', type=int, default=1234, help='Random seed')
    parser.add_argument('--model_name', type=str, default='google-bert/bert-base-uncased', help='Model name')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--fp16', action='store_true', help='Use mixed precision training')
    parser.add_argument('--output_dir', type=str, default='./output', help='Output directory')
    parser.add_argument('--dataset', '-d', type=str, default='winogrande', choices=['winogrande', 'sst2'], 
                        help='Dataset to use for training (winogrande or sst2)')
    parser.add_argument('--max_samples', type=int, default=0, 
                        help='Maximum number of training samples to use (0 for all)')
    parser.add_argument('--epochs', type=int, default=3, 
                        help='Number of training epochs')
    parser.add_argument('--max_length', type=int, default=128,
                        help='Maximum sequence length for tokenization')
    parser.add_argument('--eval_steps', type=int, default=100)
    parser.add_argument('--save_steps', type=int, default=100)
    parser.add_argument('--logging_steps', type=int, default=10)
    parser.add_argument('--learning_rate', type=float, default=3e-5)
    args = parser.parse_args()
    
    main(args)