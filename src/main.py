from comms import LossyNetwork
from trainer import DistributedTrainer
from datasets import load_dataset
from transformers import TrainingArguments, AutoModelForSequenceClassification, AutoTokenizer
import torch


def main(args):
    network = LossyNetwork(loss_rate=args.loss_rate)
    network.set_seed(args.seed)

    if args.dataset == "winogrande":
        dataset = load_dataset("allenai/winogrande", "winogrande_l", trust_remote_code=True)    
        train_dataset = dataset["train"]
        eval_dataset = dataset["validation"]
        
        # Apply sample limit if specified
        if args.max_samples > 0:
            train_dataset = train_dataset.select(range(min(args.max_samples, len(train_dataset))))
            eval_dataset = eval_dataset.select(range(min(args.max_samples // 5, len(eval_dataset))))
            
        model = AutoModelForSequenceClassification.from_pretrained("google-bert/bert-large-uncased", num_labels=2)
        model = model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        text_tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-large-uncased")

        def preprocess(data):
            question = data["sentence"] + "\n" + data["option1"] + "\n" + data["option2"]
            label = 0 if data["answer"] == '1' else 1
            return {
                'input_ids': text_tokenizer(question, truncation=True, padding="max_length", max_length=512)["input_ids"],
                'attention_mask': text_tokenizer(question, truncation=True, padding="max_length", max_length=512)["attention_mask"],
                'labels': label
            }
        
        train_dataset = train_dataset.map(preprocess, remove_columns=["sentence", "option1", "option2", "answer"])
        eval_dataset = eval_dataset.map(preprocess, remove_columns=["sentence", "option1", "option2", "answer"])
    
    elif args.dataset == "sst2":
        dataset = load_dataset("glue", "sst2")
        train_dataset = dataset["train"]
        eval_dataset = dataset["validation"]
        
        # Apply sample limit if specified
        if args.max_samples > 0:
            train_dataset = train_dataset.select(range(min(args.max_samples, len(train_dataset))))
            eval_dataset = eval_dataset.select(range(min(args.max_samples // 5, len(eval_dataset))))
            
        model = AutoModelForSequenceClassification.from_pretrained("google-bert/bert-large-uncased", num_labels=2)
        text_tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-large-uncased")
        
        def preprocess(data):
            return {
                'input_ids': text_tokenizer(data["sentence"], truncation=True, padding="max_length", max_length=128)["input_ids"],
                'attention_mask': text_tokenizer(data["sentence"], truncation=True, padding="max_length", max_length=128)["attention_mask"],
                'labels': data["label"]
            }
        
        train_dataset = train_dataset.map(preprocess, remove_columns=["sentence", "idx", "label"])
        eval_dataset = eval_dataset.map(preprocess, remove_columns=["sentence", "idx", "label"])
        
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    # Adjust training epochs based on dataset size
    epochs = args.epochs
    if args.max_samples > 0 and args.epochs == 3:
        # If using a small subset and epochs not specified, use more epochs
        epochs = 5
        
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=epochs,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_dir=f"{args.output_dir}/logs",
        logging_steps=10,
        fp16=args.fp16,
    )

    trainer = DistributedTrainer(
        num_nodes=args.num_nodes,
        network=network,
        model=model,
        tokenizer=text_tokenizer,
        args = training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    trainer.train()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Distributed Training with Packet Loss")
    parser.add_argument('--num_nodes', type=int, default=2, help='Number of nodes')
    parser.add_argument('--loss_rate', type=float, default=0.001, help='Packet loss rate')
    parser.add_argument('--seed', type=int, default=1234, help='Random seed')
    parser.add_argument('--model_name', type=str, default='openai-community/gpt2', help='Model name')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--fp16', action='store_true', help='Use mixed precision training')
    parser.add_argument('--output_dir', type=str, default='./output', help='Output directory')
    parser.add_argument('--dataset', '-d', type=str, default='winogrande', choices=['winogrande', 'sst2'], 
                        help='Dataset to use for training (winogrande or sst2)')
    parser.add_argument('--max_samples', type=int, default=0, 
                        help='Maximum number of training samples to use (0 for all)')
    parser.add_argument('--epochs', type=int, default=3, 
                        help='Number of training epochs')
    args = parser.parse_args()
    
    main(args)