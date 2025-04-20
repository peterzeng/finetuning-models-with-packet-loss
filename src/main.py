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
            # Improve Winogrande preprocessing by formatting the input as a choice task
            # Replace the placeholder '_' with each option and tokenize separately
            sentence = data["sentence"]
            option1 = data["option1"]
            option2 = data["option2"]
            
            # Find the placeholder marker in the sentence
            if "_" in sentence:
                placeholder = "_"
            else:
                # Some versions might use a different placeholder
                placeholder = "___"
                
            # Create two complete sentences with each option
            sentence1 = sentence.replace(placeholder, option1)
            sentence2 = sentence.replace(placeholder, option2)
            
            # Use the correct option as the answer
            label = 0 if data["answer"] == '1' else 1
            
            # Tokenize for binary classification - encode each option separately
            encodings = text_tokenizer(
                [sentence1, sentence2],
                truncation=True,
                padding="max_length", 
                max_length=256,
                return_tensors="pt"
            )
            
            # Select the encoding of the correct option
            return {
                'input_ids': encodings["input_ids"][label].tolist(),
                'attention_mask': encodings["attention_mask"][label].tolist(),
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
    
    elif args.dataset == "cola":
        dataset = load_dataset("glue", "cola")
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
        
    elif args.dataset == "mnli":
        dataset = load_dataset("glue", "mnli")
        train_dataset = dataset["train"]
        eval_dataset = dataset["validation_matched"]  # Using matched validation set
        
        # Apply sample limit if specified
        if args.max_samples > 0:
            train_dataset = train_dataset.select(range(min(args.max_samples, len(train_dataset))))
            eval_dataset = eval_dataset.select(range(min(args.max_samples // 5, len(eval_dataset))))
            
        model = AutoModelForSequenceClassification.from_pretrained("google-bert/bert-large-uncased", num_labels=3)
        text_tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-large-uncased")
        
        def preprocess(data):
            return {
                'input_ids': text_tokenizer(data["premise"], data["hypothesis"], 
                                          truncation=True, padding="max_length", max_length=128)["input_ids"],
                'attention_mask': text_tokenizer(data["premise"], data["hypothesis"], 
                                               truncation=True, padding="max_length", max_length=128)["attention_mask"],
                'labels': data["label"]
            }
        
        train_dataset = train_dataset.map(preprocess, remove_columns=["premise", "hypothesis", "idx", "label"])
        eval_dataset = eval_dataset.map(preprocess, remove_columns=["premise", "hypothesis", "idx", "label"])
        
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
        learning_rate= 3e-5,
        weight_decay=0.01,
        evaluation_strategy="steps",
        eval_steps=100,
        save_steps=100,
        save_strategy="steps",
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
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
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