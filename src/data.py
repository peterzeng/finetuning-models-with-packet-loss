from datasets import load_dataset

def get_dataset(args, tokenizer):
    if args.dataset == "sst2":
        return get_sst2(tokenizer, args)
    elif args.dataset == "cola":
        return get_cola(tokenizer, args)
    elif args.dataset == "mnli":
        return get_mnli(tokenizer, args)
    elif args.dataset == "winogrande":
        return get_winogrande(tokenizer, args)
    else:
        raise ValueError(f"Dataset {args.dataset} not supported.")


def get_winogrande(tokenizer, args):

    max_length = args.max_length if args.max_length > 0 else 256
    dataset = load_dataset("allenai/winogrande", "winogrande_l", trust_remote_code=True)    
    train_dataset = dataset["train"]
    eval_dataset = dataset["validation"]

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
        encodings = tokenizer(
            [sentence1, sentence2],
            truncation=True,
            padding="max_length", 
            max_length=max_length,
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

    return train_dataset, eval_dataset

def get_sst2(tokenizer, args):

    dataset = load_dataset("glue", "sst2")
    train_dataset = dataset["train"]
    eval_dataset = dataset["validation"]

    max_length = args.max_length if args.max_length > 0 else 128

    def preprocess(data):
        return {
            'input_ids': tokenizer(data["sentence"], truncation=True, padding="max_length", max_length=max_length)["input_ids"],
            'attention_mask': tokenizer(data["sentence"], truncation=True, padding="max_length", max_length=max_length)["attention_mask"],
            'labels': data["label"]
        }
        
    train_dataset = train_dataset.map(preprocess, remove_columns=["sentence", "idx", "label"])
    eval_dataset = eval_dataset.map(preprocess, remove_columns=["sentence", "idx", "label"])
    
    return train_dataset, eval_dataset


def get_cola(tokenizer, args):
    max_length = args.max_length if args.max_length > 0 else 128
    dataset = load_dataset("glue", "cola")
    train_dataset = dataset["train"]
    eval_dataset = dataset["validation"]

    def preprocess(data):
        return {
            'input_ids': tokenizer(data["sentence"], truncation=True, padding="max_length", max_length=max_length)["input_ids"],
            'attention_mask': tokenizer(data["sentence"], truncation=True, padding="max_length", max_length=max_length)["attention_mask"],
            'labels': data["label"]
        }
        
    train_dataset = train_dataset.map(preprocess, remove_columns=["sentence", "idx", "label"])
    eval_dataset = eval_dataset.map(preprocess, remove_columns=["sentence", "idx", "label"])

    return train_dataset, eval_dataset

def get_mnli(tokenizer, args):
    max_length = args.max_length if args.max_length > 0 else 128
    dataset = load_dataset("glue", "mnli")
    train_dataset = dataset["train"]
    eval_dataset = dataset["validation_matched"]  # Using matched validation set

    def preprocess(data):
        return {
            'input_ids': tokenizer(data["premise"], data["hypothesis"], truncation=True, padding="max_length", max_length=max_length)["input_ids"],
            'attention_mask': tokenizer(data["premise"], data["hypothesis"], truncation=True, padding="max_length", max_length=max_length)["attention_mask"],
            'labels': data["label"]
        }
        
    train_dataset = train_dataset.map(preprocess, remove_columns=["premise", "hypothesis", "idx", "label"])
    eval_dataset = eval_dataset.map(preprocess, remove_columns=["premise", "hypothesis", "idx", "label"])

    return train_dataset, eval_dataset