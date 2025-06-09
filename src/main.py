from comms import LossyNetwork, GillbertElliotLossyNetwork
from trainer import DistributedTrainer, MyClassifierCallback, MyQACallback, MyQATrainer, compute_classfication_metrics, compute_exact_match_metric
from data import get_dataset
from transformers import TrainingArguments
import os
import pandas as pd
import yaml
from models import get_classifier_and_tokenizer

classification_datasets = ['winogrande', 'mnli', 'sst2', 'hellaswag', 'piqa', 'arc', 'quality']
generation_datasets = ['hotpotqa']
def main(args):

    with open("src/dataset_config.yaml") as config:
        try:
            dataset_config = yaml.safe_load(config)
        except yaml.YAMLError as exc:
            print(exc)
    
    dataset_config = dataset_config[args.dataset]
    loss_type = args.loss_type
    if loss_type == 'ber':
        network = LossyNetwork(args)
    elif loss_type == 'g-e':
        configs = pd.read_csv('g_e_params.csv')
        ge_config = configs[configs['id'] == args.ge_config].iloc[0]
        network = GillbertElliotLossyNetwork(p_bg = ge_config['p_bg'],p_gb= ge_config['p_gb'],
                                             good_loss_rate=ge_config['good_loss_rate'],
                                             bad_loss_rate=ge_config['bad_loss_rate'], args=args)
    else:
        raise ValueError(f"Unsupported loss type: {loss_type}")
    network.set_seed(args.seed)

    # for tasks other than classification you will need to modify the callback and the compute_metrics function, as well as get model and tokenizer
    if args.dataset in classification_datasets:
        model, tokenizer = get_classifier_and_tokenizer(args.model_name, num_labels=dataset_config['num_labels'], num_unfrozen_layers=args.num_unfrozen_layers)
        train_dataset, eval_dataset = get_dataset(args, tokenizer)
    elif args.dataset in generation_datasets:
        from models import get_qa_model_and_tokenizer
        model, tokenizer = get_qa_model_and_tokenizer(args.model_name, num_unfrozen_layers=args.num_unfrozen_layers)
        train_dataset, eval_dataset = get_dataset(args, tokenizer)

    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    output_dir = f"{args.output_dir}/{args.run_id}"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(f"{output_dir}/args.yaml", "w") as f:
        yaml.dump(vars(args), f) # for reproducibility

    args.output_dir = output_dir

    callback_args = { # report time to accuracy #TODO change this to "steps to accuracy"
        'report_ttac' : dataset_config['report_ttac'],
        'report_file' : f"{args.output_dir}/ttac_report.txt",
        'target_acc': dataset_config['target_acc'],
    }
    if args.dataset in generation_datasets:
        callback_args['eos_token_id'] = tokenizer.eos_token_id
        compute_metrics = compute_exact_match_metric(tokenizer)
        callback = MyQACallback(callback_args)
        trainer_class = MyQATrainer
    else:
        compute_metrics = compute_classfication_metrics
        callback = MyClassifierCallback(callback_args)
        trainer_class = DistributedTrainer
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        learning_rate= args.learning_rate,
        weight_decay=0.01,
        evaluation_strategy="steps",
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        save_strategy="steps",
        save_total_limit=1,
        metric_for_best_model="accuracy" if args.dataset in classification_datasets else "exact_match",
        logging_dir=f"{output_dir}/logs",
        logging_steps=10,
        fp16=args.fp16,
        report_to="wandb"
    )

    trainer = trainer_class(
        num_nodes=args.num_nodes,
        network=network,
        model=model,
        tokenizer=tokenizer,
        args = training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        callbacks=[callback],
        compute_metrics=compute_metrics,
    )

    trainer.train()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Distributed Training with Packet Loss")
    parser.add_argument('--num_nodes', type=int, default=2, help='Number of nodes')
    parser.add_argument('--loss_rate', type=float, default=0.001, help='Packet loss rate When using Bernoulli')
    parser.add_argument('--loss_type', type=str, default='ber', choices=['ber', 'g-e'], help='Type of packet loss simulation: "ber" for Bernoulli, "g-e" for Gilbert-Elliott')
    parser.add_argument('--ge_config', type = str, default = 'default', help='configuration id for Gilbert-Elliott loss simulation. Refer to g_e_params.csv')
    parser.add_argument('--seed', type=int, default=1234, help='Random seed')
    parser.add_argument('--model_name', type=str, default='meta-llama/Llama-3.2-1B', help='Model name')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--fp16', action='store_true', help='Use mixed precision training')
    parser.add_argument('--output_dir', type=str, default='./output', help='Output directory')
    parser.add_argument('--dataset', '-d', type=str, default='winogrande', 
                        help='Dataset to use for training')
    parser.add_argument('--max_samples', type=int, default=0, 
                        help='Maximum number of training samples to use (0 for all)')
    parser.add_argument('--epochs', type=int, default=3, 
                        help='Number of training epochs')
    parser.add_argument('--max_length', type=int, default=256,
                        help='Maximum sequence length for tokenization')
    parser.add_argument('--eval_steps', type=int, default=50)
    parser.add_argument('--save_steps', type=int, default=100)
    parser.add_argument('--logging_steps', type=int, default=10)
    parser.add_argument('--learning_rate', type=float, default=3e-5)
    parser.add_argument('--run_id', type=str, required=True)
    parser.add_argument('-nunf', '--num_unfrozen_layers', type=int, default=None, 
                        help='Number of unfrozen layers in the model. If None, all layers are unfrozen.')
    args = parser.parse_args()
    
    main(args)
