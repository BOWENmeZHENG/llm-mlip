import logging
import os
from dataclasses import dataclass, field
import utils as ut
import net
import torch
from transformers import HfArgumentParser, BertForMaskedLM, BertTokenizer
import warnings
warnings.filterwarnings('ignore')

CLASSES = {'MATERIAL': 1, 'MLIP': 2, 'PROPERTY': 3, 'VALUE': 4, 'APPLICATION': 5, 'O': 0}

@dataclass
class ModelArguments:
    base_model: str = field(
        default='pranav-s/MaterialsBERT',
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    finetuned_model: str = field(
        default='2024-08-08_b_1_s_4232_l_512_w_(0.3, 1.0, 1.0, 1.0, 0.5, 0.5)_l_0.0001',
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    model_epoch: int = field(
        default=6,
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    database_path: str = field(
        default='./instance/AnnoApp.sqlite',
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    max_seq_length: int = field(
        default=512,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated."
        },
    )

@dataclass
class OtherArguments:
    output_dir: str = field(
        default='./output',
        metadata={"help": "The output directory where the model predictions and checkpoints will be written."}
    )

def main():
    parser = HfArgumentParser((ModelArguments, OtherArguments))
    model_args, other_args = parser.parse_args_into_dataclasses()
    os.makedirs(other_args.output_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.FileHandler(f'{other_args.output_dir}/inference.log')],
    )
    logging.info(f"Parameters {model_args}, {other_args}")

    # Load pretrained model
    tokenizerBERT = BertTokenizer.from_pretrained(model_args.base_model, model_max_length=model_args.max_seq_length)
    modelBERT = BertForMaskedLM.from_pretrained(model_args.base_model)
    model = net.NERBERTModel(modelBERT.base_model, output_size=len(CLASSES)+1)
    parameter_path = f'./saved_models/{model_args.finetuned_model}/{model_args.finetuned_model}_e_{model_args.model_epoch}.pt'
    model.load_state_dict(torch.load(parameter_path))
    model.eval()
    print('Model loaded')

    # Annotate
    records = []
    with open('texts2annotate.txt', 'r') as file:
        for line in file:
            records.append(line.strip())

    results = ut.inference_batch(records, model, tokenizerBERT, model_args.max_seq_length)
    for result in results:
        ut.show_pred(*result)

    # Send annotation to database
    CLASSES_inv = {value: key for key, value in CLASSES.items()}
    data_AI = ut.result2data_batch(results, CLASSES_inv)
    ut.insert_data(data_AI, model_args.database_path)

if __name__ == "__main__":
    main()