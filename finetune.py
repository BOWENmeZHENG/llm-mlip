import logging
import sys
import os
from dataclasses import dataclass, field
import random
import utils as ut
import net
from trainer import train
from transformers import HfArgumentParser, BertForMaskedLM, BertTokenizer
import warnings
warnings.filterwarnings('ignore')


CLASSES = {'MATERIAL': 1, 'MLIP': 2, 'PROPERTY': 3, 'VALUE': 4, 'APPLICATION': 5, 'O': 0}

@dataclass
class ModelArguments:
    model_name: str = field(
        default='pranav-s/MaterialsBERT',
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    database_path: str = field(
        default='./instance/AnnoApp.sqlite',
        metadata={"help": "Path to database of annotations."}
    )

@dataclass
class TrainingArguments:
    output_dir: str = field(
        default='./output',
        metadata={"help": "The output directory for logs."}
    )
    n_epochs: int = field(
        default=3,
        metadata={"help": "Total number of training epochs to perform."}
    )
    classes_weights: tuple = field(
        default=(0.3, 1., 1., 1., 0.5, 0.5),
        metadata={"help": "weights for each NER class."}
    )
    train_batch_size: int = field(
        default=1,
        metadata={"help": "Batch size for training."}
    )
    max_seq_length: int = field(
        default=512,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated."
        },
    )
    learning_rate: float = field(
        default=0.0001,
        metadata={"help": "learning rate for training."}
    )
    training_percentage: float = field(
        default=0.9,
        metadata={"help": "Percentage of total data for training."}
    )
    seed: int = field(
        default=3242,
        metadata={"help": "Random seed."}
    )
    seed_shuffle: int = field(
        default=56834,
        metadata={"help": "Random seed."}
    )

@dataclass
class OtherArguments:
    plot: bool = field(
        default=True,
        metadata={"help": "Whether to plot results."}
    )
    save_model: bool = field(
        default=True,
        metadata={"help": "Whether to save the model parameters."}
    )
    save_results: bool = field(
        default=True,
        metadata={"help": "Whether to save the prediction results."}
    )
    view_test: bool = field(
        default=True,
        metadata={"help": "Whether to view a random test data."}
    )


def main():
    parser = HfArgumentParser((ModelArguments, TrainingArguments, OtherArguments))
    
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass a json file, parse it to dictionary
        model_args, training_args, other_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, training_args, other_args = parser.parse_args_into_dataclasses()

    # Setup logging
    os.makedirs(training_args.output_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.FileHandler(f'{training_args.output_dir}/finetune.log')],
    )
    logging.info(f"Parameters {model_args}, {training_args}, {other_args}")

    # Set seed before initializing model.
    ut.seed_everything(training_args.seed_shuffle)

    # Get training data from database
    records = ut.get_data(model_args.database_path)
    record_list = ut.form_record_list(records)
    random.shuffle(record_list)
    N_train = int(training_args.training_percentage * len(record_list))
    record_list_train = record_list[:N_train]
    record_list_test = record_list[N_train:]
    print(f'Number of training data: {len(record_list_train)}')
    print(f'Number of test data: {len(record_list_test)}')

    # Load pretrained model
    ut.seed_everything(training_args.seed)
    tokenizerBERT = BertTokenizer.from_pretrained(model_args.model_name, model_max_length=training_args.max_seq_length)
    modelBERT = BertForMaskedLM.from_pretrained(model_args.model_name)
    model = net.NERBERTModel(modelBERT.base_model, output_size=len(CLASSES)+1)

    # Run training
    *_, pred_all, _, rec_list = train(
        model, tokenizerBERT,
        record_list_train, record_list_test, CLASSES, 
        training_args.train_batch_size, training_args.seed, training_args.max_seq_length, training_args.classes_weights, 
        training_args.learning_rate, training_args.n_epochs, 
        plot=other_args.plot, save_model=other_args.save_model, save_results=other_args.save_results
        )

    # Quick view of test annotation
    if other_args.view_test:
        sample_id = random.randint(0, len(record_list_test) - 1)
        word_list = rec_list[sample_id]['words']
        predictions = pred_all[sample_id, :, :].max(dim=0)[1]
        real_preds = predictions[:len(word_list)]
        ut.show_pred(real_preds, word_list)

if __name__ == "__main__":
    main()