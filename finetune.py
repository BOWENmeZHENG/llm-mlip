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
        metadata={"help": "Path to database of annotations for texts."}
    )
    database_ood_path: str = field(
        default='./instance/AnnoApp_ood.sqlite',
        metadata={"help": "Path to database of annotations for out-of-distribution texts."}
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
        default=0.00005,
        metadata={"help": "learning rate for training."}
    )
    linear_probe: bool = field(
        default=False,
        metadata={"help": "Whether to using linear probing method, i.e. only update the last-layer weights."}
    )
    training_percentage: float = field(
        default=0.9,
        metadata={"help": "Percentage of total data for training."}
    )
    seed: int = field(
        default=3242,
        metadata={"help": "Random seed for everything except data shuffling."}
    )
    seed_shuffle: int = field(
        default=56834,
        metadata={"help": "Random seed for data shuffling."}
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

    # Shuffle data before training/test partition.
    ut.seed_everything(training_args.seed_shuffle)

    # Get training data from database
    records_id = ut.get_data(model_args.database_path)
    record_list_id = ut.form_record_list(records_id)

    # Training/test partition
    random.shuffle(record_list_id)
    N_train = int(training_args.training_percentage * len(record_list_id))
    record_list_train = record_list_id[:N_train]
    record_list_test = record_list_id[N_train:]
    print(f'Number of training data: {len(record_list_train)}')
    print(f'Number of test data: {len(record_list_test)}')

    # Load OOD data
    records_ood = ut.get_data(model_args.database_ood_path)
    record_list_ood = ut.form_record_list(records_ood)
    print(f'Number of OOD data: {len(records_ood)}')

    # Load pretrained model
    ut.seed_everything(training_args.seed)
    tokenizerBERT = BertTokenizer.from_pretrained(model_args.model_name, model_max_length=training_args.max_seq_length)
    modelBERT = BertForMaskedLM.from_pretrained(model_args.model_name)
    model = net.NERBERTModel(modelBERT.base_model, output_size=len(CLASSES)+1)

    # Run training
    pred_test, pred_ood = train(
        model, tokenizerBERT,
        record_list_train, record_list_test, record_list_ood, CLASSES, 
        training_args.train_batch_size, training_args.seed, training_args.max_seq_length, training_args.classes_weights, 
        training_args.learning_rate, training_args.n_epochs, training_args.linear_probe,
        plot=other_args.plot, save_model=other_args.save_model, save_results=other_args.save_results
        )

    # Quick view of test annotation
    if other_args.view_test:
        sample_test_id = random.randint(0, len(record_list_test) - 1)
        word_test_list = record_list_test[sample_test_id]['words']
        predictions_test = pred_test[sample_test_id, :, :].max(dim=0)[1]
        real_preds_test = predictions_test[:len(word_test_list)]
        ut.show_pred(real_preds_test, word_test_list)

        sample_ood_id = random.randint(0, len(record_list_ood) - 1)
        word_ood_list = record_list_test[sample_ood_id]['words']
        predictions_ood = pred_ood[sample_ood_id, :, :].max(dim=0)[1]
        real_preds_ood = predictions_ood[:len(word_ood_list)]
        ut.show_pred(real_preds_ood, word_ood_list)

if __name__ == "__main__":
    main()