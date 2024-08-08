import logging
import sys
import os
from dataclasses import dataclass, field
# from typing import Optional
# import random
# import utils_train as ut
# import net
# from trainer import train
from transformers import HfArgumentParser, BertForMaskedLM, BertTokenizer
# import matplotlib.pyplot as plt
# from highlight_text import HighlightText
# import warnings
# warnings.filterwarnings('ignore')

# logger = logging.getLogger(__name__)

@dataclass
class ModelArguments:
    model_name: str = field(
        default='pranav-s/MaterialsBERT',
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    # ner_classes: dict = field(
    #     default={'MATERIAL': 1, 'MLIP': 2, 'PROPERTY': 3, 'VALUE': 4, 'APPLICATION': 5, 'O': 0},
    #     metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    # )
    database_path: str = field(
        default='./instance/AnnoApp.sqlite',
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )

@dataclass
class TrainingArguments:
    output_dir: str = field(
        default='./output',
        metadata={"help": "The output directory where the model predictions and checkpoints will be written."}
    )
    n_epochs: int = field(
        default=3,
        metadata={"help": "Total number of training epochs to perform."}
    )
    classes_weights: tuple = field(
        default=(0.3, 1., 1., 1., 0.5, 0.5),
        metadata={"help": "Total number of training epochs to perform."}
    )
    train_batch_size: int = field(
        default=1,
        metadata={"help": "Batch size per GPU/TPU core/CPU for training."}
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
        metadata={"help": "Batch size per GPU/TPU core/CPU for training."}
    )
    training_percentage: float = field(
        default=0.9,
        metadata={"help": "Batch size per GPU/TPU core/CPU for training."}
    )
    seed: int = field(
        default=3242,
        metadata={"help": "Batch size per GPU/TPU core/CPU for training."}
    )

def main():
    parser = HfArgumentParser((ModelArguments, TrainingArguments))
    
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass a json file, parse it to dictionary
        model_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, training_args = parser.parse_args_into_dataclasses()

    print(model_args)
    print(training_args)

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logging.info(f"Training/evaluation parameters {training_args}")



# @dataclass
# class ModelArguments:
#     """
#     Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
#     """

#     n_epochs: str = field(
#         default="6",
#         metadata={
#             "help": "try"
#         },
#     )

    # CLASSES = {'MATERIAL': 1, 'MLIP': 2, 'PROPERTY': 3,
    #         'VALUE': 4, 'APPLICATION': 5, 'O': 0}
    # BATCH_SIZE = 1
    # SEED = 3242
    # MAX_LENGTH = 512
    # CLASS_WEIGHTS = [0.3, 1., 1., 1., 0.5, 0.5]
    # LEARNING_RATE = 0.0001
    # N_EPOCHS = training_args.n_epochs
    # print(N_EPOCHS)
    # TRAIN_PCT = 0.9
    # DB_PATH = './instance/AnnoApp.sqlite'

# ut.seed_everything(SEED)
# posts = ut.get_data(DB_PATH)
# record_list = ut.form_record_list(posts)
# random.shuffle(record_list)
# N_train = int(TRAIN_PCT * len(record_list))
# record_list_train = record_list[:N_train]
# record_list_test = record_list[N_train:]
# print(f'Number of training data: {len(record_list_train)}')
# print(f'Number of test data: {len(record_list_test)}')
# tokenizerBERT = BertTokenizer.from_pretrained('pranav-s/MaterialsBERT', model_max_length=MAX_LENGTH)
# modelBERT = BertForMaskedLM.from_pretrained('pranav-s/MaterialsBERT')
# model = net.NERBERTModel(modelBERT.base_model, output_size=len(CLASSES)+1)

# model, train_losses, train_accuracies, preczision_test, recall_test, f1_test, test_accuracies, pred_classes, true_classes, pred_all, true_all, rec_list = train(model, tokenizerBERT,
#    record_list_train, record_list_test, CLASSES, BATCH_SIZE, SEED, MAX_LENGTH, CLASS_WEIGHTS, LEARNING_RATE, N_EPOCHS, plot=True, save_model=True)

# sample_id = 0
# word_list = rec_list[sample_id]['words']
# labels = true_all
# predictions = pred_all[sample_id, :, :].max(dim=0)[1]
# colors = ['black', 'red', 'blue', 'green', 'cyan', 'darkorange']
# real_preds = predictions[:len(word_list)]
# word_list_marked = ['<' + w + '>' for w in word_list]
# markers = [{"color": colors[i]} for i in real_preds]
# j = 0
# for i in range(len(word_list)):
#     if (i + 1) % 15 == 0:
#         word_list_marked.insert(i + j, '\n')
#         j += 1
# fig, ax = plt.subplots()
# ax.set_axis_off()
# HighlightText(x=0., y=1, s='<O>, <MATERIAL>, <MLIP>, <PROPERTY>, <VALUE>, <APPLICATION>',
#               highlight_textprops=[{"color": c} for c in colors], ax=ax)
# HighlightText(x=0., y=0.9, s=' '.join(word_list_marked),
#               highlight_textprops=markers, ax=ax)
# plt.show()

if __name__ == "__main__":
    main()