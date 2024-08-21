import pandas as pd
import os
import fnmatch
import matplotlib.pyplot as plt
import numpy as np
from dataclasses import dataclass, field
from transformers import HfArgumentParser

@dataclass
class ResultArguments:
    # seeds_run: tuple = field(
    #     default=None,
    #     metadata={"help": "Same parameters under different seeds."}
    # )
    stem: str = field(
        default=None,
        metadata={"help": "Date of model training."}
    )

def get_folders_starting_with(path, start_name):
    # List all items in the directory
    items = os.listdir(path)
    # Filter only directories that start with the given name
    folders = [item for item in items if os.path.isdir(os.path.join(path, item)) and fnmatch.fnmatch(item, f'{start_name}*')]
    return folders

def mean_std(res_list, metric):
    vals = []
    for res in res_list:
        vals.append(res[metric])
    vals = np.array(vals)
    return np.mean(vals, axis=0), np.std(vals, axis=0)
     

def main():
    parser = HfArgumentParser(ResultArguments)
    result_args, *_ = parser.parse_args_into_dataclasses()
    folders = get_folders_starting_with('saved_models/', result_args.stem)
    results = []
    for f in folders:
        result = pd.read_csv(f'saved_models/{f}/results.csv')
        results.append(result)

    n_epochs = len(results[0])
    mean_train_loss, std_train_loss = mean_std(results, 'train_loss')
    mean_train_accuracy, std_train_accuracy = mean_std(results, 'train_accuracy')
    mean_test_accuracy, std_test_accuracy = mean_std(results, 'test_accuracy')
    mean_train_precision, std_train_precision = mean_std(results, 'train_precision')
    mean_test_precision, std_test_precision = mean_std(results, 'test_precision')
    mean_train_recall, std_train_recall = mean_std(results, 'train_recall')
    mean_test_recall, std_test_recall = mean_std(results, 'test_recall')
    mean_train_f1, std_train_f1 = mean_std(results, 'train_f1')
    mean_test_f1, std_test_f1 = mean_std(results, 'test_f1')
    
    fig, axes = plt.subplots(2, 2, sharex=True, figsize=(20, 16))
    # fig.tight_layout()

    axes[0, 0].errorbar(range(1, n_epochs + 1), mean_train_loss, yerr=std_train_loss, marker='o',
                markersize=8, c='red', capsize=5, label='training')
    axes[0, 0].legend(fontsize=22)
    axes[0, 0].set_ylabel("loss", fontsize=22)
    axes[0, 0].tick_params(labelsize=20)

    axes[0, 1].errorbar(range(1, n_epochs + 1), mean_train_precision, yerr=std_train_precision, marker='o',
                markersize=8, c='blue', capsize=5, label='training')
    axes[0, 1].errorbar(range(1, n_epochs + 1), mean_test_precision, yerr=std_test_precision, marker='o',
                markersize=8, c='green', capsize=5, label='test')
    axes[0, 1].legend(fontsize=22)
    # axes[0, 1].set_xlabel("epoch", fontsize=22)
    axes[0, 1].set_ylabel("precision", fontsize=22)
    axes[0, 1].tick_params(labelsize=20)

    axes[1, 0].errorbar(range(1, n_epochs + 1), mean_train_recall, yerr=std_train_recall, marker='o',
                markersize=8, c='blue', capsize=5, label='training')
    axes[1, 0].errorbar(range(1, n_epochs + 1), mean_test_recall, yerr=std_test_recall, marker='o',
                markersize=8, c='green', capsize=5, label='test')
    axes[1, 0].legend(fontsize=22)
    axes[1, 0].set_xlabel("epoch", fontsize=22)
    axes[1, 0].set_ylabel("recall", fontsize=22)
    axes[1, 0].tick_params(labelsize=20)

    axes[1, 1].errorbar(range(1, n_epochs + 1), mean_train_f1, yerr=std_train_f1, marker='o',
                markersize=8, c='blue', capsize=5, label='training')
    axes[1, 1].errorbar(range(1, n_epochs + 1), mean_test_f1, yerr=std_test_f1, marker='o',
                markersize=8, c='green', capsize=5, label='test')
    axes[1, 1].legend(fontsize=22)
    axes[1, 1].set_xlabel("epoch", fontsize=22)
    axes[1, 1].set_ylabel("F1-score", fontsize=22)
    axes[1, 1].tick_params(labelsize=20)

    plt.show()

if __name__ == "__main__":
    main()