import pandas as pd
import os
import io
import fnmatch
import matplotlib.pyplot as plt
import numpy as np
import json
from dataclasses import dataclass, field
from transformers import HfArgumentParser
from highlight_text import HighlightText

@dataclass
class ResultArguments:
    stem: str = field(
        default=None,
        metadata={"help": "Stem of model folder."}
    )
    epoch: str = field(
        default=None,
        metadata={"help": "Date of model training."}
    )

@dataclass
class OtherArguments:
    plot_results: bool = field(
        default=True,
        metadata={"help": "Whether to plot results."}
    )
    save_summary: bool = field(
        default=True,
        metadata={"help": "Whether to plot results."}
    )
    viz_test: bool = field(
        default=False,
        metadata={"help": "Whether to visualize annotation on test data."}
    )

def get_folders_starting_with(path, start_name):
    # List all items in the directory
    items = os.listdir(path)
    # Filter only directories that start with the given name
    folders = [item for item in items if os.path.isdir(os.path.join(path, item)) and fnmatch.fnmatch(item, f'{start_name}*')]
    return folders

def get_files_starting_with(path, start_name):
    # List all items in the directory
    items = os.listdir(path)
    # Filter only directories that start with the given name
    files = [item for item in items if os.path.isfile(os.path.join(path, item)) and fnmatch.fnmatch(item, f'{start_name}*')]
    return files

def mean_std(res_list, metric):
    vals = []
    for res in res_list:
        vals.append(res[metric])
    vals = np.array(vals)
    return np.mean(vals, axis=0), np.std(vals, axis=0)
     

def main():
    parser = HfArgumentParser((ResultArguments, OtherArguments))
    result_args, other_args = parser.parse_args_into_dataclasses()
    folders = get_folders_starting_with('saved_models/', result_args.stem)
    results = []
    for f in folders:
        result = pd.read_csv(f'saved_models/{f}/results.csv')
        results.append(result)

    n_epochs = len(results[0])
    mean_train_loss, std_train_loss = mean_std(results, 'train_loss')
    mean_train_accuracy, std_train_accuracy = mean_std(results, 'train_accuracy')
    mean_test_accuracy, std_test_accuracy = mean_std(results, 'test_accuracy')
    mean_ood_accuracy, std_ood_accuracy = mean_std(results, 'ood_accuracy')
    mean_train_precision, std_train_precision = mean_std(results, 'train_precision')
    mean_test_precision, std_test_precision = mean_std(results, 'test_precision')
    mean_ood_precision, std_ood_precision = mean_std(results, 'ood_precision')
    mean_train_recall, std_train_recall = mean_std(results, 'train_recall')
    mean_test_recall, std_test_recall = mean_std(results, 'test_recall')
    mean_ood_recall, std_ood_recall = mean_std(results, 'ood_recall')
    mean_train_f1, std_train_f1 = mean_std(results, 'train_f1')
    mean_test_f1, std_test_f1 = mean_std(results, 'test_f1')
    mean_ood_f1, std_ood_f1 = mean_std(results, 'ood_f1')
    if other_args.save_summary:
        mean_train_loss_np, std_train_loss_np = np.array(mean_train_loss), np.array(std_train_loss)
        mean_train_accuracy_np, std_train_accuracy_np = np.array(mean_train_accuracy), np.array(std_train_accuracy)
        mean_test_accuracy_np, std_test_accuracy_np = np.array(mean_test_accuracy), np.array(std_test_accuracy)
        mean_ood_accuracy_np, std_test_ood_accuracy_np = np.array(mean_ood_accuracy), np.array(std_ood_accuracy)
        mean_train_precision_np, std_train_precision_np = np.array(mean_train_precision), np.array(std_train_precision)
        mean_test_precision_np, std_test_precision_np = np.array(mean_test_precision), np.array(std_test_precision)
        mean_ood_precision_np, std_test_ood_precision_np = np.array(mean_ood_precision), np.array(std_ood_precision)
        mean_train_recall_np, std_train_recall_np = np.array(mean_train_recall), np.array(std_train_recall)
        mean_test_recall_np, std_test_recall_np = np.array(mean_test_recall), np.array(std_test_recall)
        mean_ood_recall_np, std_test_ood_recall_np = np.array(mean_ood_recall), np.array(std_ood_recall)
        mean_train_f1_np, std_train_f1_np = np.array(mean_train_f1), np.array(std_train_f1)
        mean_test_f1_np, std_test_f1_np = np.array(mean_test_f1), np.array(std_test_f1)
        mean_ood_f1_np, std_test_ood_f1_np = np.array(mean_ood_f1), np.array(std_ood_f1)
        data = np.vstack((mean_train_loss_np, std_train_loss_np, 
                          mean_train_accuracy_np, std_train_accuracy_np, mean_test_accuracy_np, std_test_accuracy_np, mean_ood_accuracy_np, std_test_ood_accuracy_np,
                          mean_train_precision_np, std_train_precision_np, mean_test_precision_np, std_test_precision_np, mean_ood_precision_np, std_test_ood_precision_np,
                          mean_train_recall_np, std_train_recall_np, mean_test_recall_np, std_test_recall_np, mean_ood_recall_np, std_test_ood_recall_np,
                          mean_train_f1_np, std_train_f1_np, mean_test_f1_np, std_test_f1_np, mean_ood_f1_np, std_test_ood_f1_np)).T
        data_df = pd.DataFrame(data, columns=['mean_train_loss', 'std_train_loss', 
        'mean_train_accuracy', 'std_train_accuracy', 'mean_test_accuracy', 'std_test_accuracy', 'mean_ood_accuracy', 'std_ood_accuracy', 
        'mean_train_precision', 'std_train_precision', 'mean_test_precision', 'std_test_precision', 'mean_ood_precision', 'std_ood_precision',
        'mean_train_recall', 'std_train_recall', 'mean_test_recall', 'std_test_recall', 'mean_ood_recall', 'std_ood_recall',
        'mean_train_f1', 'std_train_f1', 'mean_test_f1', 'std_test_f1', 'mean_ood_f1', 'std_test_ood_f1'])
        data_df.to_csv(f'saved_models/results_{result_args.stem}.csv', index=False)
    
    if other_args.plot_results:
        fig, axes = plt.subplots(2, 2, sharex=True, figsize=(20, 16))

        axes[0, 0].errorbar(range(1, n_epochs + 1), mean_train_loss, yerr=std_train_loss, marker='o',
                    markersize=8, c='red', capsize=5, label='training')
        axes[0, 0].legend(fontsize=22)
        axes[0, 0].set_ylabel("loss", fontsize=22)
        axes[0, 0].tick_params(labelsize=20)

        axes[0, 1].errorbar(range(1, n_epochs + 1), mean_train_precision, yerr=std_train_precision, marker='o',
                    markersize=8, c='blue', capsize=5, label='training')
        axes[0, 1].errorbar(range(1, n_epochs + 1), mean_test_precision, yerr=std_test_precision, marker='o',
                    markersize=8, c='green', capsize=5, label='test')
        axes[0, 1].errorbar(range(1, n_epochs + 1), mean_ood_precision, yerr=std_ood_precision, marker='o',
                    markersize=8, c='magenta', capsize=5, label='OOD')
        axes[0, 1].legend(fontsize=22)
        axes[0, 1].set_ylabel("precision", fontsize=22)
        axes[0, 1].tick_params(labelsize=20)

        axes[1, 0].errorbar(range(1, n_epochs + 1), mean_train_recall, yerr=std_train_recall, marker='o',
                    markersize=8, c='blue', capsize=5, label='training')
        axes[1, 0].errorbar(range(1, n_epochs + 1), mean_test_recall, yerr=std_test_recall, marker='o',
                    markersize=8, c='green', capsize=5, label='test')
        axes[1, 0].errorbar(range(1, n_epochs + 1), mean_ood_recall, yerr=std_ood_recall, marker='o',
                    markersize=8, c='magenta', capsize=5, label='OOD')
        axes[1, 0].legend(fontsize=22)
        axes[1, 0].set_xlabel("epoch", fontsize=22)
        axes[1, 0].set_ylabel("recall", fontsize=22)
        axes[1, 0].tick_params(labelsize=20)

        axes[1, 1].errorbar(range(1, n_epochs + 1), mean_train_f1, yerr=std_train_f1, marker='o',
                    markersize=8, c='blue', capsize=5, label='training')
        axes[1, 1].errorbar(range(1, n_epochs + 1), mean_test_f1, yerr=std_test_f1, marker='o',
                    markersize=8, c='green', capsize=5, label='test')
        axes[1, 1].errorbar(range(1, n_epochs + 1), mean_ood_f1, yerr=std_ood_f1, marker='o',
                    markersize=8, c='magenta', capsize=5, label='OOD')
        axes[1, 1].legend(fontsize=22)
        axes[1, 1].set_xlabel("epoch", fontsize=22)
        axes[1, 1].set_ylabel("F1-score", fontsize=22)
        axes[1, 1].tick_params(labelsize=20)

        plt.show()


    if other_args.viz_test:
        folder = get_folders_starting_with('saved_models/', result_args.stem)[0]
        stem_file = f'{folder}_e_{result_args.epoch}_'
        files = get_files_starting_with(f'saved_models/{folder}', stem_file)
        for file in files:
            with open(f'saved_models/{folder}/{file}', encoding='utf-8') as f:
                for jsonObj in f:
                    record = json.loads(jsonObj)
            # calculate accuracy
            counts = 0
            for ii in range(len(record['pred'])):
                if record['pred'][ii] == record['labels'][ii]:
                    counts += 1
            accuracy = counts / len(record['pred'])

            colors = ['black', 'red', 'blue', 'green', 'cyan', 'darkorange']
            word_list_marked = ['<' + w + '>' for w in record['words']]
            markers_labels = [{"color": colors[i]} for i in record['labels']]
            markers_pred = [{"color": colors[i]} for i in record['pred']]
            j = 0
            for i in range(len(record['words'])):
                if (i + 1) % 15 == 0:
                    word_list_marked.insert(i + j, '\n')
                    j += 1

            fig, ax = plt.subplots(figsize=(12, 8))
            ax.set_axis_off()
            HighlightText(x=0., y=1, s='<O>, <MATERIAL>, <MLIP>, <PROPERTY>, <VALUE>, <APPLICATION>',
              highlight_textprops=[{"color": c} for c in colors], ax=ax)
            HighlightText(x=0., y=0.8, s=' '.join(word_list_marked),
                        highlight_textprops=markers_pred, ax=ax)
            plt.show()



if __name__ == "__main__":
    main()