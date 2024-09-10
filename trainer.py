import torch
import torch.nn as nn
import utils as ut
import matplotlib.pyplot as plt
from datetime import date
import os
import numpy as np
import pandas as pd
import json

def train(model, tokenizer, record_list_train, record_list_test, record_list_ood, classes, 
          batch_size, seed, max_length, class_weights: list, lr, n_epochs, linear_probe=False,
          plot=True, save_model=True, save_results=True):
    folder = f'{date.today()}_l_{lr}_lp_{linear_probe}_w_{class_weights}_b_{batch_size}_s_{seed}'
    os.makedirs(f'saved_models/{folder}', exist_ok=True)
    data_batches, target_batches, att_mask_batches = ut.preprocess(record_list=record_list_train, classes=classes, tokenizer=tokenizer, 
                                                                   batch_size=batch_size, max_length=max_length, test=False)
    weights = torch.tensor(class_weights)
    weights_n = weights / torch.norm(weights)
    weights_n = torch.cat((weights_n, torch.tensor([0])))  # weights for padding = 0
    criterion = nn.CrossEntropyLoss(weight=weights_n)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    if linear_probe:
        # Freeze weights of all but the last layer
        for param in model.bert.parameters():
            param.requires_grad = False

    train_losses = []
    train_accuracies, train_precisions, train_recalls, train_f1s = [], [], [], []
    test_accuracies, test_precisions, test_recalls, test_f1s = [], [], [], []
    ood_accuracies, ood_precisions, ood_recalls, ood_f1s = [], [], [], []

    for epoch in range(n_epochs):
        epoch += 1
        train_loss_batch = []
        train_accuracy_batch = []
        train_precision_batch, train_recall_batch, train_f1_batch = [], [], []
        for b, X in enumerate(data_batches):
            y_pred = model(X, attention_mask=att_mask_batches[b])
            y_pred = torch.swapaxes(y_pred, 1, 2)
            y = target_batches[b]            
            loss = criterion(y_pred, y)
            acc, *_ = ut.accuracy(0, len(classes), y_pred, y)
            precision, recall, f1 = ut.scores(0, len(classes), y_pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss_batch.append(loss.item())
            train_accuracy_batch.append(acc)
            train_precision_batch.append(precision)
            train_recall_batch.append(recall)
            train_f1_batch.append(f1)
        
        train_loss_batch_mean = ut.calc_mean(train_loss_batch)
        train_accuracy_batch_mean = ut.calc_mean(train_accuracy_batch)
        train_precision_batch_mean = ut.calc_mean(train_precision_batch)
        train_recall_batch_mean = ut.calc_mean(train_recall_batch)
        train_f1_batch_mean = ut.calc_mean(train_f1_batch)
        print(f'Epoch {epoch}')    
        print(f'Mean training loss: {train_loss_batch_mean:.4f}')
        print(f'Mean training accuracy: {train_accuracy_batch_mean:.4f}')
        print(f'Mean training precision: {train_precision_batch_mean:.4f}')
        print(f'Mean training recall: {train_recall_batch_mean:.4f}')
        print(f'Mean training f1: {train_f1_batch_mean:.4f}')
        train_losses.append(train_loss_batch_mean)
        train_accuracies.append(train_accuracy_batch_mean)
        train_precisions.append(train_precision_batch_mean)
        train_recalls.append(train_recall_batch_mean)
        train_f1s.append(train_f1_batch_mean)

        precision_test, recall_test, f1_test, acc_test, _, _, pred_test, true_test = testing(model, record_list_test, classes, tokenizer, max_length)
        print(f'Mean test accuracy: {acc_test:.4f}')
        print(f'Mean test precision: {precision_test:.4f}')
        print(f'Mean test recall: {recall_test:.4f}')
        print(f'Mean test f1: {f1_test:.4f}')
        test_accuracies.append(acc_test)
        test_precisions.append(precision_test)
        test_recalls.append(recall_test)
        test_f1s.append(f1_test)
            
        precision_ood, recall_ood, f1_ood, acc_ood, _, _, pred_ood, true_ood = testing(model, record_list_ood, classes, tokenizer, max_length)
        print(f'Mean test_ood accuracy: {acc_ood:.4f}')
        print(f'Mean test_ood precision: {precision_ood:.4f}')
        print(f'Mean test_ood recall: {recall_ood:.4f}')
        print(f'Mean test_ood f1: {f1_ood:.4f}')
        ood_accuracies.append(acc_ood)
        ood_precisions.append(precision_ood)
        ood_recalls.append(recall_ood)
        ood_f1s.append(f1_ood)
            
        # Save model and results    
        model_name = f'{folder}/{folder}_e_{epoch}'
        if save_model:             
            torch.save(model.state_dict(), f"saved_models/{model_name}.pt")
        ut.save_annotations(record_list_test, true_test, pred_test, model_name)
        ut.save_annotations(record_list_ood, true_ood, pred_ood, model_name)
        # for sample_id, d in enumerate(record_list_test):
        #     words_test = d['words']
        #     labels_test = true_all[sample_id][:len(words_test)].tolist()
        #     pred_test = pred_all[sample_id, :, :].max(dim=0)[1][:len(words_test)].tolist()
        #     data_test_dict = {'words': words_test, 'labels': labels_test, 'pred': pred_test}
        #     with open(f"saved_models/{model_name}_test_{sample_id}.json", 'w') as f_test:
        #         json.dump(data_test_dict, f_test)
        # for sample_id, d in enumerate(record_list_ood):
        #     words_test_ood = d['words']
        #     labels_test_ood = true_all_ood[sample_id][:len(words_test_ood)].tolist()
        #     pred_test_ood = pred_all_ood[sample_id, :, :].max(dim=0)[1][:len(words_test_ood)].tolist()
        #     data_test_ood_dict = {'words': words_test_ood, 'labels': labels_test_ood, 'pred': pred_test_ood}
        #     with open(f"saved_models/{model_name}_test_ood_{sample_id}.json", 'w') as f_test_ood:
        #         json.dump(data_test_ood_dict, f_test_ood)

    if save_results:
        train_losses_np = np.array(train_losses)
        train_accuracies_np = np.array(train_accuracies)
        train_precisions_np = np.array(train_precisions)
        train_recalls_np = np.array(train_recalls)
        train_f1s_np = np.array(train_f1s)
        test_accuracies_np = np.array(test_accuracies)
        test_precisions_np = np.array(test_precisions)
        test_recalls_np = np.array(test_recalls)
        test_f1s_np = np.array(test_f1s)
        ood_accuracies_np = np.array(ood_accuracies)
        ood_precisions_np = np.array(ood_precisions)
        ood_recalls_np = np.array(ood_recalls)
        ood_f1s_np = np.array(ood_f1s)
        data = np.vstack((train_losses_np, train_accuracies_np, train_precisions_np, train_recalls_np, train_f1s_np,
                          test_accuracies_np, test_precisions_np, test_recalls_np, test_f1s_np,
                          ood_accuracies_np, ood_precisions_np, ood_recalls_np, ood_f1s_np)).T
        data_df = pd.DataFrame(data, columns=['train_loss', 'train_accuracy', 'train_precision', 'train_recall', 'train_f1',
                                'test_accuracy', 'test_precision', 'test_recall', 'test_f1',
                                'ood_accuracy', 'ood_precision', 'ood_recall', 'ood_f1'])
        # else:
        #     data = np.vstack((train_losses_np, train_accuracies_np, train_precisions_np, train_recalls_np, train_f1s_np)).T
        #     data_df = pd.DataFrame(data, columns=['train_loss', 'train_accuracy', 'train_precision', 'train_recall', 'train_f1'])
        data_df.to_csv(f'saved_models/{folder}/results.csv', index=False)
    
    if plot:
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, sharex=True, figsize=(4, 8))
        fig.tight_layout()

        ax1.plot(range(1, n_epochs + 1), train_losses, 'o-', c='red', label='training')
        ax1.legend(fontsize=12)
        ax1.set_ylabel("loss", fontsize=12)

        ax2.plot(range(1, n_epochs + 1), train_precisions, 'o-', c='blue', label='training')
        ax2.plot(range(1, n_epochs + 1), test_precisions, 'o-', c='green', label='test')
        ax2.plot(range(1, n_epochs + 1), ood_precisions, 'o-', c='magenta', label='OOD')
        ax2.legend(fontsize=12)
        ax2.set_ylabel("precision", fontsize=12)

        ax3.plot(range(1, n_epochs + 1), train_recalls, 'o-', c='blue', label='training')
        ax3.plot(range(1, n_epochs + 1), test_recalls, 'o-', c='green', label='test')
        ax3.plot(range(1, n_epochs + 1), ood_recalls, 'o-', c='magenta', label='OOD')
        ax3.legend(fontsize=12)
        ax3.set_ylabel("recall", fontsize=12)

        ax4.plot(range(1, n_epochs + 1), train_f1s, 'o-', c='blue', label='training')
        ax4.plot(range(1, n_epochs + 1), test_f1s, 'o-', c='green', label='test')
        ax4.plot(range(1, n_epochs + 1), ood_f1s, 'o-', c='magenta', label='OOD')
        ax4.legend(fontsize=12)
        ax4.set_xlabel("epoch", fontsize=12)
        ax4.set_ylabel("F1 score", fontsize=12)

        plt.show()

    return pred_test, pred_ood


def testing(model, record_list, classes, tokenizer, max_length):
    data_test, target_test, att_mask_test = ut.preprocess(record_list, classes, tokenizer, 
                                                          batch_size=0, max_length=max_length, test=True)
    with torch.no_grad():
        y_pred_test = model(data_test, attention_mask=att_mask_test)
        y_pred_test = torch.swapaxes(y_pred_test, 1, 2)
        acc_test, predicted_classes, true_classes = ut.accuracy(0, len(classes), y_pred_test, target_test)
        precision_test, recall_test, f1_test = ut.scores(0, len(classes), y_pred_test, target_test)
    return precision_test, recall_test, f1_test, acc_test, predicted_classes, true_classes, y_pred_test, target_test