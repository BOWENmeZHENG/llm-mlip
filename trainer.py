import torch
import torch.nn as nn
import utils_train as ut
import matplotlib.pyplot as plt
from datetime import date
import os
import numpy as np
import pandas as pd
import json

# change
def train(model, tokenizer, record_list_train, record_list_test, classes, 
          batch_size, seed, max_length, class_weights: list, lr, n_epochs,
          plot=True, save_model=True, save_results=True):
    folder = f'{date.today()}_b_{batch_size}_s_{seed}_l_{max_length}_w_{class_weights}_l_{lr}'
    os.makedirs(f'saved_models/{folder}', exist_ok=True)
    data_batches, target_batches, att_mask_batches = ut.preprocess(record_list=record_list_train, classes=classes, tokenizer=tokenizer, 
                                                                   batch_size=batch_size, max_length=max_length, test=False)
    weights = torch.tensor(class_weights)
    weights_n = weights / torch.norm(weights)
    weights_n = torch.cat((weights_n, torch.tensor([0])))  # weights for padding = 0
    criterion = nn.CrossEntropyLoss(weight=weights_n)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_losses = []
    train_accuracies = []
    train_precisions, train_recalls, train_f1s = [], [], []
    if record_list_test != None:
        test_accuracies = []
        test_precisions, test_recalls, test_f1s = [], [], []
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
        
        train_loss_batch_mean = sum(train_loss_batch) / len(train_loss_batch)
        train_accuracy_batch_mean = sum(train_accuracy_batch) / len(train_accuracy_batch)
        train_precision_batch_mean = sum(train_precision_batch) / len(train_precision_batch)
        train_recall_batch_mean = sum(train_recall_batch) / len(train_recall_batch)
        train_f1_batch_mean = sum(train_f1_batch) / len(train_f1_batch)
        print(f'Epoch {epoch}')    
        print(f'Mean training loss: {train_loss_batch_mean:.4f}')
        print(f'Mean training accuracy: {train_accuracy_batch_mean:.4f}')
        print(f'Mean training precision: {train_precision_batch_mean:.4f}')
        print(f'Mean training recall: {train_recall_batch_mean:.4f}')
        print(f'Mean training f1: {train_f1_batch_mean:.4f}')
        if record_list_test != None:
            precision_test, recall_test, f1_test, acc_test, pred_classes, true_classes, pred_all, true_all, record_list = testing(model, record_list_test, classes, tokenizer, max_length)
            test_accuracies.append(acc_test)
            test_precisions.append(precision_test)
            test_recalls.append(recall_test)
            test_f1s.append(f1_test)
            
            print(f'Mean test accuracy: {acc_test:.4f}')
            print(f'Mean test precision: {precision_test:.4f}')
            print(f'Mean test recall: {recall_test:.4f}')
            print(f'Mean test f1: {f1_test:.4f}')
        print('\n')
        train_losses.append(train_loss_batch_mean)
        train_accuracies.append(train_accuracy_batch_mean)
        train_precisions.append(train_precision_batch_mean)
        train_recalls.append(train_recall_batch_mean)
        train_f1s.append(train_f1_batch_mean)

        model_name = f'{folder}/{folder}_e_{epoch}'
        if save_model:             
            torch.save(model.state_dict(), f"saved_models/{model_name}.pt")
        for sample_id, d in enumerate(record_list):
            words_test = d['words']
            labels_test = true_all[sample_id][:len(words_test)].tolist()
            pred_test = pred_all[sample_id, :, :].max(dim=0)[1][:len(words_test)].tolist()
            data_test_dict = {'words': words_test, 'labels': labels_test, 'pred': pred_test}
            with open(f"saved_models/{model_name}_test_{sample_id}.json", 'w') as f_test:
                json.dump(data_test_dict, f_test)

    if save_results:
        train_losses_np = np.array(train_losses)
        train_accuracies_np = np.array(train_accuracies)
        train_precisions_np = np.array(train_precisions)
        train_recalls_np = np.array(train_recalls)
        train_f1s_np = np.array(train_f1s)
        if record_list_test != None:
            test_accuracies_np = np.array(test_accuracies)
            test_precisions_np = np.array(test_precisions)
            test_recalls_np = np.array(test_recalls)
            test_f1s_np = np.array(test_f1s)
            data = np.vstack((train_losses_np, train_accuracies_np, train_precisions_np, train_recalls_np, train_f1s_np,
                              test_accuracies_np, test_precisions_np, test_recalls_np, test_f1s_np)).T
            data_df = pd.DataFrame(data, columns=['train_loss', 'train_accuracy', 'train_precision', 'train_recall', 'train_f1',
                                  'test_accuracy', 'test_precision', 'test_recall', 'test_f1'])
            
        else:
            data = np.vstack((train_losses_np, train_accuracies_np, train_precisions_np, train_recalls_np, train_f1s_np)).T
            data_df = pd.DataFrame(data, columns=['train_loss', 'train_accuracy', 'train_precision', 'train_recall', 'train_f1'])
        data_df.to_csv(f'saved_models/{folder}/results.csv', index=False)
    
    if plot:
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, sharex=True, figsize=(4, 8))
        fig.tight_layout()

        ax1.plot(range(1, n_epochs + 1), train_losses, 'o-', c='red', label='training')
        ax1.legend(fontsize=12)
        ax1.set_ylabel("loss", fontsize=12)

        ax2.plot(range(1, n_epochs + 1), train_precisions, 'o-', c='blue', label='training')
        if record_list_test != None:
            ax2.plot(range(1, n_epochs + 1), test_precisions, 'o-', c='green', label='test')
        ax2.legend(fontsize=12)
        ax2.set_ylabel("precision", fontsize=12)

        ax3.plot(range(1, n_epochs + 1), train_recalls, 'o-', c='blue', label='training')
        if record_list_test != None:
            ax3.plot(range(1, n_epochs + 1), test_recalls, 'o-', c='green', label='test')
        ax3.legend(fontsize=12)
        ax3.set_ylabel("recall", fontsize=12)

        ax4.plot(range(1, n_epochs + 1), train_f1s, 'o-', c='blue', label='training')
        if record_list_test != None:
            ax4.plot(range(1, n_epochs + 1), test_f1s, 'o-', c='green', label='test')
        ax4.legend(fontsize=12)
        ax4.set_xlabel("epoch", fontsize=12)
        ax4.set_ylabel("F1 score", fontsize=12)

        plt.show()

    if record_list_test != None:
        return model, train_losses, train_accuracies, precision_test, recall_test, f1_test, test_accuracies, pred_classes, true_classes, pred_all, true_all, record_list
    else:
        return model, train_losses, train_accuracies

def testing(model, record_list, classes, tokenizer, max_length):
    data_test, target_test, att_mask_test, data_list = ut.preprocess(record_list, classes, tokenizer, 
                                                          batch_size=0, max_length=max_length, test=True)
    with torch.no_grad():
        y_pred_test = model(data_test, attention_mask=att_mask_test)
        y_pred_test = torch.swapaxes(y_pred_test, 1, 2)
        acc_test, predicted_classes, true_classes = ut.accuracy(0, len(classes), y_pred_test, target_test)
        precision_test, recall_test, f1_test = ut.scores(0, len(classes), y_pred_test, target_test)
    return precision_test, recall_test, f1_test, acc_test, predicted_classes, true_classes, y_pred_test, target_test, data_list