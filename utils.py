import re
import torch
import json
import math
import random
import os
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
import sqlite3
import matplotlib.pyplot as plt
from highlight_text import HighlightText

def split_para(para):
    return re.findall(r"[\w]+|[-.,\'!=±?%–−;:/\(\)\[\]]", para)
    
def text2token(tokenizer, text):
    text_list = tokenizer.tokenize(text)
    token_list = tokenizer.convert_tokens_to_ids(text_list)
    token = torch.tensor(token_list)[None, :]
    return token

def list2token(tokenizer, text_list, max_length):
    token_list = tokenizer.convert_tokens_to_ids(text_list)
    token_list_padded = token_list + [0] * (max_length - len(token_list))
    token = torch.tensor(token_list_padded)[None, :]
    return token

def get_data(database):
    connection = sqlite3.connect(database)
    cursor = connection.execute("SELECT id, title, body, annotation from mlip ")
    texts = []
    for row in cursor:
        texts.append(row)
    return texts

def format_data(text):
    id, para, annotation_string = text[0], text[2], text[3]
    para_words = split_para(para)
    annotation = json.loads(annotation_string)
    ner = ["O"] * len(para_words)
    for index, cls in annotation.items():
        ner[int(index)] = cls
    ml_data = {}
    ml_data["words"], ml_data["ner"], ml_data["id"] = para_words, ner, id
    return ml_data

def form_record_list(texts, max_length=0):
    record_list = []
    for text in texts:
        record = format_data(text)
        if max_length != 0:
            if len(record['words']) > max_length:
                record['words'] = record['words'][:max_length]
                record['ner'] = record['ner'][:max_length]
        record_list.append(record)
    return record_list

def cat2digit(classes, cat_text, max_length):
    label_digit = [classes.get(item, item) for item in cat_text]
    label_digit_padded = label_digit + [len(classes)] * (max_length - len(label_digit))
    att_mask = [1] * len(label_digit) + [0] * (max_length - len(label_digit))
    return torch.tensor(label_digit_padded), torch.tensor(att_mask)

def to_batches(x, batch_size):
    num_batches = math.ceil(x.size()[0] / batch_size)
    return [x[batch_size * y: batch_size * (y+1),:] for y in range(num_batches)]

def accuracy(index_other, index_pad, y_pred, y):
    indices = (y < index_pad).nonzero(as_tuple=True)
    _, predicted_classes = y_pred[indices[0], :, indices[1]].max(dim=1)
    true_classes = y[indices[0], indices[1]]
    accuracy = torch.eq(predicted_classes, true_classes).sum() / true_classes.shape[0]
    return accuracy, predicted_classes, true_classes

def scores(index_other, index_pad, y_pred, y):
    indices = (y < index_pad).nonzero(as_tuple=True)
    _, predicted_classes = y_pred[indices[0], :, indices[1]].max(dim=1)
    true_classes = y[indices[0], indices[1]]
    precision = precision_score(true_classes, predicted_classes, average='macro')
    recall = recall_score(true_classes, predicted_classes, average='macro')
    f1 = f1_score(true_classes, predicted_classes, average='macro')
    return precision, recall, f1

def preprocess(record_list, classes, tokenizer, batch_size, max_length, test=False):
    token_tensors_all_list = [list2token(tokenizer, d['words'], max_length) for d in record_list]
    data = torch.cat(token_tensors_all_list, dim=0)
    if test:
        batch_size = data.shape[0]
    data_batches = to_batches(data, batch_size)
    target_tensors_all_list = [cat2digit(classes, d['ner'], max_length)[0] for d in record_list]
    target = torch.stack(target_tensors_all_list, dim=0)
    target_batches = to_batches(target, batch_size)
    att_mask_all_list = [cat2digit(classes, d['ner'], max_length)[1] for d in record_list]
    att_mask = torch.stack(att_mask_all_list, dim=0)
    att_mask_batches = to_batches(att_mask, batch_size)
    if test:
        return data_batches[0], target_batches[0], att_mask_batches[0], record_list
    else:
        c = list(zip(data_batches, target_batches, att_mask_batches))
        random.shuffle(c)
        data_batches, target_batches, att_mask_batches = zip(*c)
        return data_batches, target_batches, att_mask_batches

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def inference(text, model, tokenizer, max_length):
    text_list = split_para(text)
    token_list = tokenizer.convert_tokens_to_ids(text_list)
    data_test = torch.tensor(token_list + [0] * (max_length - len(token_list)))[None, :]
    att_mask_test = torch.tensor([1] * len(text_list) + [0] * (max_length - len(text_list)))[None, :]
    y_pred_test = model(data_test, attention_mask=att_mask_test)
    y_pred_test = torch.swapaxes(y_pred_test, 1, 2)
    pred = y_pred_test.max(dim=1)[1][0][:len(text_list)]
    return pred, text_list

def inference_batch(texts, model, tokenizer, max_length):
    results = []
    for text in texts:
        pred, text_list = inference(text, model, tokenizer, max_length)
        results.append((pred, text_list))
    return results

def show_pred(pred, text_list, n_word_line=15):
    colors = ['black', 'red', 'blue', 'green', 'cyan', 'darkorange']
    word_list_marked = ['<' + w + '>' for w in text_list]
    markers = [{"color": colors[i]} for i in pred]
    j = 0
    for i in range(len(text_list)):
        if (i + 1) % n_word_line == 0:
            word_list_marked.insert(i + j, '\n')
            j += 1
    fig, ax = plt.subplots()
    ax.set_axis_off()
    HighlightText(x=0., y=1, s='<O>, <MATERIAL>, <MLIP>, <PROPERTY>, <VALUE>, <APPLICATION>',
                  highlight_textprops=[{"color": c} for c in colors], ax=ax)
    HighlightText(x=0., y=0.9, s=' '.join(word_list_marked),
                  highlight_textprops=markers, ax=ax)
    plt.show()

def result2data(pred, text_list, classes_inv):
    body = " ".join(text_list)
    annotation = {}
    annotation_list = [(index, classes_inv[i.item()]) for index, i in enumerate(pred) if i.item() > 0]
    for i in annotation_list:
        annotation[str(i[0])] = i[1]
    annotation = str(annotation).replace("'", '"')
    return body, annotation

def result2data_batch(results, classes_inv):
    bodies = []
    annotations = []
    for result in results:
        body, annotation = result2data(*result, classes_inv)
        bodies.append(body)
        annotations.append(annotation)
        data = [bodies, annotations]
    return list(zip(*data))

def insert_data(data, db_path):
    con = sqlite3.connect(db_path)
    cur = con.cursor()
    for d in data:
        cur.execute("INSERT INTO mlip (body, annotation) VALUES (?, ?)", d)
    con.commit()
    con.close()