#!/usr/bin/env python
# coding: utf-8

import os

if os.environ.get("KAGGLE_KERNEL_RUN_TYPE") is None:
    ON_KAGGLE = False
    PLATFORM = "Colab"
    PLATFORM = "Local"
else:
    ON_KAGGLE = True
    
if not ON_KAGGLE:
    import shutil
    from requests import get

    if PLATFORM is "Colab":
        from google.colab import drive, files
        # mount Google Drive
        drive.mount("/content/drive")

        os.makedirs("/root/.kaggle", exist_ok=True)  # API token 用のディレクトリを作成
        token_path = "/content/drive/MyDrive/competition/kaggle.json"  # token の path
        shutil.copy(token_path, "/root/.kaggle")  # tokenを"/root/.signate"にコピー

        DRIVE = "/content/drive/MyDrive/competition/Feedback_Prize"  # Drive のパス
    else:
        os.makedirs("/root/.kaggle", exist_ok=True)  # API token 用のディレクトリを作成
        token_path = "/root/sumi/_kaggle/kaggle.json"  # token の path
        shutil.copy(token_path, "/root/.kaggle")  # tokenを"/root/.signate"にコピー

        DRIVE = "/root/sumi/_kaggle/feedback-prize-2021"  # Drive のパス
        
else:
    for dirname, _, filenames in os.walk('/kaggle/input'):
        for filename in filenames:
            print(os.path.join(dirname, filename))


INPUT = os.path.join(DRIVE, "input")
OUTPUT = os.path.join(DRIVE, "output")
SUBMISSION = os.path.join(DRIVE, "submission")

if not os.path.isdir(INPUT):  # 初回時にのみデータをINPUTフォルダにロード
  os.makedirs(INPUT, exist_ok=True)
  get_ipython().system(' kaggle competitions download -c feedback-prize-2021 --path=$INPUT')

# 実験ごとのフォルダを作成
dirs = [INPUT, OUTPUT, SUBMISSION]
for d in dirs:
    os.makedirs(d, exist_ok=True)

    
import argparse

parser = argparse.ArgumentParser(description='このプログラムの説明（なくてもよい）') 
parser.add_argument('config_yaml', help='input config')
args = parser.parse_args()


# Config
# 
# model setting
# - prepare dataset
# - set model_savename and model_name you want to train

from omegaconf import OmegaConf

# config_yaml = DRIVE + "/notebooks/config/" + args.config_yaml
Config = OmegaConf.load(args.config_yaml)

Config.data_dir = os.path.join(INPUT, "feedback-prize-2021")
Config.pre_data_dir = os.path.join(INPUT, 'preprocessed')
Config.model_dir = os.path.join(DRIVE, f'model/{Config.name}')
Config.output_dir = os.path.join(DRIVE, f'output/{Config.name}')


# constants

IGNORE_INDEX = -100
NON_LABEL = -1
OUTPUT_LABELS = ['O', 'B-Lead', 'I-Lead', 'B-Position', 'I-Position', 'B-Claim', 'I-Claim', 'B-Counterclaim', 'I-Counterclaim', 
                 'B-Rebuttal', 'I-Rebuttal', 'B-Evidence', 'I-Evidence', 'B-Concluding Statement', 'I-Concluding Statement']
LABELS_TO_IDS = {v:k for k,v in enumerate(OUTPUT_LABELS)}
IDS_TO_LABELS = {k:v for k,v in enumerate(OUTPUT_LABELS)}

MIN_THRESH = {
    "I-Lead": 9,
    "I-Position": 5,
    "I-Evidence": 14,
    "I-Claim": 3,
    "I-Concluding Statement": 11,
    "I-Counterclaim": 6,
    "I-Rebuttal": 4,
}

PROB_THRESH = {
    "I-Lead": 0.7,
    "I-Position": 0.55,
    "I-Evidence": 0.65,
    "I-Claim": 0.55,
    "I-Concluding Statement": 0.7,
    "I-Counterclaim": 0.5,
    "I-Rebuttal": 0.55,
}


if not ON_KAGGLE:
    if not os.path.exists(Config.model_dir):
        os.makedirs(Config.model_dir, exist_ok=True)
    if not os.path.exists(Config.output_dir):
        os.makedirs(Config.output_dir, exist_ok=True)


# general
import pandas as pd
import numpy as np
import random
from tqdm.notebook import tqdm
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
import gc
from collections import defaultdict
# nlp
from sklearn.feature_extraction.text import CountVectorizer
import torch
import torch.nn as nn
from transformers import LongformerConfig, LongformerModel, LongformerTokenizerFast, AutoConfig, AutoModel, AutoTokenizer
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler


# set seed & split train/test

def seed_everything(seed=Config.random_seed):
    #os.environ['PYTHONSEED'] = str(seed)
    np.random.seed(seed%(2**32-1))
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic =True
    torch.backends.cudnn.benchmark = False

seed_everything()
# device optimization
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

print(f'Using device: {device}')


# data load

df_alltrain = pd.read_csv( os.path.join(INPUT, 'corrected-train-csv-feedback-prize/corrected_train.csv') )

import ast
alltrain_texts = pd.read_csv(f"{Config.pre_data_dir}/fold-alltrain_texts.csv")
alltrain_texts['text_split'] = [ast.literal_eval(d) for d in alltrain_texts['text_split']]
alltrain_texts['entities'] = [ast.literal_eval(d) for d in alltrain_texts['entities']]

if Config.is_debug:
    alltrain_texts = alltrain_texts.sample(Config.debug_sample).reset_index(drop=True)
print(len(alltrain_texts))


# ## dataset

class FeedbackPrizeDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len, has_labels):
        self.len = len(dataframe)
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.has_labels = has_labels
    
    def __getitem__(self, index):
        text = self.data.text[index]
        encoding = self.tokenizer(
            text.split(),
            is_split_into_words = True,
            padding = 'max_length',
            truncation = True,
            max_length = self.max_len
        )
        word_ids = encoding.word_ids()

        # targets
        if self.has_labels:
            word_labels = self.data.entities[index]
            prev_word_idx = None
            labels_ids = []
            for word_idx in word_ids:
                if word_idx is None:
                    labels_ids.append(IGNORE_INDEX)
                elif word_idx != prev_word_idx:
                    labels_ids.append(LABELS_TO_IDS[word_labels[word_idx]])
                else:
                    if Config.label_subtokens:
                        labels_ids.append(LABELS_TO_IDS[word_labels[word_idx]])
                    else:
                        labels_ids.append(IGNORE_INDEX)
                prev_word_idx = word_idx
            encoding['labels'] = labels_ids
        # convert to torch.tensor
        item = {k: torch.as_tensor(v) for k, v in encoding.items()}
        word_ids2 = [w if w is not None else NON_LABEL for w in word_ids]
        item['word_ids'] = torch.as_tensor(word_ids2)
        return item

    def __len__(self):
        return self.len


# ## model

class FeedbackModel(nn.Module):
    def __init__(self):
        super(FeedbackModel, self).__init__()
        if Config.model_savename == 'longformer':
            model_config = LongformerConfig.from_pretrained(Config.model_name)
            self.backbone = LongformerModel.from_pretrained(Config.model_name, config=model_config)
        else:
            model_config = AutoConfig.from_pretrained(Config.model_name)
            self.backbone = AutoModel.from_pretrained(Config.model_name, config=model_config)
        self.model_config = model_config
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.2)
        self.dropout3 = nn.Dropout(0.3)
        self.dropout4 = nn.Dropout(0.4)
        self.dropout5 = nn.Dropout(0.5)
        self.head = nn.Linear(model_config.hidden_size, Config.num_labels)
    
    def forward(self, input_ids, mask):
        x = self.backbone(input_ids, mask)
        logits1 = self.head(self.dropout1(x[0]))
        logits2 = self.head(self.dropout2(x[0]))
        logits3 = self.head(self.dropout3(x[0]))
        logits4 = self.head(self.dropout4(x[0]))
        logits5 = self.head(self.dropout5(x[0]))
        logits = (logits1 + logits2 + logits3 + logits4 + logits5) / 5
        return logits

    
def build_model_tokenizer():
    if Config.model_savename == 'longformer':
        tokenizer = LongformerTokenizerFast.from_pretrained(Config.model_name, add_prefix_space = True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(Config.model_name, add_prefix_space = True)
    model = FeedbackModel()
    return model, tokenizer


# ## utility function

def active_logits(raw_logits, word_ids):
    word_ids = word_ids.view(-1)
    active_mask = word_ids.unsqueeze(1).expand(word_ids.shape[0], Config.num_labels)
    active_mask = active_mask != NON_LABEL
    active_logits = raw_logits.view(-1, Config.num_labels)
    active_logits = torch.masked_select(active_logits, active_mask) # return 1dTensor
    active_logits = active_logits.view(-1, Config.num_labels) 
    return active_logits

def active_labels(labels):
    active_mask = labels.view(-1) != IGNORE_INDEX
    active_labels = torch.masked_select(labels.view(-1), active_mask)
    return active_labels

def active_preds_prob(active_logits):
    active_preds = torch.argmax(active_logits, axis = 1)
    active_preds_prob, _ = torch.max(active_logits, axis = 1)
    return active_preds, active_preds_prob


# ## evaluating function

def calc_overlap(row):
    """
    calculate the overlap between prediction and ground truth
    """
    set_pred = set(row.new_predictionstring_pred.split(' '))
    set_gt = set(row.new_predictionstring_gt.split(' '))
    # length of each end intersection
    len_pred = len(set_pred)
    len_gt = len(set_gt)
    intersection = len(set_gt.intersection(set_pred))
    overlap_1 = intersection / len_gt
    overlap_2 = intersection / len_pred
    return [overlap_1, overlap_2]

def score_feedback_comp(pred_df, gt_df):
    """
    A function that scores for the kaggle
        Student Writing Competition
        
    Uses the steps in the evaluation page here:
        https://www.kaggle.com/c/feedback-prize-2021/overview/evaluation
    """
    gt_df = gt_df[['id', 'discourse_type', 'new_predictionstring']].reset_index(drop = True).copy()
    pred_df = pred_df[['id', 'class', 'new_predictionstring']].reset_index(drop = True).copy()
    gt_df['gt_id'] = gt_df.index
    pred_df['pred_id'] = pred_df.index
    joined = pred_df.merge(
        gt_df,
        left_on = ['id', 'class'],
        right_on = ['id', 'discourse_type'],
        how = 'outer',
        suffixes = ['_pred', '_gt']
    )
    joined['new_predictionstring_gt'] =  joined['new_predictionstring_gt'].fillna(' ')
    joined['new_predictionstring_pred'] =  joined['new_predictionstring_pred'].fillna(' ')
    joined['overlaps'] = joined.apply(calc_overlap, axis = 1)
    # overlap over 0.5: true positive
    # If nultiple overlaps exists, the higher is taken.
    joined['overlap1'] = joined['overlaps'].apply(lambda x: eval(str(x))[0])
    joined['overlap2'] = joined['overlaps'].apply(lambda x: eval(str(x))[1])

    joined['potential_TP'] = (joined['overlap1'] >= 0.5) & (joined['overlap2'] >= 0.5)
    joined['max_overlap'] = joined[['overlap1', 'overlap2']].max(axis = 1)
    tp_pred_ids = joined.query('potential_TP').sort_values('max_overlap', ascending = False)                  .groupby(['id', 'new_predictionstring_gt']).first()['pred_id'].values
    
    fp_pred_ids = [p for p in joined['pred_id'].unique() if p not in tp_pred_ids]
    matched_gt_ids = joined.query('potential_TP')['gt_id'].unique()
    unmatched_gt_ids = [c for c in joined['gt_id'].unique() if c not in matched_gt_ids]

    TP = len(tp_pred_ids)
    FP = len(fp_pred_ids)
    FN = len(unmatched_gt_ids)
    macro_f1_score = TP / (TP + 1/2 * (FP + FN))
    return macro_f1_score

def oof_score(df_val, oof):
    f1score = []
    classes = ['Lead', 'Position','Claim', 'Counterclaim', 'Rebuttal','Evidence','Concluding Statement']
    for c in classes:
        pred_df = oof.loc[oof['class'] == c].copy()
        gt_df = df_val.loc[df_val['discourse_type'] == c].copy()
        f1 = score_feedback_comp(pred_df, gt_df)
        print(f'{c:<10}: {f1:4f}')
        f1score.append(f1)
    f1avg = np.mean(f1score)
    return f1avg


# ## inferencing function

def inference(model, dl, criterion, valid_flg):
    final_predictions = []
    final_predictions_prob = []
    stream = tqdm(dl)
    model.eval()
    
    valid_loss = 0
    valid_accuracy = 0
    all_logits = None
    for batch_idx, batch in enumerate(stream, start = 1):
        ids = batch['input_ids'].to(device, dtype = torch.long)
        mask = batch['attention_mask'].to(device, dtype = torch.long)
        with torch.no_grad():
            raw_logits = model(input_ids=ids, mask = mask)
        del ids, mask
        
        word_ids = batch['word_ids'].to(device, dtype = torch.long)
        logits = active_logits(raw_logits, word_ids)
        sf_logits = torch.softmax(logits, dim= -1)
        sf_raw_logits = torch.softmax(raw_logits, dim=-1)
        if valid_flg:    
            raw_labels = batch['labels'].to(device, dtype = torch.long)
            labels = active_labels(raw_labels)
            preds, preds_prob = active_preds_prob(sf_logits)
            valid_accuracy += accuracy_score(labels.cpu().numpy(), preds.cpu().numpy())
            loss = criterion(logits, labels)
            valid_loss += loss.item()
        
        if batch_idx == 1:
            all_logits = sf_raw_logits.cpu().numpy()
        else:
            all_logits = np.append(all_logits, sf_raw_logits.cpu().numpy(), axis=0)

    
    if valid_flg:        
        epoch_loss = valid_loss / batch_idx
        epoch_accuracy = valid_accuracy / batch_idx
    else:
        epoch_loss, epoch_accuracy = 0, 0
    return all_logits, epoch_loss, epoch_accuracy


def preds_class_prob(all_logits, dl):
    print("predict target class and its probabilty")
    final_predictions = []
    final_predictions_score = []
    stream = tqdm(dl)
    len_sample = all_logits.shape[0]

    for batch_idx, batch in enumerate(stream, start=0):
        for minibatch_idx in range(Config.valid_batch_size):
            sample_idx = int(batch_idx * Config.valid_batch_size + minibatch_idx)
            if sample_idx > len_sample - 1 : break
            word_ids = batch['word_ids'][minibatch_idx].numpy()
            predictions =[]
            predictions_prob = []
            pred_class_id = np.argmax(all_logits[sample_idx], axis=1)
            pred_score = np.max(all_logits[sample_idx], axis=1)
            pred_class_labels = [IDS_TO_LABELS[i] for i in pred_class_id]
            prev_word_idx = -1
            for idx, word_idx in enumerate(word_ids):
                if word_idx == -1:
                    pass
                elif word_idx != prev_word_idx:
                    predictions.append(pred_class_labels[idx])
                    predictions_prob.append(pred_score[idx])
                    prev_word_idx = word_idx
            final_predictions.append(predictions)
            final_predictions_score.append(predictions_prob)
    return final_predictions, final_predictions_score


def get_preds_onefold(model, df, dl, criterion, valid_flg):
    logits, valid_loss, valid_acc = inference(model, dl, criterion, valid_flg)
    all_preds, all_preds_prob = preds_class_prob(logits, dl)
    df_pred = post_process_pred(df, all_preds, all_preds_prob)
    return df_pred, valid_loss, valid_acc

def get_preds_folds(model, df, dl, criterion, valid_flg=False):
    for i_fold in range(Config.n_fold):
        model_filename = os.path.join(Config.model_dir, f"{Config.model_savename}_{i_fold}.bin")
        print(f"{model_filename} inference")
        model = model.to(device)
        model.load_state_dict(torch.load(model_filename))
        logits, valid_loss, valid_acc = inference(model, dl, criterion, valid_flg)
        if i_fold == 0:
            avg_pred_logits = logits
        else:
            avg_pred_logits += logits
    avg_pred_logits /= Config.n_fold
    all_preds, all_preds_prob = preds_class_prob(avg_pred_logits, dl)
    df_pred = post_process_pred(df, all_preds, all_preds_prob)
    return df_pred

def post_process_pred(df, all_preds, all_preds_prob):
    final_preds = []
    for i in range(len(df)):
        idx = df.id.values[i]
        pred = all_preds[i]
        pred_prob = all_preds_prob[i]
        j = 0
        while j < len(pred):
            cls = pred[j]
            if cls == 'O': j += 1
            else: cls = cls.replace('B', 'I')
            end = j + 1
            while end < len(pred) and pred[end] == cls:
                end += 1
            if cls != 'O' and cls !='':
                avg_score = np.mean(pred_prob[j:end])
                if end - j > MIN_THRESH[cls] and avg_score > PROB_THRESH[cls]:
                    final_preds.append((idx, cls.replace('I-', ''), ' '.join(map(str, list(range(j, end))))))
            j = end
    df_pred = pd.DataFrame(final_preds)
    df_pred.columns = ['id', 'class', 'new_predictionstring']
    return df_pred


# ## training and validating function

def train_fn(model, dl_train, optimizer, epoch, criterion):
    model.train()
    train_loss = 0
    train_accuracy = 0
    stream = tqdm(dl_train)
    scaler = GradScaler()

    for batch_idx, batch in enumerate(stream, start = 1):
        ids = batch['input_ids'].to(device, dtype = torch.long)
        mask = batch['attention_mask'].to(device, dtype = torch.long)
        raw_labels = batch['labels'].to(device, dtype = torch.long)
        word_ids = batch['word_ids'].to(device, dtype = torch.long)
        optimizer.zero_grad()
        with autocast():
            raw_logits = model(input_ids = ids, mask = mask)
        
        logits = active_logits(raw_logits, word_ids)
        labels = active_labels(raw_labels)
        sf_logits = torch.softmax(logits, dim=-1)
        preds, preds_prob = active_preds_prob(sf_logits)
        train_accuracy += accuracy_score(labels.cpu().numpy(), preds.cpu().numpy())
        criterion = nn.CrossEntropyLoss()
        loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        train_loss += loss.item()
        
        if batch_idx % Config.verbose_steps == 0:
            loss_step = train_loss / batch_idx
            print(f'Training loss after {batch_idx:04d} training steps: {loss_step}')
            
    epoch_loss = train_loss / batch_idx
    epoch_accuracy = train_accuracy / batch_idx
    del dl_train, raw_logits, logits, raw_labels, preds, labels
    torch.cuda.empty_cache()
    gc.collect()
    print(f'epoch {epoch} - training loss: {epoch_loss:.4f}')
    print(f'epoch {epoch} - training accuracy: {epoch_accuracy:.4f}')


def valid_fn(model, df_val, df_val_eval, dl_val, epoch, criterion):
    oof, valid_loss, valid_acc  = get_preds_onefold(model, df_val, dl_val, criterion, valid_flg=True)
    f1score =[]
    # classes = oof['class'].unique()
    classes = ['Lead', 'Position', 'Claim','Counterclaim', 'Rebuttal','Evidence','Concluding Statement']
    print(f"Validation F1 scores")

    for c in classes:
        pred_df = oof.loc[oof['class'] == c].copy()
        gt_df = df_val_eval.loc[df_val_eval['discourse_type'] == c].copy()
        f1 = score_feedback_comp(pred_df, gt_df)
        print(f' * {c:<10}: {f1:4f}')
        f1score.append(f1)
    f1avg = np.mean(f1score)
    print(f'Overall Validation avg F1: {f1avg:.4f} val_loss:{valid_loss:.4f} val_accuracy:{valid_acc:.4f}')
    return valid_loss, oof


# ## training loop

oof = pd.DataFrame()
for i_fold in range(Config.n_fold):
    print(f'=== fold{i_fold} training ===')
    model, tokenizer = build_model_tokenizer()
    model = model.to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=Config.lr)
    
    df_train = alltrain_texts[alltrain_texts['fold'] != i_fold].reset_index(drop = True)
    ds_train = FeedbackPrizeDataset(df_train, tokenizer, Config.max_length, True)
    df_val = alltrain_texts[alltrain_texts['fold'] == i_fold].reset_index(drop = True)
    val_idlist = df_val['id'].unique().tolist()
    df_val_eval = df_alltrain.query('id==@val_idlist').reset_index(drop=True)
    ds_val = FeedbackPrizeDataset(df_val, tokenizer, Config.max_length, True)
    dl_train = DataLoader(ds_train, batch_size=Config.train_batch_size, shuffle=True, num_workers=2, pin_memory=True)
    dl_val = DataLoader(ds_val, batch_size=Config.valid_batch_size, shuffle=False, num_workers=2, pin_memory=True)

    best_val_loss = np.inf
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, Config.n_epoch + 1):
        train_fn(model, dl_train, optimizer, epoch, criterion)
        valid_loss, _oof = valid_fn(model, df_val, df_val_eval, dl_val, epoch, criterion)
        if valid_loss < best_val_loss:
            best_val_loss = valid_loss
            _oof_fold_best = _oof
            _oof_fold_best['fold'] = i_fold
            model_filename = f'{Config.model_dir}/{Config.model_savename}_{i_fold}.bin'
            torch.save(model.state_dict(), model_filename)
            print(f'{model_filename} saved')

    oof = pd.concat([oof, _oof_fold_best])

print( oof.head() )

oof.to_csv(f'{Config.output_dir}/oof_{Config.name}.csv', index=False)


# ## cv score

if Config.is_debug:
    idlist = alltrain_texts['id'].unique().tolist()
    df_train = df_alltrain.query('id==@idlist')
else:
    df_train = df_alltrain.copy()
print(f'overall cv score: {oof_score(df_train, oof)}')


#Slackに完了通知
import requests, json
# WEB_HOOK_URL = "用意したWEB_HOOK_URL"
WEB_HOOK_URL = "https://hooks.slack.com/services/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"
requests.post(WEB_HOOK_URL, data = json.dumps({
    'text': f'実行完了: {Config.name} \n overall cv score: {oof_score(df_train, oof)}', 
    'username': u'[colab]:norikatsu.sumi.2019@gmail.com',  
    'icon_emoji': u':smile_cat:',  
    'link_names': 1,  
}))




