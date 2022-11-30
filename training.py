from __future__ import print_function
import time
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from tqdm.auto import tqdm

from torch.utils.data import DataLoader
import math
from sentence_transformers import SentenceTransformer,  LoggingHandler, losses, models, util
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers.readers import InputExample
import logging
from datetime import datetime
import sys
import os
import gzip
import csv

import torch
import math
import torch
import random
import time
import numpy as np
import collections
import pandas as pd
from tqdm.auto import tqdm

import datasets
from datasets import load_dataset
import matplotlib.pyplot as plt
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import Trainer, TrainingArguments
from transformers import DataCollatorForLanguageModeling
from transformers import AutoModelForCausalLM, AutoModelForMaskedLM
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import AutoModelForSequenceClassification, AdamW
from transformers import get_linear_schedule_with_warmup, get_cosine_with_hard_restarts_schedule_with_warmup
from datasets import Features, ClassLabel, Value

def classification_model (dataset, model_checkpoint = 'roberta-base'):

  classes = dataset['train'].features['label'].names
  id2label = {i: element for i, element in enumerate (classes)}
  label2id = {element: i for i, element in enumerate (classes)}

  model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, 
                                                             num_labels = len (classes), 
                                                             id2label = id2label, 
                                                             label2id = label2id)
  return model

# Function to calculate the accuracy of our predictions vs labels
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat), pred_flat, labels_flat

def train (model, optimizer, scheduler, data_loader, epoch, eval = False):

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    stage = 'Eval' if eval else 'Train'
    total_loss = 0
    total_accuracy = 0
    total = 0
    true_labels = []
    predictions = []

    if eval == True:
      model.eval ()
    else:
      model.train()

    pbar = tqdm(data_loader)
    for batch in pbar:

        b_input_ids = batch['input_ids'].to(device)
        b_input_mask = batch['attention_mask'].to(device)
        b_labels = batch['label'].to(device)

        if eval:
          with torch.no_grad():        
            result = model(b_input_ids, 
                       token_type_ids=None, 
                       attention_mask=b_input_mask, 
                       labels=b_labels,
                       return_dict=True)
        else:
          model.zero_grad()        
          result = model(b_input_ids, 
                       token_type_ids=None, 
                       attention_mask=b_input_mask, 
                       labels=b_labels,
                       return_dict=True)
          
        loss = result.loss
        logits = result.logits
        total_loss += loss.item()

        if eval == False:
          loss.backward()
          torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
          optimizer.step()

          scheduler.step()
            
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
        (acc, pred, lbls) = flat_accuracy(logits, label_ids)
        total_accuracy += acc
        predictions += list(pred)
        true_labels += list(lbls)
        total += len (label_ids)
        pbar.set_description(f"[{stage}] Epoch: {epoch:3}, Total: {total:6}, Accuracy : {(total_accuracy / total):.4f}, Loss: {loss:.4f}")
    pbar.close ()
    return [total, total_accuracy, total_loss, predictions, true_labels]

def model_training (model, epochs, tokenized_datasets, batch_size):
  
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to (device)

    train_loader = DataLoader (tokenized_datasets['train'],
                            sampler = RandomSampler(tokenized_datasets['train']),
                            batch_size = batch_size)
    val_loader = DataLoader (tokenized_datasets['validation'],
                            sampler = SequentialSampler(tokenized_datasets['validation']),
                            batch_size = batch_size)
    test_loader = DataLoader (tokenized_datasets['test'],
                            sampler = SequentialSampler(tokenized_datasets['test']),
                            batch_size = batch_size)

    optimizer = torch.optim.AdamW(model.parameters(),
                    lr = 2e-5, # args.learning_rate - default is 5e-5, our notebook had 2e-5
                    eps = 1e-8 # args.adam_epsilon  - default is 1e-8.
                    )
    total_steps = len(train_loader) * epochs

    # Create the learning rate scheduler.
    #scheduler = get_linear_schedule_with_warmup(optimizer, 
    #                                          num_warmup_steps = 0, # Default value in run_glue.py
    #                                          num_training_steps = total_steps)
    scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(optimizer, 
                                                num_warmup_steps = 50, # Default value in run_glue.py
                                                num_training_steps = total_steps,
                                                num_cycles = 2)
    
    seed_val = 42
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

    training_stats = {'epoch': [], 'train_loss': [], 'eval_loss': [], 'train_acc': [], 'eval_acc': []}

    for epoch_i in tqdm(range(0, epochs)):

        [total_train, total_train_accuracy, total_train_loss, predictions, true_labels] = train (model, optimizer, scheduler, train_loader, epoch_i, eval = False)
        [total_eval, total_eval_accuracy, total_eval_loss, predictions, true_labels] = train (model, optimizer, scheduler, val_loader, epoch_i, eval = True)

        # Record all statistics from this epoch.
        training_stats['epoch'].append (epoch_i + 1)
        training_stats['train_loss'].append (total_train_loss / len (train_loader))
        training_stats['eval_loss'].append (total_eval_loss / len (val_loader))
        training_stats['train_acc'].append (total_train_accuracy / total_train)
        training_stats['eval_acc'].append (total_eval_accuracy / total_eval)

    test_metrics = {}    
    [test_metrics['total'], test_metrics['accuracy'], test_metrics['loss'], test_metrics['predictions'], test_metrics['true_labels']] = train (model, optimizer, scheduler, test_loader, 0, eval = True)

    fig, axs = plt.subplots(1,2, figsize=(15,5))
    fig.tight_layout()
    # Graph for Loss
    axs[0].plot(training_stats['epoch'], training_stats['train_loss'], 'g', label='Training loss')
    axs[0].plot(training_stats['epoch'], training_stats['eval_loss'], 'b', label='Validation loss')
    axs[0].set_title ('Loss')
    axs[0].set_xlabel('Epochs')
    axs[0].set_ylabel('Loss')
    axs[0].legend ()

    # Graph for Accuracy
    axs[1].plot(training_stats['epoch'], training_stats['train_acc'], 'g', label='Training accuracy')
    axs[1].plot(training_stats['epoch'], training_stats['eval_acc'], 'b', label='Validation accuracy')
    axs[1].set_title ('Accuracy')
    axs[1].set_xlabel('Epochs')
    axs[1].set_ylabel('Accuracy')
    axs[1].legend ()

    plt.show ()
    return (model, test_metrics)