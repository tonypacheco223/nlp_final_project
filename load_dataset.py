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

def visualize_dataset (dataset: datasets.dataset_dict.DatasetDict) -> None:

  labels = dataset['train'].features['label'].names

  try:
    # Graph for split wise graph
    split_wise_labels = [key for key, value in dataset.items ()]
    split_wise_numbers = [value.num_rows for key, value in dataset.items ()]
    split_wise_explode = [random.randint (1, 5) / 10 if key != 'train' else 0 for key, value in dataset.items ()]

    sen_length = {}

    # Graph for total label wise split
    split_wise_label_wise = {}
    total_label_wise_split = [0] * len (labels)
    for key, value in dataset.items ():
      counter = collections.Counter (value['label'])
      temp = [0] * len (labels)
      for index, total in counter.items ():
        temp[ index ] += total
      split_wise_label_wise[key] = temp  
      total_label_wise_split = [x + y for x, y in zip(total_label_wise_split, temp)]

      sen_length[key] = {'premise': [], 'hypothesis': [], 'sum': []}
      
      for element in value:
        sen_length[key]['premise'].append ( len(element['premise'].split ()) )
        sen_length[key]['hypothesis'].append ( len(element['hypothesis'].split ()) )
        sen_length[key]['sum'].append (sen_length[key]['premise'][-1] + sen_length[key]['hypothesis'][-1])
      
    # Graph for split wise and label wise
    split_wise_label_wise = [split_wise_label_wise[label] for label in split_wise_labels]
    split_wise_label_wise = [[split_wise_label_wise[j][i] for j in range(len(split_wise_label_wise))] for i in range(len(split_wise_label_wise[0]))]

    fig, axs = plt.subplots(2,3, figsize=(20,10))
    fig.tight_layout()
    # Graph 1
    axs[0, 0].pie(x = split_wise_numbers, explode=split_wise_explode, labels=split_wise_labels, autopct='%1.1f%%', startangle=90)
    axs[0, 0].axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    axs[0, 0].set_title ('Train, Test, Validation Split')

    # Graph 2
    axs[0, 1].pie(x = total_label_wise_split, labels=labels, autopct='%1.1f%%', startangle=90)
    axs[0, 1].axis('equal')
    axs[0, 1].set_title ('Label wise split for entire Dataset')

    # Graph 3
    X = np.arange(3)
    axs[0, 2].bar(X + 0.00, split_wise_label_wise[0], color = 'b', width = 0.25)
    axs[0, 2].bar(X + 0.25, split_wise_label_wise[1], color = 'g', width = 0.25)
    axs[0, 2].bar(X + 0.50, split_wise_label_wise[2], color = 'r', width = 0.25)
    axs[0, 2].legend(labels=labels)
    axs[0, 2].set_xticklabels(["", split_wise_labels[0], "", split_wise_labels[1], "", split_wise_labels[2]]) 
    axs[0, 2].set_title ('Split Wise, Label Wise Data')

    axs[1, 0].hist(sen_length['train']['premise'], bins=50, label="premise length in train set", alpha=0.5)
    axs[1, 0].hist(sen_length['validation']['premise'], bins=50, label="premise length in validation set", alpha=0.5)
    axs[1, 0].legend(loc='best')
    axs[1, 0].set_title ('Premise Length Comparison')
    axs[1, 0].set_xlabel('Sentence Length')
    axs[1, 0].set_ylabel('Frequency')
    
    axs[1, 1].hist(sen_length['train']['hypothesis'], bins=50, label="hypothesis length in train set", alpha=0.5)
    axs[1, 1].hist(sen_length['validation']['hypothesis'], bins=50, label="hypothesis length in validation set", alpha=0.5)
    axs[1, 1].legend(loc='best')
    axs[1, 1].set_title ('Hypothesis Length Comparison')
    axs[1, 1].set_xlabel('Sentence Length')
    axs[1, 1].set_ylabel('Frequency')

    axs[1, 2].hist(sen_length['train']['sum'], bins=50, label="Input length in train set", alpha=0.5)
    axs[1, 2].hist(sen_length['validation']['sum'], bins=50, label="Input length in validation set", alpha=0.5)
    axs[1, 2].legend(loc='best')
    axs[1, 2].set_title ('Input Length Comparison')
    axs[1, 2].set_xlabel('Sentence Length')
    axs[1, 2].set_ylabel('Frequency')

    plt.show()

  except Exception as e:
    print ("Cannot plot the graphs: ", str (e))

def load_and_filter_dataset ():
  dataset = load_dataset("esnli")
  # dataset = dataset.filter(lambda example: example['label'] != '-')
  for sample in tqdm(dataset['train']) :
    sample['label'] = float(sample['label'])
  for sample in tqdm(dataset['validation']):
    sample['label'] = float(sample['label'])
  for sample in tqdm(dataset['test']):
    sample['label'] = float(sample['label'])
  return dataset

def tokenize_dataset (dataset, model_checkpoint = 'roberta-base'):
  tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
  
  def tokenize_function (examples):
    return tokenizer (examples['premise'], 
                      examples['hypothesis'],
                      add_special_tokens = True,
                      max_length=64, 
                      padding = 'max_length', 
                      truncation = True)
    
  tokenized_datasets = dataset.map (tokenize_function,
                                  batched = True,
                                  remove_columns = ['premise', 'hypothesis']
                                  ).with_format("torch")
  return (tokenizer, tokenized_datasets)

