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

from load_dataset import * 
from training import * 
import json
import torch, gc

def main():
    model_save_path = "/users/PAS2348/ramirez537/snli/model_saved"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = load_and_filter_dataset ()
    # visualize_dataset(dataset)
    (tokenizer, tokenized_dataset) = tokenize_dataset (dataset, model_checkpoint = model_save_path)
    model = classification_model (dataset, model_checkpoint = model_save_path)
    model.to(device)
   
    gc.collect()
    torch.cuda.empty_cache()
    (model, test_metrics) = model_training (model, 1, tokenized_dataset, 128)

    with open("test_metrics", "w") as outfile:
        json.dump(test_metrics, outfile)

    model.eval ()
    model.save_pretrained (model_save_path)

if __name__ == '__main__':
    main()