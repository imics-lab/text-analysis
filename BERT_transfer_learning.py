# -*- coding: utf-8 -*-
"""Creates and trains a BERT transfer learning model.
    
@author: Morgan Byers
"""

#imports
from configparser import ConfigParser
import torch
from tqdm.notebook import tqdm
from transformers import BertTokenizer
from torch.utils.data import TensorDataset
from transformers import BertForSequenceClassification
import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

#read data using config parser
config_object = ConfigParser()
config_object.read("config.ini")

paths = config_object["appSettings"]
data_path = paths["datapath"]
target_path = paths["targetpath"]

#create data frame
sentences = pd.read_csv(data_path)
sentences.head()

#get list of clean sentences
clean_sentences = list(sentences['sentences_clean'].values)

#get target 
ratings = pd.read_csv(target_path)
ratings.head()

#get rounded target classes
float_target = list(ratings.mean(axis=1))
target = [int(round(num)) for num in float_target]

#delete sentences missing labels
del clean_sentences[1044]
del clean_sentences[314]

#train test split
xtrain, xtest, ytrain, ytest = train_test_split(clean_sentences, target, 
                                                test_size=0.2, stratify=target)

#load pretrained bert tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', 
                                          do_lower_case=True)

train_encoded_data = tokenizer.batch_encode_plus(xtrain, 
                                                 add_special_tokens=True, 
                                                 return_attention_mask = True,
                                                 pad_to_max_length=True,
                                                 max_length=256,
                                                 return_tensors='pt')

test_encoded_data = tokenizer.batch_encode_plus(xtest, add_special_tokens=True, 
                                                 return_attention_mask = True,
                                                 pad_to_max_length=True,
                                                 max_length=256,
                                                 return_tensors='pt')

#get input id's attention masks and target for train and test
input_ids_train = train_encoded_data['input_ids']
attention_masks_train = train_encoded_data['attention_mask']
train_target = torch.tensor(ytrain)

input_ids_test = test_encoded_data['input_ids']
attention_masks_test = test_encoded_data['attention_mask']
test_target = torch.tensor(ytest)

#create tensor data sets
train_set = TensorDataset(input_ids_train, attention_masks_train, train_target)
test_set = TensorDataset(input_ids_test, attention_masks_test, test_target)

#create small pre-trained bert model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased',
                                                      num_labels=4,
                                                      output_attentions=False,
                                                      output_hidden_states=False)

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

batch_size=1 #small batch size to preserve memory

data_loader_train = DataLoader(train_set, sampler=RandomSampler(train_set),
                               batch_size=batch_size)

data_loader_test = DataLoader(test_set, sampler=SequentialSampler(test_set),
                               batch_size=batch_size)

#create optimizer
from transformers import AdamW, get_linear_schedule_with_warmup

optimizer = AdamW(model.parameters(), lr=1e-5, eps=1e-8)

epochs = 2

scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, 
                                            num_training_steps=len(data_loader_train)*epochs)

#create helper functions to collect metrics

def get_f1(preds, labels):
  '''
  Returns f1 score given model's predictions and true labels
  '''
  preds_flat = np.argmax(preds, axis=1).flatten()
  labels_flat = labels.flatten()
  return f1_score(labels_flat, preds_flat, average='weighted')

def get_class_acc(preds, labels):
  '''
  Returns the accuracy per class
  '''
  preds_flat = np.argmax(preds, axis=1).flatten()
  labels_flat = labels.flatten()

  for label in np.unique(labels_flat):
    y_preds = preds_flat[labels_flat==label]
    y_true = labels_flat[labels_flat==label]
    print(f'Class: {label}')
    print(f'Accuracy: {len(y_preds[y_preds==label])}/{len(y_true)}\n')

#training loop
seed_val = 17
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

print(device)

def evaluate(dataloader_val):
  '''
  Evaluates model
  '''
  model.eval()

  loss_val_total = 0
  predictions, true_vals = [], []
    
  for batch in dataloader_val: 
    batch = tuple(b.to(device) for b in batch)
    inputs = {'input_ids': batch[0],
              'attention_mask': batch[1],
              'labels': batch[2],
              }

    with torch.no_grad():        
        outputs = model(**inputs)
            
    loss = outputs[0]
    logits = outputs[1]
    loss_val_total += loss.item()

    logits = logits.detach().cpu().numpy()
    label_ids = inputs['labels'].cpu().numpy()
    predictions.append(logits)
    true_vals.append(label_ids)
    
  loss_val_avg = loss_val_total/len(dataloader_val) 
    
  predictions = np.concatenate(predictions, axis=0)
  true_vals = np.concatenate(true_vals, axis=0)
            
  return loss_val_avg, predictions, true_vals

for epoch in tqdm(range(1, epochs+1)):
  model.train()
  loss_train_total = 0

  progress_bar = tqdm(data_loader_train, desc='Epoch {:1d}'.format(epoch), 
                      leave=False, disable=False)

  for batch in progress_bar:
    model.zero_grad()
    batch = tuple(b.to(device) for b in batch)
        
    inputs = {'input_ids': batch[0],
              'attention_mask': batch[1],
              'labels': batch[2],
              }       

    outputs = model(**inputs)
        
    loss = outputs[0]
    loss_train_total += loss.item()
    loss.backward()

    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    optimizer.step()
    scheduler.step()
        
    progress_bar.set_postfix({'training_loss': '{:.3f}'.format(loss.item()/len(batch))})
        
  torch.save(model.state_dict(), f'finetuned_BERT_epoch_{epoch}.model')
        
  tqdm.write(f'\nEpoch {epoch}')
    
  loss_train_avg = loss_train_total/len(data_loader_train)            
  tqdm.write(f'Training loss: {loss_train_avg}')
    
  test_loss, predictions, true_vals = evaluate(data_loader_test)
  test_f1 = get_f1(predictions, true_vals)
  tqdm.write(f'Test loss: {test_loss}')
  tqdm.write(f'F1 Score (Weighted): {test_f1}')

model = BertForSequenceClassification.from_pretrained("bert-base-uncased",
                                                      num_labels=4,
                                                      output_attentions=False,
                                                      output_hidden_states=False)

model.to(device)

model.load_state_dict(torch.load('/content/finetuned_BERT_epoch_2.model', map_location=torch.device('cpu')))

_, predictions, true_vals = evaluate(data_loader_test)
get_class_acc(predictions, true_vals)