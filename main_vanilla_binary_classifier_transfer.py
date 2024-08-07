import argparse
import json
import tqdm
import functools
import os
import numpy as np
import pickle
import torch
import random
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import transformers
from sklearn.metrics import roc_auc_score
from transformers import AutoTokenizer, AutoModel
from dataloader import *
from detection_utils import *

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_source', default="xsum-data/open-generation-data")
    parser.add_argument('--train_output_file', default="chatgpt.jsonl_pp")
    parser.add_argument('--test_source', default="lfqa-data")
    parser.add_argument('--test_output_file', default="chatgpt.jsonl_pp")
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--max_len', default=256, type=int)
    args = parser.parse_args()
    return args

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    random.seed(seed)



class Detector(torch.nn.Module):
    def __init__(self):
        super(Detector, self).__init__()
        self.l1 = transformers.AutoModel.from_pretrained("roberta-base")
        self.l1.classifier = nn.Sequential()
        self.pre_classifier = torch.nn.Linear(768, 768)
        self.dropout = torch.nn.Dropout(0.3)
        self.classifier = torch.nn.Linear(768, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, attention_mask, token_type_ids):
        output_1 = self.l1(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        
        hidden_state = output_1[0]
        pooler = hidden_state[:, 0]
        
        pooler = self.pre_classifier(pooler)
        pooler = torch.nn.ReLU()(pooler)
        pooler = self.dropout(pooler)
        output = self.classifier(pooler)
        output = self.sigmoid(output)
        
        return output
    


if __name__ == "__main__":
    args = get_args()
    set_seed(0)


    tokenizer = transformers.AutoTokenizer.from_pretrained("roberta-base")
    model = Detector()
    model.cuda()
    optimizer = AdamW(model.parameters(), lr = 1e-5,  eps = 1e-8 )
    
    train_data = []
    file_name_list = args.train_source.split('/')
    
    for data_name in file_name_list:
        file_dir = '{}/{}'.format(data_name, args.train_output_file)
        with open(file_dir, "r") as f:
            train_data_temp = [json.loads(x) for x in f.read().strip().split("\n")]
            
        train_data += train_data_temp
        
    test_data = [] 
    
    file_name_list = args.test_source.split('/')
        
    for data_name in file_name_list:
        file_dir = '{}/{}'.format(data_name, args.test_output_file)
        with open(file_dir, "r") as f:
            test_data_temp = [json.loads(x) for x in f.read().strip().split("\n")]
            
        test_data += test_data_temp
        

    
    train_data = random.sample(train_data, int(1.0*len(train_data)))
    print(len(train_data), len(test_data))

    training_set = GHData(args.train_output_file, train_data, tokenizer, args.max_len)
    testing_set = GHData(args.test_output_file, test_data, tokenizer, args.max_len)
    training_loader = DataLoader(training_set, batch_size =args.batch_size, shuffle=True, num_workers=0, drop_last=True)
    testing_loader = DataLoader(testing_set, batch_size =args.batch_size, shuffle=False, num_workers=0, drop_last=True)
    

    model.train()
    
    bce_criterion = nn.BCELoss()
    print(len(training_loader))
    
    for idx, data in enumerate(training_loader):
        print(idx)
        batch_size = data[0].shape[0]
        gen_input_ids = data[0].cuda()
        gen_attention_mask = data[1].cuda()
        gen_target = torch.ones(batch_size).cuda()

        gold_input_ids = data[2].cuda()
        gold_attention_mask = data[3].cuda()
        gold_target = torch.zeros(batch_size).cuda()

        input_ids = torch.cat((gen_input_ids, gold_input_ids, gen_input_ids, gold_input_ids))
        attention_mask = torch.cat((gen_attention_mask, gold_attention_mask, gen_attention_mask, gold_attention_mask))
        
        
        model.zero_grad()
        output = model(input_ids,  attention_mask, None)
        output_gen, output_gold = torch.split(output, batch_size*2)
        output = output.squeeze(1)
        output_gen = output_gen.squeeze(1)
        output_gold = output_gold.squeeze(1)
        
        target = torch.cat((gen_target, gold_target, gen_target, gold_target))
        loss = bce_criterion(output, target) 

        loss.backward()
        optimizer.step()


    acc_gen = []
    acc_gold = []
    acc_pp0 = []
    target_gen = []
    target_gold = []

    model.eval()

    for idx, data in enumerate(testing_loader):
        print(idx)
        batch_size = data[0].shape[0]
        gen_input_ids = data[0].cuda()
        gen_attention_mask = data[1].cuda()
        gen_target = torch.ones(batch_size)

        gold_input_ids = data[2].cuda()
        gold_attention_mask = data[3].cuda()
        gold_target = torch.zeros(batch_size)

        pp_input_ids = data[4].cuda()
        pp_attention_mask = data[5].cuda()

        input_ids = torch.cat((gen_input_ids, gold_input_ids, pp_input_ids))
        attention_mask = torch.cat((gen_attention_mask, gold_attention_mask, pp_attention_mask))
        with torch.no_grad():
            output = model(input_ids, attention_mask, None)
        gen_z, gold_z, pp0_z = torch.split(output, batch_size)

        acc_gen.append(gen_z.detach().cpu())
        target_gen.append(gen_target.detach().cpu())
        acc_gold.append(gold_z.detach().cpu())
        target_gold.append(gold_target.detach().cpu())
        acc_pp0.append(pp0_z.detach().cpu())
        


        del gen_z, gold_z, pp0_z, gen_target, gold_target



    acc_gen = torch.cat(acc_gen).squeeze(1)
    target_gen = torch.cat(target_gen)
    acc_gold = torch.cat(acc_gold).squeeze(1)
    target_gold = torch.cat(target_gold)
    acc_pp0 = torch.cat(acc_pp0).squeeze(1)

    auroc1 = roc_auc_score(torch.cat((target_gen, target_gold)).numpy(), torch.cat((acc_gen, acc_gold)).numpy())
    auroc2 = roc_auc_score(torch.cat((target_gen, target_gold)).numpy(), torch.cat((acc_pp0, acc_gold)).numpy())
    print("generation AUROC : ", auroc1)
    print("paraphrase AUROC : ", auroc2)
