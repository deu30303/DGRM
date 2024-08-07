import argparse
import json
import tqdm
import functools
import os
import numpy as np
import pickle
import torch
import random
import copy
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import transformers
from sklearn.metrics import roc_auc_score
from transformers import AutoTokenizer, AutoModel, RobertaForMaskedLM, AdamW
import torch.nn.functional as F
from dataloader import *
from detection_utils import *

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_source', default="lfqa-data/xsum-data")
    parser.add_argument('--train_output_file', default="gpt3.jsonl_pp")
    parser.add_argument('--test_source', default="open-generation-data")
    parser.add_argument('--test_output_file', default="gpt3.jsonl_pp")
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--dim', default=768, type=int)
    parser.add_argument('--max_len', default=256, type=int)
    parser.add_argument('--local_rank', default=0, type=int)
    parser.add_argument('--temp', default=1.0, type=float)
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
        print(self.l1)
        self.l1.classifier = nn.Sequential()
        self.pre_classifier = torch.nn.Linear(args.dim, args.dim)
        self.aux_classifier = nn.Linear(args.dim, 1)
        self.dropout = torch.nn.Dropout(0.3)
        self.classifier = nn.Linear(args.dim, 1)
        self.domain_classifier = nn.Linear(args.dim, 1)
        self.sigmoid = nn.Sigmoid()
        self.s_seperation = nn.Sequential(nn.Linear(768, args.dim, bias=False), 
                                        nn.BatchNorm1d(args.dim),
                                        nn.ReLU(inplace=True), 
                                        nn.Linear(args.dim, args.dim, bias=False))
        self.c_seperation = nn.Sequential(nn.Linear(768, args.dim, bias=False), 
                                        nn.BatchNorm1d(args.dim),
                                        nn.ReLU(inplace=True), 
                                        nn.Linear(args.dim, args.dim, bias=False))
 
        self.recalibration = nn.Sequential(nn.Linear(args.dim, args.dim, bias=False), 
                                           nn.BatchNorm1d(args.dim),
                                           nn.ReLU())
        
        self.predictionMLP = nn.Sequential(
            nn.Linear(args.dim, args.dim, bias=False),
            nn.BatchNorm1d(args.dim),
            nn.ReLU(inplace=True),
            nn.Linear(args.dim, 256, bias=False),
        )


 
 
    def forward(self, input_ids, attention_mask, token_type_ids, is_eval=False, swap=None, residual=None):
        output_1 = self.l1(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        hidden_state = output_1[0]
        pooler = hidden_state[:, 0]

       
        c_feature = self.c_seperation(pooler)
        t_feature = self.s_seperation(pooler)
        
        c_feature_proj = self.predictionMLP(self.dropout(c_feature))
        
        if residual == None:
            t_output = self.aux_classifier(t_feature)
        else:
            residual = grad_reverse(residual, lambd=0.2)
            residual = self.pre_classifier(residual)
            residual = torch.nn.ReLU()(residual)
            t_output = self.classifier(residual)
            
        t_output = self.aux_classifier(t_feature)
        
        
        if swap == None:
            pooler = self.recalibration(c_feature) + t_feature
            swap_output = None
        else:
            lam = np.random.beta(0.2, 0.2)
            lam = max(lam, 1-lam)
            swap = (1-lam)*swap + lam*c_feature
            pooler = self.recalibration(swap) + t_feature

        pooler = self.pre_classifier(pooler)
        pooler = torch.nn.ReLU()(pooler)
        pooler = self.dropout(pooler)
        output = self.classifier(pooler)
        
        output = self.sigmoid(output)
        t_output = self.sigmoid(t_output)
        return output, t_output, t_feature, c_feature, c_feature_proj
    


if __name__ == "__main__":
    args = get_args()
    set_seed(0)
    tokenizer = transformers.AutoTokenizer.from_pretrained("roberta-base")
    model = Detector()
    model.cuda()
    
    mlm_model = RobertaForMaskedLM.from_pretrained('roberta-base')
    mlm_model.cuda()

    optimizer = AdamW(model.parameters(), lr = 1e-5,  eps = 1e-8 )
    train_data = []

    file_name_list = args.train_source.split('/')
    domain_target = 0
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
    training_set = GHData(args.train_source, train_data, tokenizer, args.max_len)
    testing_set = GHData(args.test_source, test_data, tokenizer, args.max_len)
    training_loader = DataLoader(training_set, batch_size =args.batch_size, shuffle=True, num_workers=0, drop_last=True)
    testing_loader = DataLoader(testing_set, batch_size =args.batch_size, shuffle=False, num_workers=0, drop_last=True)

    model.train()
    bce_criterion = nn.BCELoss()
    cos = nn.CosineSimilarity(dim=1, eps=1e-6).cuda()
    nce_criterion = InfoNCE(temperature=1.0)
    ce_criterion = nn.CrossEntropyLoss().cuda()

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
        gen_text = tokenizer.batch_decode(gen_input_ids, padding=False)
 
        input_ids = torch.cat((gen_input_ids, gold_input_ids))
        attention_mask = torch.cat((gen_attention_mask, gold_attention_mask))
        
        mask_input_ids = mask_tokens(input_ids.cpu(), tokenizer).cuda()
        
        
        m_outputs = mlm_model(mask_input_ids, attention_mask)

        x_input_ids= m_outputs.logits.argmax(-1)
        
        nsp_inputs = torch.cat((input_ids, x_input_ids)).cuda()     
        nsp_masks = torch.cat((attention_mask, attention_mask)).cuda()
        
        
        model.zero_grad() 
        output, t_output, t_feature, c_feature, c_feature_proj = model(nsp_inputs, nsp_masks, None)
        t_feature_gen, t_feature_gold = torch.split(t_feature[:batch_size*2], batch_size)
        c_feature_gen, c_feature_gold = torch.split(c_feature[:batch_size*2], batch_size)
        
        
        loss_sim = nce_criterion(c_feature_proj)
        

        t_feature_logit = torch.cat((cos(t_feature_gen, t_feature_gold).unsqueeze(1), cos(t_feature_gen, c_feature_gold).unsqueeze(1), \
                                     cos(t_feature_gen, c_feature_gen).unsqueeze(1), cos(t_feature_gold, c_feature_gold).unsqueeze(1), \
                                     cos(t_feature_gold, c_feature_gen).unsqueeze(1)), dim=1).cuda()
        c_feature_logit = torch.cat((cos(c_feature_gen, c_feature_gold).unsqueeze(1), cos(t_feature_gen, c_feature_gold).unsqueeze(1), \
                                      cos(t_feature_gen, c_feature_gen).unsqueeze(1), cos(t_feature_gold, c_feature_gold).unsqueeze(1), \
                                     cos(t_feature_gold, c_feature_gen).unsqueeze(1)), dim=1).cuda()
        target_orth = torch.zeros(batch_size).long().cuda()
        loss_orth =  (ce_criterion(t_feature_logit / args.temp, target_orth) + ce_criterion(c_feature_logit / args.temp, target_orth))

        
        permute_idx = np.random.permutation(batch_size*2).tolist()
        c_feature_swp = c_feature[:batch_size*2][permute_idx]
        
        alpha = 0.5
        t_feature_org, t_feature_per = torch.split(t_feature, batch_size*2)
        t_feature_res = (t_feature_per - alpha*t_feature_org) / (1 - alpha) 
        
        swap_output, residual_output, _, _, _ = model(input_ids, attention_mask, None, swap=c_feature_swp, residual=t_feature_res)
        target = torch.cat((gen_target, gold_target, gen_target, gold_target))
        
        output = output.squeeze(1)
        t_output = t_output.squeeze(1)
        swap_output = swap_output.squeeze(1)
        residual_output = residual_output.squeeze(1)
        
        swap_loss = bce_criterion(swap_output[:batch_size*2], target[:batch_size*2])
        res_loss = bce_criterion(residual_output[:batch_size], target[:batch_size]) 
        t_loss = bce_criterion(t_output[:batch_size*2], target[:batch_size*2],) 
        
        loss = swap_loss + t_loss + loss_sim + 0.3*res_loss + 0.3*loss_orth
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
            output = model(input_ids, attention_mask,  None, is_eval=True)
        gen_z, gold_z, pp0_z = torch.split(output[0], batch_size)
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
    # target_pp0 = torch.cat(target_pp0).numpy()
    auroc1 = roc_auc_score(torch.cat((target_gen, target_gold)).numpy(), torch.cat((acc_gen, acc_gold)).numpy())
    auroc2 = roc_auc_score(torch.cat((target_gen, target_gold)).numpy(), torch.cat((acc_pp0, acc_gold)).numpy())
    print("generation AUROC : ", auroc1)
    print("paraphrase AUROC : ", auroc2)
