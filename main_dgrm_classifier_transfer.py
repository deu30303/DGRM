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
import transformers
from dataloader import *
from detection_utils import *
from sklearn.metrics import roc_auc_score
from transformers import AutoTokenizer, AdamW
import torch.nn.functional as F

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_source', default="xsum-data/open-generation-data")
    parser.add_argument('--train_output_file', default="gpt2.jsonl_pp")
    parser.add_argument('--test_source', default="lfqa-data")
    parser.add_argument('--test_output_file', default="gpt2.jsonl_pp")
    parser.add_argument('--lr', type=float, default=1e-5, help='learning rate (default: 1e-5)')
    parser.add_argument('--eps', type=float, default=1e-8, help='learning rate (default: 1e-8)')
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--dim', default=768, type=int)
    parser.add_argument('--max_len', default=256, type=int)
    parser.add_argument('--alpha', default=0.5, type=float)
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
        self.l1 = transformers.BertModel.from_pretrained("bert-base-uncased")
        self.l1.classifier = nn.Sequential()
        self.pre_classifier = torch.nn.Linear(args.dim, args.dim)
        self.aux_classifier = nn.Linear(args.dim, 1)
        self.nsp_classifier = nn.Linear(args.dim, 1)
        self.dropout = torch.nn.Dropout(0.1)
        self.classifier = nn.Linear(args.dim, 1)
        self.domain_classifier = nn.Linear(args.dim, 1)
        self.sigmoid = nn.Sigmoid()
        self.a_seperation = nn.Sequential(nn.Linear(768, args.dim), 
                                        nn.ReLU())
        self.na_seperation = nn.Sequential(nn.Linear(768, args.dim), 
                                        nn.ReLU())
 
        self.recalibration = nn.Sequential(nn.Linear(args.dim, args.dim), 
                                        nn.ReLU())
        self.residual = nn.Sequential(nn.Linear(args.dim, args.dim), 
                                        nn.ReLU())

 
 
    def forward(self, input_ids, attention_mask, token_type_ids, is_eval=False, swap=None, residual=None):
        output_1 = self.l1(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        hidden_state = output_1[0]
        pooler = hidden_state[:, 0]
        a_feature = self.a_seperation(pooler)
        na_feature = self.na_seperation(pooler)
        d_output = self.domain_classifier(a_feature)
        nsp_output = self.nsp_classifier(a_feature)
        
        if residual == None:
            a_output = self.aux_classifier(a_feature)
        else:
            residual = self.pre_classifier(residual)
            residual = torch.nn.ReLU()(residual)
            residual = grad_reverse(residual, lambd=1.0)
            a_output = self.classifier(residual)
            
        na_output = self.aux_classifier(na_feature)
        if swap == None:
            pooler = self.recalibration(a_feature) + na_feature
        else:
            lam = np.random.beta(0.2, 0.2)
            lam = max(lam, 1-lam)
            swap = (1-lam)*swap + lam*a_feature
            pooler = self.recalibration(swap) + na_feature
 
        pooler = self.pre_classifier(pooler)
        pooler = torch.nn.ReLU()(pooler)
        output = self.classifier(pooler)
        output = self.sigmoid(output)
        a_output = self.sigmoid(a_output)
        na_output = self.sigmoid(na_output)
        d_output = self.sigmoid(d_output)
        nsp_output = self.sigmoid(nsp_output)
        
        return output, a_output, na_output, nsp_output, a_feature, na_feature, d_output
    

 
 
if __name__ == "__main__":
    args = get_args()
    set_seed(0)
    tokenizer = transformers.BertTokenizer.from_pretrained("bert-base-uncased")
    model = Detector()
    model.cuda()
    optimizer = AdamW(model.parameters(), lr = args.lr,  eps = args.eps )
    train_data = []

    file_name_list = args.train_source.split('/')
    domain_target = 0
    for data_name in file_name_list:
        file_dir = './data/{}/{}'.format(data_name, args.train_output_file)
        with open(file_dir, "r") as f:
            train_data_temp = [json.loads(x) for x in f.read().strip().split("\n")]
        train_data += train_data_temp

    test_data = []
    file_name_list = args.test_source.split('/')
    for data_name in file_name_list:
        file_dir = './data/{}/{}'.format(data_name, args.test_output_file)
        with open(file_dir, "r") as f:
            test_data_temp = [json.loads(x) for x in f.read().strip().split("\n")]
        test_data += test_data_temp
        
    print(len(train_data), len(test_data))
    training_set = GHData(args, train_data, tokenizer, args.max_len)
    testing_set = GHData(args, test_data, tokenizer, args.max_len)
    training_loader = DataLoader(training_set, batch_size =args.batch_size, shuffle=True, num_workers=0, drop_last=True)
    testing_loader = DataLoader(testing_set, batch_size =args.batch_size, shuffle=False, num_workers=0, drop_last=True)

    model.train()
    bce_criterion = nn.BCELoss()
    cos = nn.CosineSimilarity(dim=1, eps=1e-6).cuda()
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
 
        input_ids = torch.cat((gen_input_ids, gold_input_ids))
        attention_mask = torch.cat((gen_attention_mask, gold_attention_mask))
        x_input_ids= random_seq(input_ids)
        p_x_input_ids, p_mask, _ = get_permutation_batch(x_input_ids, attention_mask)
        nsp_inputs = torch.cat((x_input_ids, p_x_input_ids)).cuda()     
        nsp_masks = torch.cat((p_mask, p_mask)).cuda()
        p_lbl = torch.ones(batch_size*2)
        r_lbl = torch.zeros(batch_size*2)
        nsp_lbl = torch.cat([r_lbl, p_lbl], dim=0).cuda()
        model.zero_grad()
 
        output, a_output, na_output, nsp_output, a_feature, na_feature, _ = model(nsp_inputs, None, None)
        a_feature_gen, a_feature_gold = torch.split(a_feature[:batch_size*2], batch_size)
        na_feature_gen, na_feature_gold = torch.split(na_feature[:batch_size*2], batch_size)
        na_feature_original, na_feature_permute = torch.split(na_feature, batch_size*2)
 
        permute_idx = np.random.permutation(batch_size*2).tolist()
        swap_a_feature = a_feature[:batch_size*2][permute_idx]
        alpha = 0.5
        residual_na_feature = (na_feature_permute - alpha*na_feature_original) / (1 -alpha) 
        swap_output, residual_output, _, _, _, _, _ = model(input_ids, attention_mask, None, swap=swap_a_feature, residual=residual_na_feature)

        a_feature_logit = torch.cat((cos(a_feature_gen, a_feature_gold).unsqueeze(1), cos(a_feature_gen, na_feature_gold).unsqueeze(1), \
                                     cos(a_feature_gen, na_feature_gen).unsqueeze(1), cos(a_feature_gold, na_feature_gold).unsqueeze(1), \
                                     cos(a_feature_gold, na_feature_gen).unsqueeze(1)), dim=1).cuda()
        na_feature_logit = torch.cat((cos(na_feature_gen, na_feature_gold).unsqueeze(1), cos(a_feature_gen, na_feature_gold).unsqueeze(1), \
                                      cos(a_feature_gen, na_feature_gen).unsqueeze(1), cos(a_feature_gold, na_feature_gold).unsqueeze(1), \
                                     cos(a_feature_gold, na_feature_gen).unsqueeze(1)), dim=1).cuda()
        
        target_orth = torch.zeros(batch_size).long().cuda()
        loss_orth =  (ce_criterion(na_feature_logit / args.temp, target_orth) + ce_criterion(a_feature_logit / args.temp, target_orth))
        target = torch.cat((gen_target, gold_target, gen_target, gold_target))
        
        output = output.squeeze(1)
        na_output = na_output.squeeze(1)
        nsp_output = nsp_output.squeeze(1)
        swap_output = swap_output.squeeze(1)
        residual_output = residual_output.squeeze(1)
        swap_loss = bce_criterion(swap_output[:batch_size*2], target[:batch_size*2])
        res_loss = bce_criterion(residual_output[:batch_size], target[:batch_size]) 
        na_loss = bce_criterion(na_output, target) 
        c_loss = bce_criterion(nsp_output, nsp_lbl)
        loss = swap_loss + 0.1*res_loss + na_loss + c_loss + 0.1*loss_orth
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
            output, a_output, na_output , _, _, _, _ = model(input_ids, attention_mask, None)
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