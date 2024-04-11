import argparse
import numpy as np
import nltk
from nltk.tokenize import sent_tokenize
import tqdm
import json
import torch
import os
import random
from transformers import GPT2Tokenizer, GPT2LMHeadModel, LogitsProcessorList, AutoTokenizer, AutoModelForCausalLM

nltk.download('punkt')


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default="./iclr-data/ICLR_non_native_data_abstract.jsonl")
parser.add_argument('--output_dir', default="iclr-data")
parser.add_argument('--model_size', default="xl")
parser.add_argument('--num_instances', default=3000, type=int)
parser.add_argument('--num_samples', default=1, type=int)
parser.add_argument('--max_new_tokens', default=512, type=int)
parser.add_argument('--top_k', default=None, type=int)
parser.add_argument('--top_p', default=0.9, type=float)
parser.add_argument('--typical_p', default=None, type=float)
parser.add_argument('--num_shards', default=1, type=int)
parser.add_argument('--local_rank', default=0, type=int)
parser.add_argument('--strength', default=0.0, type=float)
parser.add_argument('--debug', action="store_true")
args = parser.parse_args()

data = []
with open(args.dataset, "r") as f:
    for x in f.read().strip().split("\n"):
        data.append(json.loads(x))


tokenizer = AutoTokenizer.from_pretrained('gpt2-xl', local_files_only=True)
tokenizer.pad_token = tokenizer.eos_token
model = GPT2LMHeadModel.from_pretrained(f"gpt2-{args.model_size}")
model.cuda()
model.eval()

output_file = f"{args.output_dir}/gpt2_{args.model_size}_strength_{args.strength}_{args.max_new_tokens}_len_top_p_{args.top_p}_intro.jsonl"


random.seed(43)
device = "cuda" if torch.cuda.is_available() else "cpu"

outputs = []


if os.path.exists(output_file):
    with open(output_file, "r") as f:
        num_curr_outputs = len(f.read().strip().split("\n"))
else:
    num_curr_outputs = 0
    
def trim_to_shorter_length(texta, textb):
    # truncate to shorter of o and s
    shorter_length = min(len(texta.split( )), len(textb.split( )))
    texta = ' '.join(texta.split()[:shorter_length])
    textb = ' '.join(textb.split()[:shorter_length])
    return texta, textb

def strip_newlines(text):
    return ' '.join(text.split())

for idx, data in tqdm.tqdm(enumerate(data), total=min(len(data), args.num_instances)):
    if idx < num_curr_outputs:
        continue
    if len(outputs) >= args.num_instances:
        break
        
    print(data)
    country = data['country']  
    continent = data['continent']
    dd = data['inputs']
    
    try:
        gold_sents = sent_tokenize(dd)
        # use the first 2 sentences as prefix
        prefix = " ".join(gold_sents[:2])
        gold_completion = " ".join(gold_sents[30:])

        all_encoded = tokenizer(dd, truncation=True, padding="longest", return_tensors="pt", max_length=1024 - args.max_new_tokens).to(device)
        all_encoded = {key: value[:, :30] for key, value in all_encoded.items()}


        tries = 0

        torch.manual_seed(0)
        np.random.seed(0)

        with torch.inference_mode():
            generation = model.generate(**all_encoded,
                logits_processor=None,
                do_sample=True,
                min_length=100, 
                max_new_tokens=args.max_new_tokens,
                top_k=args.top_k,
                typical_p=args.typical_p,
                top_p=args.top_p,
                num_return_sequences=args.num_samples,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id)
            decode = tokenizer.batch_decode(generation, skip_special_tokens=True)[0]

        decode= strip_newlines(decode)

        gold_text, gen_text = trim_to_shorter_length(dd, decode)
        print("gold : ", gold_text)
        print("generation : ", gen_text)

        outputs.append(json.dumps({
            "prefix": prefix,
            "country": country,
            "continent": continent,
            "gold_completion": gold_text,
            "gen_completion": gen_text
        }))

        with open(output_file, "a") as f:
            f.write("\n".join(outputs) + "\n")
            outputs = []

    except:
        pass


