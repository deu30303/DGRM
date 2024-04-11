from torch.utils.data import Dataset, DataLoader
import torch

class GHData(Dataset):
    def __init__(self, args, data, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.data = data
        self.max_len = max_len
        self.gen_list = []
        self.gold_list = []
        self.pp0_list = []

        for index in range(0,len(self.data)):
            if isinstance(self.data[index]['gen_completion'], str):
                gen_tokens = self.data[index]['gen_completion'].split()
            else:
                gen_tokens = self.data[index]['gen_completion'][0].split()

            gold_tokens = self.data[index]['gold_completion'].split()
            
            try:
                pp0_tokens = self.data[index]['paraphrase_outputs'][args.paraphrase_type]['output'][0].split()
            except:
                pp0_tokens = self.data[index]['paraphrase_outputs'][0].split()
                
            gen_text = " ".join(gen_tokens)     
            gold_text = " ".join(gold_tokens)
            pp0_text = " ".join(pp0_tokens)
            
            gen_input = tokenizer.encode_plus(gen_text,  None, add_special_tokens=True, max_length=self.max_len, \
                                              pad_to_max_length=True, return_token_type_ids=False)

            gold_input = tokenizer.encode_plus(gold_text,  None, add_special_tokens=True, max_length=self.max_len, \
                                              pad_to_max_length=True, return_token_type_ids=False)

            pp0_input = tokenizer.encode_plus(pp0_text,  None, add_special_tokens=True,max_length=self.max_len, \
                                              pad_to_max_length=True, return_token_type_ids=False)
            self.gen_list.append(gen_input)
            self.gold_list.append(gold_input)
            self.pp0_list.append(pp0_input)
 

    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):                       
        gen_input = self.gen_list[index]
        gold_input = self.gold_list[index]
        pp0_input = self.pp0_list[index]
        
        return torch.tensor(gen_input['input_ids']), torch.tensor(gen_input['attention_mask']), \
                torch.tensor(gold_input['input_ids']), torch.tensor(gold_input['attention_mask']), \
                torch.tensor(pp0_input['input_ids']), torch.tensor(pp0_input['attention_mask'])