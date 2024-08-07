# DGRM: Enhancing Domain Generalization for Robust Machine-Generated Text Detection

Pytorch Implementation of DGRM: Enhancing Domain Generalization for Robust Machine-Generated Text Detection


## To run the code ##
GPT 3 XSUM experiments with Vanilla BCE + DGRM 
```
python main_dgrm_classifier_transfer.py --train_source lfqa-data/open-generation-data --train_output_file gpt3.jsonl_pp --test_source xsum-data --test_output_file gpt3.jsonl_pp --batch_size 8
```
