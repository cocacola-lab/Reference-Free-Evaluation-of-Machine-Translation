import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from bert_score_for_training import get_model, bert_cos_score_idf, score

class BertScore_Model(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_type)
        self.model = get_model(args.model_type, num_layers=9, all_layers=None)
    
    def forward(self, permuted_texts, source, args):
        all_cands, all_source = [], []
        for i in range(len(permuted_texts)):
            all_cands.extend(permuted_texts[i])
            all_source.extend(source)
        all_score = score(self.tokenizer, self.model, all_cands, all_source, model_type=args.model_type, num_layers=9, idf=False)[-1]
        batch_score = all_score.view(-1, min(len(source), args.batch_size)).transpose(0, 1)
        return batch_score