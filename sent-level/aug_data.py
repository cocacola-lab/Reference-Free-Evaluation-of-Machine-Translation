import random
import time
import os
import logging
import torch
import nlpaug.augmenter.word as naw
import argparse
import random
import tqdm
import torch
import os
import time
import logging
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup, AdamW

logger = logging.getLogger(__name__)
logging.basicConfig(
		format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
		datefmt="%m/%d/%Y %H:%M:%S",
		level=logging.INFO,
	)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)

def swap_text_a_b(a_text, b_text):
    swap_num = len(a_text) // 3
    index = list(range(len(a_text)))
    to_be_swaped = random.sample(index, swap_num)
    for i in to_be_swaped:
        a_text[i], b_text[i] = b_text[i], a_text[i]
    return a_text, b_text

def shuffle_sents(a_text, permute_num, permuted_file):
    all_permuted_texts, all_labels = [], []
    all_permuted_texts.append(a_text) 
    for i in range(permute_num):
        permuted_path = os.path.join(permuted_file, type+'_texts_'+str(i)+'.txt')
        f_permute = open(permuted_path, 'r').readlines()
        permuted_lines = [line.strip() for line in f_permute]
        print(len(permuted_lines))
        all_permuted_texts.append(permuted_lines)

    for i in range(len(all_permuted_texts[0])):
        label = random.randint(0, len(all_permuted_texts)-1)
        all_labels.append(label)
        all_permuted_texts[0][i], all_permuted_texts[label][i] = all_permuted_texts[label][i], all_permuted_texts[0][i]
    return all_permuted_texts, all_labels

def sent_trans_model(teacher_tokenizer, teacher_model, all_source, all_permuted_texts, teacher_batch_size):
    def mean_pooling(model_output, attention_mask):
        token_embeddings = model_output[0] 
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()

        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    score_list = []
    teacher_model.eval()
    for i in tqdm.tqdm(range(len(all_permuted_texts))):
        logger.info("%d labels begin to generate." % (i))
        column_score_list = []
        for batch_start in tqdm.tqdm(range(0, len(all_source), teacher_batch_size)):
            source = all_source[batch_start: min(batch_start+teacher_batch_size, len(all_permuted_texts[0]))]
            cands = all_permuted_texts[i][batch_start: min(batch_start+teacher_batch_size, len(all_permuted_texts[0]))]
            encoded_input1 = teacher_tokenizer(source, padding=True, truncation=True, return_tensors="pt")
            encoded_input2 = teacher_tokenizer(cands, padding=True, truncation=True, return_tensors="pt")

            with torch.no_grad():
                model_output1 = teacher_model(
                    input_ids = encoded_input1['input_ids'].to(device),
                    attention_mask = encoded_input1['attention_mask'].to(device)
                )
                model_output2 = teacher_model(
                    input_ids = encoded_input2['input_ids'].to(device),
                    attention_mask = encoded_input2['attention_mask'].to(device)
                )

            sentence_embeddings1 = mean_pooling(model_output1, encoded_input1['attention_mask'].to(device))
            # [batch_size, hidden_size]
            sentence_embeddings2 = mean_pooling(model_output2, encoded_input2['attention_mask'].to(device))

            sentence_embeddings1 = F.normalize(sentence_embeddings1, p=2, dim=1)
            sentence_embeddings2 = F.normalize(sentence_embeddings2, p=2, dim=1)

            cosine_score = F.cosine_similarity(sentence_embeddings1, sentence_embeddings2)
            
            column_score_list.extend(cosine_score.tolist())
        score_list.append(column_score_list)
    
    return torch.tensor(score_list).transpose(0, 1)


def permute_sub_sequence(text, path):
    f_out = open(path, 'w')
    permuted_texts = []
    for line in text:
        num_sub_seq = 1
        if "," in line:
            num_sub_seq = len(line.split(','))
        elif "，" in line:
            num_sub_seq = len(line.split('，'))
        if ("，" and "," not in line) or num_sub_seq < 3:
            flag = 0
            if " 。" in line:
                sub_seq_list = line.replace(' 。', '').split(' ')
                flag = 1 # chinese
            elif " ." in line:
                sub_seq_list = line.replace(' .', '').split(' ')
                flag = 2 # english
            else:
                sub_seq_list = line.split(' ')
            if len(sub_seq_list) == 1:
                permuted_texts.append(line)
                f_out.writelines(permuted_res)
                f_out.writelines('\n')
                continue
            start = random.randint(1, len(sub_seq_list)-1)
            sub_seq_list = sub_seq_list[start:] + sub_seq_list[: start]
            if flag==1:
                permuted_res = ' '.join(sub_seq_list)+' 。'
                permuted_texts.append(permuted_res)
            elif flag==2:
                permuted_res = ' '.join(sub_seq_list)+' .'
                permuted_texts.append(permuted_res)
            else:
                permuted_res = ' '.join(sub_seq_list)
                permuted_texts.append(permuted_res)

        elif "，" in line:
            line = line.strip().replace('。 」', ' 」').replace('。', '，').replace("：", '，')
            sub_seq_list = line.split('，')
            if sub_seq_list[-1] == "":
                sub_seq_list = sub_seq_list[:-1]
            pre = sub_seq_list.copy()
            start = time.perf_counter()
            while(pre == sub_seq_list):
                random.shuffle(sub_seq_list)
                end = time.perf_counter()
                if round(end-start) > 10:
                    logger.info("Time out...")
                    logger.info(sub_seq_list)
                    break
            permuted_res = '，'.join(sub_seq_list)+'。'
            permuted_texts.append(permuted_res)
        elif "," in line:
            flag = 0
            if " 。" in line:
                flag = 1
            elif " ." in line:
                flag = 2
            line = line.strip().replace('.', ',').replace(":", ',').replace('。', ',').replace("：", ',')
            sub_seq_list = line.split(',')
            if sub_seq_list[-1] == "":
                sub_seq_list = sub_seq_list[:-1]
            pre = sub_seq_list.copy()
            start = time.perf_counter()
            while(pre == sub_seq_list):
                random.shuffle(sub_seq_list)
                end = time.perf_counter()
                if round(end-start) > 10:
                    logger.info("Time out...")
                    logger.info(sub_seq_list)
                    break
            if flag == 1:
                permuted_res = ','.join(sub_seq_list) + '。'
            elif flag == 2:
                permuted_res = ','.join(sub_seq_list)+'.'
            else:
                permuted_res = ', '.join(sub_seq_list)
            permuted_texts.append(permuted_res)
        f_out.writelines(permuted_res)
        f_out.writelines('\n')
    # return permuted_texts


def generate_augmentation_file(examples, augmented_file, aug_num):
    for i in range(aug_num):
        path = os.path.join(augmented_file, type+'_texts_'+str(i)+'.txt')
        if type == 'del':
            logger.info("Augment type is DELETION")
            deletion(examples=examples, path=path)
        elif type == 'rep':
            logger.info("Augment type is Repetition")
            repetition(examples=examples, path=path, dup_rate=0.4)
        elif type == 'shuf':
            logger.info("Augment type is SHUFFLE")
            shuf_word(examples=examples, path=path, shuf_rate=0.5)
        elif type == 'shuf_whole':
            logger.info("Augment type SHUFFLE all the words")
            all_shuf(examples=examples, path=path)
        elif type == 'permuted':
            logger.info("Augment type is PERMUTE")
            permute_sub_sequence(text=examples, path=path)
        else:
            logger.info("Type error!")
            break
        logger.info("Generation of %d / %d augmented file is finished." % (i+1, aug_num))


type = 'permuted'
augmented_file = type+'_data_mix/'
os.makedirs(augmented_file, exist_ok=True)
aug_num = 7
teacher_model_type = SENTENCE_BERT_MODEL_PATH

device = "cuda:1"

f_text_a = open('all_ref.txt').readlines()
f_text_b = open('all_source.txt').readlines()
a_textlines, b_textlines = [], []
for i in range(len(f_text_a)):
    a, b = f_text_a[i].strip(), f_text_b[i].strip()
    a_textlines.append(a)
    b_textlines.append(b)

a_text, b_text = a_textlines.copy(), b_textlines.copy()
print(len(b_text))
generate_augmentation_file(a_text, augmented_file, aug_num)

all_augmented_texts, all_labels = shuffle_sents(a_text, aug_num, augmented_file)
label_path = os.path.join(augmented_file, 'label_tensor.pt')
torch.save(torch.tensor(all_labels), label_path)

print(b_text[:5])

for i in range(len(all_augmented_texts)):
    path = os.path.join(augmented_file, 'all_'+type+'_texts_'+str(i)+'.txt')
    f_aug = open(path, 'w')
    print(i)
    print(len(all_augmented_texts[i]))
    
    assert len(b_text) == len(all_augmented_texts[i])
    print(all_augmented_texts[i][:5])
    for line in all_augmented_texts[i]:
        f_aug.writelines(line+'\n')


teacher_batch_size = 100
teacher_tokenizer = AutoTokenizer.from_pretrained(teacher_model_type)
teacher_model = AutoModel.from_pretrained(teacher_model_type).to(device)
teacher_labels = sent_trans_model(teacher_tokenizer, teacher_model, b_text, all_augmented_texts, teacher_batch_size)
tensor_path = os.path.join(augmented_file, 'sent_trans_score.pt')
torch.save(teacher_labels, tensor_path)
print(teacher_labels[0])