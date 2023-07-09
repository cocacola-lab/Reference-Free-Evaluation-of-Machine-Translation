import argparse
import random
import torch
import os
import time
import logging
import numpy as np
from torch.utils.data import RandomSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from bert_score_for_training import score, get_model
from bs_model import BertScore_Model
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

def swap_text_a_b(a_text, b_text):
    swap_num = len(a_text) // 3
    index = list(range(len(a_text)))
    to_be_swaped = random.sample(index, swap_num)
    for i in to_be_swaped:
        a_text[i], b_text[i] = b_text[i], a_text[i]
    return a_text, b_text

def permute_word(text, ratio):
    permuted_texts = []
    for line in text:
        text_list = line.split(' ')
        text_index = list(range(len(text_list)))
        permute_num = int(len(text_list) * ratio)
        to_be_permuted = random.sample(text_index, permute_num)
        to_be_permuted_text = []
        to_be_replaced = sorted(to_be_permuted)
        for i in to_be_permuted:
            to_be_permuted_text.append(text_list[i])
        for i, id in enumerate(to_be_replaced):
            text_list[id] = to_be_permuted_text[i]
        permuted_texts.append(' '.join(text_list))
    return permuted_texts

def permute_sub_sequence(text, ratio, path):
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
    return permuted_texts

def generate_permuted_sents(a_text, permuted_file, permute_num, ratio):
    for i in range(permute_num):
        path = os.path.join(permuted_file, 'permuted_texts_'+str(i)+'.txt')
        a_permuted_text = permute_sub_sequence(a_text, ratio, path)
        # p0, r0, f0 = score(a_text, b_text, model_type=args.model_type, num_layers=9, idf=False)
        # p, r, f = score(a_permuted_text, b_text, model_type=args.model_type, num_layers=9, idf=False)
        # print(f>f0)
        # index = torch.tensor(list(range(300)))[f>f0].tolist()
        # for j in index:
        #     print(a_permuted_text[j])
        #     print(a_text[j])
    # b_permuted_text = permute_word(b_text, ratio)

def shuffle_sents(a_text, permute_num, permuted_file):
    all_permuted_texts, all_labels = [], []
    all_permuted_texts.append(a_text)
    for i in range(permute_num):
        permuted_path = os.path.join(permuted_file, 'permuted_texts_'+str(i)+'.txt')
        f_permute = open(permuted_path, 'r').readlines()
        permuted_lines = [line.strip() for line in f_permute]
        all_permuted_texts.append(permuted_lines)

    for i in range(len(all_permuted_texts[0])):
        label = random.randint(0, len(all_permuted_texts)-1)
        all_labels.append(label)
        all_permuted_texts[0][i], all_permuted_texts[label][i] = all_permuted_texts[label][i], all_permuted_texts[0][i]
    return all_permuted_texts, all_labels

    #Mean Pooling - Take attention mask into account for correct averaging
def sent_trans_model(teacher_tokenizer, teacher_model, all_source, all_permuted_texts, args):
    def mean_pooling(model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    score_list = []
    teacher_model.eval()
    for i in tqdm.tqdm(range(len(all_permuted_texts))):
        column_score_list = []
        for batch_start in tqdm.tqdm(range(0, len(all_source), args.teacher_batch_size)):
            source = all_source[batch_start: min(batch_start+args.teacher_batch_size, len(all_permuted_texts[0]))]
            cands = all_permuted_texts[i][batch_start: min(batch_start+args.teacher_batch_size, len(all_permuted_texts[0]))]
            encoded_input1 = teacher_tokenizer(source, padding=True, truncation=True, return_tensors="pt")
            encoded_input2 = teacher_tokenizer(cands, padding=True, truncation=True, return_tensors="pt")

            with torch.no_grad():
                model_output1 = teacher_model(
                    input_ids = encoded_input1['input_ids'].to(args.device),
                    attention_mask = encoded_input1['attention_mask'].to(args.device)
                )
                model_output2 = teacher_model(
                    input_ids = encoded_input2['input_ids'].to(args.device),
                    attention_mask = encoded_input2['attention_mask'].to(args.device)
                )

            sentence_embeddings1 = mean_pooling(model_output1, encoded_input1['attention_mask'].to(args.device))
            # [batch_size, hidden_size]
            sentence_embeddings2 = mean_pooling(model_output2, encoded_input2['attention_mask'].to(args.device))

            sentence_embeddings1 = F.normalize(sentence_embeddings1, p=2, dim=1)
            sentence_embeddings2 = F.normalize(sentence_embeddings2, p=2, dim=1)

            cosine_score = F.cosine_similarity(sentence_embeddings1, sentence_embeddings2)
            
            column_score_list.extend(cosine_score.tolist())
        score_list.append(column_score_list)
    
    return torch.tensor(score_list).transpose(0, 1)

class dataset(torch.utils.data.Dataset):
    def __init__(self, all_source, all_permuted_texts, all_labels, teacher_labels):
        super().__init__()
        self.all_source, self.all_permuted_texts = all_source, all_permuted_texts
        self.all_labels, self.teacher_labels = all_labels, teacher_labels
    def __len__(self):
        return len(self.all_source)
    def __getitem__(self, index):
        perm = [self.all_permuted_texts[i][index] for i in range(len(self.all_permuted_texts))]
        # [a1, a2, ... ]
        return {
            "source": self.all_source[index], 
            "permuted_texts": perm, 
            "labels": self.all_labels[index], 
            "teacher_labels": self.teacher_labels[index]
        }


def train(model, tokenizer, train_dataset, args):
    # all_source, all_permuted_texts, all_labels, 
    # model = get_model(args.model_type, num_layers=9, all_layers=None)
    # model = BertScore_Model(args).to(args.device)
    # tokenizer = AutoTokenizer.from_pretrained(model_type)
    # teacher_tokenizer = AutoTokenizer.from_pretrained(args.teacher_model_type)
    # teacher_model = AutoModel.from_pretrained(args.teacher_model_type).to(args.device)
    # teacher_labels = sent_trans_model(teacher_tokenizer, teacher_model, all_source, all_permuted_texts, args)
    # teacher_labels = torch.load(os.path.join(args.permuted_file, 'sent_trans_score.pt'))
    # train_dataset = dataset(all_source, all_permuted_texts, all_labels, teacher_labels)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler)

    t_total = min(args.num_epoch * len(train_dataloader) // args.gradient_accumulation_steps, args.max_steps)
    no_decay = ["bias", "LayerNorm.weight"]


    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if ( not (any(nd in n for nd in no_decay)) )],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if ( (any(nd in n for nd in no_decay)) )], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    # multi-gpu training
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True
        )

    loss_func = nn.CrossEntropyLoss()
    kd_loss_func = nn.KLDivLoss(reduction="batchmean")
    global_step, sum, correct, best_acc = 0, 0, 0, 0.0
    tr_loss, logging_loss = 0.0, 0.0
    def backward_loss(loss, tot_loss):
        if args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training
        if args.gradient_accumulation_steps > 1:
            loss = loss / args.gradient_accumulation_steps

        tot_loss += loss.item()
        loss.backward()
        return tot_loss

    tqdm_iterator = trange(int(t_total), desc="Iteration", disable=args.local_rank not in [-1, 0])
    for epoch in range(args.num_epoch):
        for step, train_batch in enumerate(train_dataloader):
            model.train()
            source = train_batch["source"]
            permuted_texts = train_batch["permuted_texts"]
            # [(a1, b1, ...), (a2, b2, ...)]
            permuted_texts_list = [list(tup) for tup in permuted_texts]
            batch_labels = train_batch["labels"].to(args.device)
            batch_teacher_labels = train_batch["teacher_labels"].to(args.device)

            batch_preds = model(permuted_texts_list, source, args).to(args.device)
            correct += torch.sum(torch.max(batch_preds,dim=-1)[-1]==batch_labels).item()
            sum += args.batch_size
            
            ce_loss = loss_func(batch_preds, batch_labels)
            
            kd_loss = kd_loss_func(F.log_softmax(batch_preds, dim=-1), F.softmax(batch_teacher_labels, dim=-1))
            
            loss = args.beta * ce_loss + args.alpha * kd_loss
            # loss.backward()
            tr_loss = backward_loss(loss, tr_loss)
            if (step+1)%args.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1
                tqdm_iterator.update()
                if global_step % args.logging_steps == 0 and args.local_rank in [-1, 0]:
                    acc = correct / sum
                    logger.info("Step: %d, loss: %.6f, ce_loss: %.6f, kd_loss: %.6f acc: %.6f" % (global_step, (tr_loss-logging_loss)/args.logging_steps, args.beta*ce_loss, args.alpha*kd_loss, acc))
                    logging_loss = tr_loss
                    correct, sum = 0, 0
                if global_step % args.logging_steps == 0 and args.local_rank in [-1, 0]:
                    logger.info("Prediction: ")
                    logger.info(batch_preds)
                    logger.info(torch.max(batch_preds, dim=-1))
                    logger.info("Label: ")
                    logger.info(batch_labels)

                    if acc > best_acc:
                        best_acc = acc
                        output_dir = args.output_dir
                        f_res = open(os.path.join(output_dir, 'result.txt'), 'a')
                        f_res.write(str(global_step)+"-acc-"+str(acc))
                        f_res.write('\n')
                        f_res.write("Step: %d, loss: %.6f, acc: %.6f" % (global_step, loss, acc))
                        f_res.write("\n")
                        f_res.close()
                        if args.local_rank == -1 or torch.distributed.get_rank() == 0:
                            model_to_save = (
                                model.module if hasattr(model, "module") else model.model
                            )  # Take care of distributed/parallel training
                            
                            logger.info("Saving best trained model checkpoint to %s", output_dir)
                            model_to_save.save_pretrained(output_dir)
                            tokenizer.save_pretrained(output_dir)

                            torch.save(args, os.path.join(output_dir, "training_args.bin"))
                            logger.info("Saving best model checkpoint to %s", output_dir)
                            torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_type", default='xlm-roberta-base'
    )
    parser.add_argument(
        "--teacher_model_type", default=None, type=str, help="Teacher model."
    )
    parser.add_argument(
        "--train_data_file", default='zhen.6000.src-tgt', type=str, required=True, help="THe input training data file."
    )
    parser.add_argument(
        "--permuted_file", default=None, type=str, required=True, help="The path to store permuted file."
    )
    parser.add_argument(
        "--output_dir", default=None, type=str, help="The path to save model checkpoints."
    )
    parser.add_argument(
        "--ratio", default=0.2, type=float, help="The ratio of pemuting words."
    )
    parser.add_argument(
        "--type", default="permuted", type=str, help="Augmented data type."
    )
    parser.add_argument(
        "--alpha", default=1, type=int, help="Weight of KD loss."
    )
    parser.add_argument(
        "--beta", default=1, type=int, help="Weight of CE loss."
    )
    parser.add_argument(
        "--permute_num", default=8, type=int, help="The number of permuted sentences."
    )
    parser.add_argument(
        "--batch_size", default=32, type=int, help="Batch size."
    )
    parser.add_argument(
        "--teacher_batch_size", default=1000, type=int, help="Batch size of teacher model."
    )
    parser.add_argument(
        "--num_epoch", default=1, type=int, help="The number of epoch."
    )
    parser.add_argument(
        "--max_steps", default=10000, type=int
    )
    parser.add_argument(
        "--weight_decay", default=0.0, type=float,
    )
    parser.add_argument(
        "--learning_rate", default=1e-5, type=float, help="Learning rate."
    )
    parser.add_argument(
        "--warmup_steps", default=1000, type=int, help="Warmup steps."
    )
    parser.add_argument(
        "--adam_epsilon", default=1e-8, type=float, help="adam epsilon."
    )
    parser.add_argument(
        "--local_rank", default=-1, type=int, help="node rank for distributed training."
    )
    parser.add_argument(
        "--gradient_accumulation_steps", default=1, type=int, help="accumulation steps."
    )
    parser.add_argument(
        "--logging_steps", default=10, type=int, help="Logging steps."
    )
    parser.add_argument(
        "--seed", type=int, default=42
    )

    args = parser.parse_args()
    set_seed(args.seed)

    # For distributed training.
    if args.local_rank == -1:
        device = torch.device("cuda")
        args.n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device
    

    logger.info(args)
    # For distributed training.
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1)
    )

    if os.path.exists(args.output_dir) == True:
        for fix in range(1000):
            if os.path.exists(args.output_dir + str(fix)) == False:
                args.output_dir = args.output_dir + str(fix)
                os.makedirs(args.output_dir)
                f_res = open(os.path.join(args.output_dir, 'result.txt'), 'a')
                for para, val in vars(args).items():
                    f_res.write(para+': '+str(val)+'\n')
                f_res.write('\n')
                f_res.close()
                break
            else:
                continue
    generate = False
    if generate:
        import pdb; pdb.set_trace()
        f_text = open(args.train_data_file).readlines()
        a_textlines, b_textlines = [], []
        for line in f_text:
            a, b = line.strip().split(' ||| ')
            if len(a.split()) == 1:
                continue
            a_textlines.append(a)
            b_textlines.append(b)
        
        a_text, b_text = swap_text_a_b(a_textlines, b_textlines)
        generate_permuted_sents(a_text, args.permuted_file, args.permute_num, args.ratio)
        
        all_permuted_texts, all_labels = shuffle_sents(a_text, args.permute_num, args.permuted_file)
    else:
        if args.type == "del" or args.type == "shuf_whole":
            f_source = open(os.path.join("data1/", 'all_source_2.txt')).readlines()
        else:
            f_source = open(os.path.join("data2/", 'all_source.txt')).readlines()

        b_text = [line.strip() for line in f_source]
        all_permuted_texts = []
        for i in range(args.permute_num+1):
            f_text = open(os.path.join(args.permuted_file, 'all_'+args.type+'_texts_'+str(i)+'.txt')).readlines()
            permuted_texts = [line.strip() for line in f_text]
            print(i)
            assert len(f_source) == len(f_text)
            assert len(permuted_texts) == len(b_text)
            all_permuted_texts.append(permuted_texts)
        all_labels = torch.load(os.path.join(args.permuted_file, 'label_tensor.pt'))


    # For distributed training.
    # Load pretrained model and tokenizer
    # Barrier to make sure only the first process in distributed training download the model & vocab.
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    model = BertScore_Model(args).to(args.device)
    tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base')

    # End of barrier to make sure only the first process in distributed training download the model & vocab.
    if args.local_rank == 0:
        torch.distributed.barrier()


    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    logger.info("------------------------Sent Model------------------------")
    teacher_labels = torch.load(os.path.join(args.permuted_file, 'sent_trans_score.pt'))
    train_dataset = dataset(b_text, all_permuted_texts, all_labels, teacher_labels)

    if args.local_rank == 0:
        torch.distributed.barrier()
    train(
        model = model,
        tokenizer = tokenizer,
        train_dataset = train_dataset,
        # all_source = b_text,
        # all_permuted_texts=all_permuted_texts, 
        # all_labels=all_labels, 
        args=args
    )


if __name__ == '__main__':
    main()