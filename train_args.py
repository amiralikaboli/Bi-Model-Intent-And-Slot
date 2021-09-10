import argparse
from statistics import mean

from torch import optim
import numpy as np
import torch

import utils
from config import device
import config as cfg
from data2index_ver2 import train_data, test_data, index2slot_dict
from data2index_ver2 import intent_dict, slot_dict
from model import *
from tqdm import tqdm
from sklearn.metrics import classification_report, accuracy_score


def train_func(args):
    epoch_num = args.total_epoch

    slot_model = Slot(args).to(device)
    intent_model = Intent(args).to(device)

    # print(slot_model)
    # print(intent_model)

    slot_optimizer = optim.Adam(slot_model.parameters(), lr=args.learning_rate)       # optim.Adamax
    intent_optimizer = optim.Adam(intent_model.parameters(), lr=args.learning_rate)   # optim.Adamax

    # best_correct_num = 0
    best_intent_acc = 0
    best_epoch = -1
    best_F1_score = 0.0
    best_epoch_slot = -1
    for epoch in range(epoch_num):
        slot_loss_history = []
        intent_loss_history = []
        for batch_index, data in tqdm(enumerate(utils.get_batch(train_data, batch_size=args.batch))):
        # for batch_index, data in enumerate(utils.get_batch(train_data, batch_size=args.batch)):

            # Preparing data
            sentence, real_len, slot_label, intent_label = data

            mask = utils.make_mask(real_len, len(slot_dict), batch=args.batch).to(device)
            x = torch.tensor(sentence).to(device)
            y_slot = torch.tensor(slot_label).to(device)
            y_slot = utils.one_hot(y_slot, Num=len(slot_dict)).to(device)
            y_intent = torch.tensor(intent_label).to(device)
            y_intent = utils.one_hot(y_intent, Num=len(intent_dict)).to(device)

            # Calculate compute graph
            slot_optimizer.zero_grad()
            intent_optimizer.zero_grad()
            
            hs = slot_model.enc(x)
            slot_model.share_memory = hs.clone()

            hi = intent_model.enc(x)
            intent_model.share_memory = hi.clone()
            
            slot_logits = slot_model.dec(hs, intent_model.share_memory.detach())
            log_slot_logits = utils.masked_log_softmax(slot_logits, mask, dim=-1)
            slot_loss = -1.0*torch.sum(y_slot*log_slot_logits)
            slot_loss_history.append(slot_loss.item())
            slot_loss.backward()
            torch.nn.utils.clip_grad_norm_(slot_model.parameters(), 5.0)
            slot_optimizer.step()

            # Asynchronous training
            intent_logits = intent_model.dec(hi, slot_model.share_memory.detach(), real_len)
            log_intent_logits = F.log_softmax(intent_logits, dim=-1)
            intent_loss = -1.0*torch.sum(y_intent*log_intent_logits)
            intent_loss_history.append(intent_loss.item())
            intent_loss.backward()
            torch.nn.utils.clip_grad_norm_(intent_model.parameters(), 5.0)
            intent_optimizer.step()
            
        # Log
        # print('Slot loss: {:.4f} \t Intent loss: {:.4f}'.format(mean(slot_loss_history), mean(intent_loss_history)))

        # Evaluation 
        # total_test = len(test_data) // args.batch * args.batch
        correct_num = 0
        TP, FP, FN = 0, 0, 0
        all_intent_label, all_intent_pred = [], []
        # for batch_index, data_test in tqdm(enumerate(utils.get_batch(test_data, batch_size=1))):
        for batch_index, data_test in tqdm(enumerate(utils.get_batch(test_data, batch_size=args.batch))):
        # for batch_index, data_test in enumerate(utils.get_batch(test_data, batch_size=args.batch)):
            sentence_test, real_len_test, slot_label_test, intent_label_test = data_test
            # print(sentence[0].shape, real_len.shape, slot_label.shape)
            x_test = torch.tensor(sentence_test).to(device)

            # mask_test = utils.make_mask(real_len_test, len(slot_dict), batch=1).to(device)
            mask_test = utils.make_mask(real_len_test, len(slot_dict), batch=args.batch).to(device)
            # Slot model generate hs_test and intent model generate hi_test
            hs_test = slot_model.enc(x_test)
            hi_test = intent_model.enc(x_test)

            # Slot
            slot_logits_test = slot_model.dec(hs_test, hi_test)
            log_slot_logits_test = utils.masked_log_softmax(slot_logits_test, mask_test, dim=-1)
            slot_pred_test = torch.argmax(log_slot_logits_test, dim=-1)
            # Intent
            intent_logits_test = intent_model.dec(hi_test, hs_test, real_len_test)
            log_intent_logits_test = F.log_softmax(intent_logits_test, dim=-1)
            intent_pred_test = torch.argmax(log_intent_logits_test, dim=-1)

            all_intent_label.extend(list(intent_label_test))
            all_intent_pred.extend(list(intent_pred_test.cpu().detach().numpy()))
            correct_num += np.sum(intent_pred_test.cpu().detach().numpy() == intent_label_test)
            # if correct_num > best_correct_num:
            #     best_correct_num = correct_num
            #     best_epoch = epoch
                # Save and load the entire model.
                # torch.save(intent_model, 'model_intent_best_woz.ckpt')
                # torch.save(slot_model, 'model_slot_best_woz.ckpt')
            
            # if res_test.item() == intent_label_test[0]:
            #     correct_num += 1
            # if correct_num > best_correct_num:
            #     best_correct_num = correct_num
            #     best_epoch = epoch
            # 	# Save and load the entire model.
            #     torch.save(intent_model, 'model_intent_best_woz.ckpt')
            #     torch.save(slot_model, 'model_slot_best_woz.ckpt')
        
            # Calc slot F1 score

            # print(slot_pred_test)
            # print(slot_label_test)

            for spt, slt, rl in zip(slot_pred_test, slot_label_test, real_len_test):
                spt = spt[:rl]
                slt = slt[:rl]

                spt = [int(item) for item in spt]
                slt = [int(item) for item in slt]

                spt = [index2slot_dict[item] for item in spt]
                slt = [index2slot_dict[item] for item in slt]

                pred_chunks = utils.get_chunks(['O'] + spt + ['O'])
                label_chunks = utils.get_chunks(['O'] + slt + ['O'])
                for pred_chunk in pred_chunks:
                    if pred_chunk in label_chunks:
                        TP += 1
                    else:
                        FP += 1
                for label_chunk in label_chunks:
                    if label_chunk not in pred_chunks:
                        FN += 1

        intent_acc = accuracy_score(all_intent_label, all_intent_pred)
        if intent_acc > best_intent_acc:
            best_intent_acc = intent_acc
            best_epoch = epoch


        F1_score = 100.0 * 2 * TP / (2 * TP + FN + FP)
        if F1_score > best_F1_score:
            best_F1_score = F1_score
            best_epoch_slot = epoch
        # print('*' * 20)
        print('Epoch: [{}/{}], Intent Val Acc: {:.2f} \t Slot F1 score: {:.2f}'.format(epoch + 1, epoch_num, round(100.0 * intent_acc, 2), F1_score))
        # print('*' * 20)
        print(classification_report(all_intent_label, all_intent_pred, digits=4))
        
        print('Best Intent Acc: {:.2f} at Epoch: [{}]'.format(round(100.0 * best_intent_acc, 2), best_epoch + 1))
        print('Best F1 score: {:.2f} at Epoch: [{}]'.format(best_F1_score, best_epoch_slot + 1))
        print("*" * 50)
        # print('Epoch {}: {}'.format(epoch + 1, round(100.0 * intent_acc, 2)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--total_epoch', '-te', type=int, default=cfg.total_epoch)
    parser.add_argument('--batch', '-bb', type=int, default=cfg.batch)
    parser.add_argument("--learning_rate", '-lr', type=float, default=cfg.learning_rate)
    parser.add_argument('--DROPOUT', '-dr', type=float, default=cfg.DROPOUT)
    parser.add_argument('--embedding_size', '-es', type=int, default=cfg.embedding_size)
    parser.add_argument('--lstm_hidden_size', '-lhs', type=int, default=cfg.lstm_hidden_size)
    args = parser.parse_args()

    train_func(args)