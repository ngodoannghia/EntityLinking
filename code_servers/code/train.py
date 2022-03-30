from json import encoder
import pickle
import random

import torch
import torch.nn as nn
from EL_Dataset import CustomerDataset
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import torch.nn.functional as F

from lstm import *
from mlp import *
from loss_margin import MyMarginLoss
from encoder import Encoder
from model_el import Entity_Linking
import torch.optim as optim

import os

def load_data():

    ## Load word2id and char2id
    with open('../data/word2id.pkl', 'rb') as f:
        word2id = pickle.load(f)   
    with open('../data/char2id.pkl', 'rb') as f:
        char2id = pickle.load(f) 

    ## Load sentences
    with open('../data/sentences.pkl', 'rb') as f:
        sentences = pickle.load(f)

    ## Load candidate elasticsearch
    with open('../data/candidate_elasticsearch_alias_top20.pkl', 'rb') as f:
        candidate_elsearch = pickle.load(f)

    ## Load candidate prob 
    with open('../data/output_e_give_m.pkl', 'rb') as f:
        candidate_prob = pickle.load(f)

    ## Load samples
    with open('../data/samples_all.pkl', 'rb') as f:
        samples = pickle.load(f)

    ## Data test
    with open('../data/sample_test.pkl', 'rb') as f:
        samples_test = pickle.load(f)
    with open('../data/sentences_test.pkl', 'rb') as f:
        sentences_test = pickle.load(f)

    ## Load summary 
    with open('../data/summary.pkl', 'rb') as f:
        summary = pickle.load(f)

    return word2id, char2id, sentences, candidate_elsearch, candidate_prob, samples, samples_test, sentences_test, summary

def dataloader(sample_dataset, batch_size=64, train=True):
    
    if train:
        data = DataLoader(sample_dataset, batch_size=batch_size, shuffle=True)

    else:
        data = DataLoader(sample_dataset, batch_size=batch_size, shuffle=False)

    return data

def train():
    loss_epoch = 0
    predicts = []
    labels = []
    model.train(True)
    iters = 0
    for batch in tqdm(train_dataloader):
        index_candidates, index_mentions, mask_mentions, mask_candidates, index_sentence, index_summary, char_start, char_end, idx_entities = batch
        
        # print(type(index_sentence), type(index_summary))
        # print(index_sentence.shape, index_summary.shape)
        
        score_candidate = model(index_candidates, index_mentions, index_sentence, index_summary)
        score_candidate = F.softmax(score_candidate, dim=-1)

        optimizer.zero_grad()

        l = loss(score_candidate, idx_entities, mask_mentions, mask_candidates)
        l.backward()

        pred = torch.argmax(score_candidate, dim=2)
        pred = torch.masked_select(pred, mask_mentions).tolist()
        label = torch.masked_select(idx_entities, mask_mentions).tolist()

        nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        loss_epoch += l.item()
        
        predicts.extend(pred)
        labels.extend(label)

        if iters % 1000 == 0:
            print(f"Loss train: {loss_epoch}, Acc train: {accuracy_score(labels, predicts)}")

        iters += 1
    acc = accuracy_score(labels, predicts)

    return loss_epoch, acc

def evaluate(data_loaders):
    with torch.no_grad():
        loss_eval = 0
        predict = []
        labels = []
        for batch in tqdm(data_loaders):
            index_candidates, index_mentions, mask_mentions, mask_candidates, index_sentence, index_summary, char_start, char_end, idx_entities = batch

            model.eval()
            score_candidate = model(index_candidates, index_mentions, index_sentence, index_summary)
            score_candidate = F.softmax(score_candidate, dim=-1)
            loss_eval += loss(score_candidate, idx_entities, mask_mentions, mask_candidates)

            pred = torch.argmax(score_candidate, dim=2)
            pred = torch.masked_select(pred, mask_mentions).tolist()
            label = torch.masked_select(idx_entities, mask_mentions).tolist()

            predict.extend(pred)
            labels.extend(label)

        acc = accuracy_score(labels, predict)

    return loss_eval, acc

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        # m.bias.data.fill_(0.01)

if __name__=='__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    num_mentions = 3
    num_candidates = 15
    max_length_seq_char = 10
    max_length_seq_sence = 50
    max_length_word = 12

    batch_size, epoch = 32, 20
    output_dim_word = 128 ## dim of ouput mention, candidate when that pass averge_mention

    ## argument encoder
    d_model, n_layers, n_heads, d_ff, clip, pad_idx = 256, 6, 4, 512, 1.0, 0
    output_encoder = 128
    dropout_rate = 0.2

    ## argument lstm
    input_dim_lstm, hidden_size, bidirection, num_layers = 300, 50, True, 2 

    ## argument mlp embed
    input_dim_mlp_embed, output_dim_mlp_embed, num_hidden_mlp_embed = max_length_word * output_dim_word * 2, 128, 256

    ## argument mlp score
    input_dim_mlp_score, output_dim_mlp_score, num_hidden_mlp_score1, num_hidden_mlp_score2 = output_encoder*2 + output_dim_mlp_embed + max_length_word * output_dim_word * 3 + 1, 1, 256, 10

    word2id, char2id, sentences, candidate_elsearch, candidate_prob, samples, samples_test, sentences_test, summary = load_data()
    vocab_chars = len(char2id)
    vocab_words = len(word2id)
    dim_char = hidden_size * 2

    encoder = Encoder(vocab_words, d_model, n_layers, n_heads, d_ff, pad_idx, dropout_rate, output_encoder, max_length_seq_sence)
    char_embed = EmbedCharLayer(vocab_chars, max_length_seq_char, input_dim_lstm, hidden_size, num_layers, device, dropout_rate, bidirection)
    sentence_embed = EmbedSentenceLayer(vocab_words, input_dim_lstm, hidden_size, num_layers, device, dropout_rate, bidirection)
    mlp_embed = MLPEmbedingLayer(input_dim_mlp_embed, output_dim_mlp_embed, num_hidden_mlp_embed, dropout_rate)
    mlp_score = MLPScoreLayer(input_dim_mlp_score, output_dim_mlp_score, num_hidden_mlp_score1, num_hidden_mlp_score2, dropout_rate)

    random.seed(100)
    random.shuffle(samples)
    train_samples = samples[:10000]
    valid_samples = samples[10000:15000]

    train_dataset = CustomerDataset(train_samples, sentences, summary, num_mentions, num_candidates, max_length_seq_char, max_length_seq_sence, max_length_word, char2id, word2id, candidate_prob, candidate_elsearch, device)
    valid_dataset = CustomerDataset(valid_samples, sentences, summary, num_mentions, num_candidates, max_length_seq_char, max_length_seq_sence, max_length_word, char2id, word2id, candidate_prob, candidate_elsearch, device)
    test_dataset = CustomerDataset(samples_test, sentences_test, summary, num_mentions, num_candidates, max_length_seq_char, max_length_seq_sence, max_length_word, char2id, word2id, candidate_prob, candidate_elsearch, device)

    train_dataloader = dataloader(train_dataset, batch_size, train=True)
    valid_dataloader = dataloader(valid_dataset, batch_size, train=False)
    test_dataloader = dataloader(test_dataset, batch_size=2, train=False)

    model = Entity_Linking(encoder, char_embed, sentence_embed, mlp_embed, mlp_score, num_mentions, num_candidates, max_length_seq_char, max_length_seq_sence, max_length_word, dim_char, output_dim_word, device)
    loss = MyMarginLoss()
    optimizer =  torch.optim.Adam(model.parameters(), lr=1e-3,  betas=(0.9, 0.98), eps=1e-9, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)


    model.apply(init_weights)
    model.to(device)


    # path_save_model = '../data/model_el_10k.pt'
    # if os.path.exists(path_save_model):
    #     model.load_state_dict(torch.load(path_save_model))
    #     loss_valid, acc_valid = evaluate(valid_dataloader)
    #     print("Loss valid: ", loss_valid, " Acc Valid: ", acc_valid)
    #     best = acc_valid
    # else:
    #     best = -10000

    # best = -1000

    for e in range(epoch):
        loss_train, acc_train = train()
        loss_valid, acc_valid = evaluate(valid_dataloader)
        loss_test, acc_test = evaluate(test_dataloader)

        # if acc_valid > best:
        #     best = acc_valid
        #     torch.save(model.state_dict(), path_save_model)
        
        scheduler.step()
        print("Learning rate = ", optimizer.param_groups[0]['lr'])

        print("Epoch: ", e)
        print(f"Loss train = {loss_train}, Loss valid = {loss_valid}, Loss test = {loss_test}")
        print(f"Accuracy train = {acc_train}, Accuracy valid = {acc_valid}, Accuracy test = {acc_test}")

    


