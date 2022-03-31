import torch
from torch.utils.data import Dataset
import re

import nltk
from nltk.tokenize import word_tokenize
nltk.download('punkt')


client = Elasticsearch("http://localhost:9200")
s = Search(using=client, index='candidate').extra(size=20)

def load_data():

    ## Load word2id and char2id
    with open('../data/word2id.pkl', 'rb') as f:
        word2id = pickle.load(f)   
    with open('../data/char2id.pkl', 'rb') as f:
        char2id = pickle.load(f) 

    ## Load sentences
    # with open('../data/sentences.pkl', 'rb') as f:
    #     sentences = pickle.load(f)

    ## Load candidate elasticsearch
    with open('../data/candidate_elasticsearch_alias_top20.pkl', 'rb') as f:
        candidate_elsearch = pickle.load(f)

    ## Load candidate prob 
    with open('../data/output_e_give_m.pkl', 'rb') as f:
        candidate_prob = pickle.load(f)

    ## Load samples
    # with open('../data/samples_550000.pkl', 'rb') as f:
    #     samples = pickle.load(f)
    samples = None
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
        data = DataLoader(sample_dataset, batch_size=batch_size,shuffle=True)

    else:
        data = DataLoader(sample_dataset, batch_size=batch_size, shuffle=False)

    return data

def evaluate(data_loaders):
    with torch.no_grad():
        loss_eval = 0
        predict = []
        labels = []
        for batch in tqdm(data_loaders):
            index_candidates, index_mentions, mask_mentions, mask_candidates, index_sentence, index_summary, char_start, char_end, idx_entities = batch

            model.eval()
            score_candidate = model(index_candidates, index_mentions, index_sentence, index_summary)
            # print(score_candidate.shape)
            score_candidate = F.softmax(score_candidate, dim=-1)
            loss_eval += loss(score_candidate, idx_entities, mask_mentions, mask_candidates)

            # print(score_candidate.shape)
            # score_candidate.masked_fill(mask_candidates, 0)
            pred = torch.argmax(score_candidate, dim=-1)
            pred = torch.masked_select(pred, mask_mentions).tolist()
            label = torch.masked_select(idx_entities, mask_mentions).tolist()

            predict.extend(pred)
            labels.extend(label)
        print("Label: \n", labels)
        print("Predict: \n", predict)
        acc = accuracy_score(labels, predict)

    return loss_eval, acc

class CustomerDataset(Dataset):
    
    def __init__(self, samples, sentences, summary, num_mention, num_candidate, 
                 max_length_seq_char, max_length_seq_sence, max_length_word,
                 char2id, word2id, candidate_prob, candidate_elsearch, device='cpu', search=None, train=True):
        super().__init__()
        self.samples = samples
        self.sentences = sentences
        self.summary = summary
        self.num_mention = num_mention
        self.num_candidate = num_candidate
        self.max_length_seq_char = max_length_seq_char 
        self.max_length_seq_sence = max_length_seq_sence 
        self.max_length_word = max_length_word
        self.char2id = char2id 
        self.word2id = word2id
        self.candidate_prob = candidate_prob
        self.candidate_elsearch = candidate_elsearch
        self.search = search
        self.train = train
        self.device = device
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        
        # get data from index
        sample = self.samples[idx]
        mentions = sample['mention'][:self.num_mention]
        entities = sample['entity'][:self.num_mention]
        char_start = sample['char_start'][:self.num_mention]
        char_end = sample['char_end'][:self.num_mention]
        sentence_id = sample['id']
        doc_id = sample['id_doc']
        
        # Padding char_start, char_end
        start = char_start + [0] * (self.num_mention - len(mentions))
        end = char_end + [0] * (self.num_mention - len(mentions))

        # index mention shape: (num_mention, dim_char_x, dim_char_y)
        index_mentions = self.word2chars_mention(mentions, self.max_length_word, self.max_length_seq_char)
        index_mentions += [[[0] * self.max_length_seq_char] * self.max_length_word] * (self.num_mention - len(mentions))

        # generate candidate from mention, mask denote addtional padding
        # candidates shape: (num_mention, num_candidate)
        # masks shape: (num_mention, num_candidate)
        # idx_entities shape: (num_mention, 1)
        # index_candidates shape: (num_mention, num_candidate, max_num_words, max_length_char)
        candidates, mask_candidates, idx_entities = self.generate_candidate(mentions, entities, self.num_candidate)
        index_candidates = self.word2chars_candidate(candidates, mask_candidates, self.max_length_word, self.max_length_seq_char)
        # index_candidates = torch.Tensor(index_candidates)

        # Padding mask candidates
        mask_candidates += [[False] * self.num_candidate] * (self.num_mention - len(mentions))

        # Padding idx_entities
        idx_entities += [-1] * (self.num_mention - len(mentions))

        # create vector candidate padding
        # mention_padding = torch.zeros((self.num_mention - len(mentions), self.num_candidate, self.max_length_word, self.max_length_seq_char))
        # index_candidates = torch.cat((index_candidates, mention_padding), dim=0)
        index_candidates += [[[[0] * self.max_length_seq_char] * self.max_length_word] * self.num_candidate] * (self.num_mention -len(mentions))
        mask_mentions = [True if i < len(mentions) else False for i in range(self.num_mention)]
        

        # index sentences shape: (max_length_seq_sence)
        index_sentence = torch.tensor(self.sence2word(sentence_id, self.max_length_seq_sence)).to(self.device)
        index_summary = torch.tensor(self.summary2word(doc_id, self.max_length_seq_sence)).to(self.device)
        index_candidates = torch.tensor(index_candidates).to(self.device)
        index_mentions = torch.tensor(index_mentions).to(self.device)
        mask_mentions = torch.tensor(mask_mentions).to(self.device)
        mask_candidates = torch.tensor(mask_candidates).to(self.device)
        # index_sentence = torch.tensor(index_sentence).to(self.device)
        start = torch.tensor(start).to(self.device)
        end = torch.tensor(end).to(self.device)
        idx_entities = torch.tensor(idx_entities).to(self.device)

        return index_candidates, index_mentions, mask_mentions, mask_candidates, index_sentence, index_summary, start, end, idx_entities

    def word2chars_mention(self, mentions, max_length_word, max_length_seq_char):
        index_chars = []

        for mention in mentions:
            mention = mention.lower()
            mention = word_tokenize(mention)[:max_length_word]
            id_chars = [[self.char2id[c] if c in self.char2id else self.char2id['<unk>'] for c in list(w)[:max_length_seq_char]] for w in mention]


            ## Padding follow dim=1
            for i in range(len(id_chars)):
                id_chars[i] += [0] * (max_length_seq_char - len(id_chars[i]))
            
            ## Padding follow dim=0
            id_chars += [[0] * max_length_seq_char] * (max_length_word - len(id_chars))
            index_chars.append(id_chars)

        return index_chars
    
    def word2chars_candidate(self, candidates, masks, max_length_word, max_length_seq_char):
        index_chars = []
        for candidate, mask in zip(candidates, masks):
            index_char = []
            for cand, msk in zip(candidate, mask):
                if msk:
                    cand = cand.lower()
                    cand = word_tokenize(cand)[:max_length_word]
                    id_chars = [[self.char2id[c] if c in self.char2id else self.char2id['<unk>'] for c in list(w)[:max_length_seq_char]] for w in cand]
                    ## Padding follow dim=1
                    for i in range(len(id_chars)):
                        id_chars[i] += [0] * (max_length_seq_char - len(id_chars[i]))
                    
                    ## Padding follow dim=0
                    id_chars += [[0] * max_length_seq_char] * (max_length_word - len(id_chars))
                    index_char.append(id_chars)
                else:
                    id_chars = [[0] * max_length_seq_char] * max_length_word
                    index_char.append(id_chars)
            
            index_chars.append(index_char)
        
        return index_chars
    
    def mention2word(self, mentions, max_length_word):
        index_words = []
        for mention in mentions:
            mention = mention.lower()
            mention = word_tokenize(mention)[:max_length_word]

            id_word = [self.word2id[w] if w in self.word2id else self.word2id['unk'] for w in mention]
            id_word += [0] * (max_length_word - len(id_word))
        
            index_words.append(id_word)
        
        mask_mention_word = [[tok == 0 for tok in men] for men in index_words]

        return index_words, mask_mention_word
    
    def candidate2word(self, candidates, masks, max_length_word):
        index_words = []
        for candidate, mask in zip(candidates, masks):
            index_word = []
            for cand, msk in zip(candidate, mask):
                if msk:
                    cand = cand.lower()
                    cand = word_tokenize(cand)[:max_length_word]
                    id_word = [self.word2id[w] if w in self.word2id else self.word2id['unk'] for w in candidate]
                    id_word += [0] * (max_length_word - len(id_word))
                    index_word.append(id_word)
                else:
                    id_word = [0] * max_length_word
                    index_word.append(id_word)
            
            index_words.append(index_word)
         
        mask_candidate_word = [[[c == 0 for c in cand] for cand in men] for men in index_words]
        
        return index_words, mask_candidate_word

    def sence2word(self, sentence_id, max_length_seq_sence):

        # print(self.sentences[sentence_id])
        sentence = self.sentences[sentence_id]
        sentence = sentence.lower()
        sentence = word_tokenize(sentence)[:max_length_seq_sence]


        id_sence = [self.word2id[s] if s in self.word2id else self.word2id['<unk>'] for s in sentence]
        id_sence += [0] * (max_length_seq_sence - len(id_sence))

        # print(id_sence)
        return id_sence
    
    def summary2word(self, doc_id, max_length_seq_sence):
        
        # print(self.summary[doc_id])
        summary = self.summary[doc_id]
        summary = summary.lower()
        summary = word_tokenize(summary)[:max_length_seq_sence]


        id_sence = [self.word2id[s] if s in self.word2id else self.word2id['<unk>'] for s in summary]
        id_sence += [0] * (max_length_seq_sence - len(id_sence))

        # print(id_sence)
        return id_sence

    ### Generate candidate using available elasticsearch and probility
    def generate_candidate(self, mentions, entities, k=10):

        candidates = []
        masks = []
        idx_entities = []
        for i, mention in enumerate(mentions):
            ## Get candidate from elasticsearch and probility
            candidate = []
            mention = mentions[i].lower()
            entity = entities[i].lower()


            ## Get candidate:
            if mention in self.candidate_prob:
                cand_prob = [e[1] for e in self.candidate_prob[mention]]
            else:
                cand_prob = []
            
            if mention in self.candidate_elsearch:
                cand_elsearch = [e[1] for e in self.candidate_elsearch[mention]]
            else:
                if not self.train:
                    # print("Vao")
                    if self.search is not None:
                        cand_elsearch = [e for e in self.search_candidate(mention)][:k-1]
                    else:
                        cand_elsearch = []
                    
                    # print(cand_elsearch)
                else:
                    cand_elsearch = []
                
                # cand_elsearch = []

            ## Create candidate
            if len(cand_prob) + len(cand_elsearch) <= k:
                candidate.extend(cand_prob)
                candidate.extend(cand_elsearch)
                candidate = candidate[:k-1]
            elif len(cand_prob) >= k:
                candidate.extend(cand_prob[:k-1])
            else:
                candidate.extend(cand_prob)
                for cand in cand_elsearch:
                    if len(candidate) >= k - 1:
                        break
                    if cand not in candidate:
                        candidate.append(cand)


            candidate = list(set(candidate))
            ## Check entity in candidate, if not in, ==> append entity into candidate ==> label
            if entity not in candidate:
                # print("Not in")
                candidate.append(entity)     

            pad = ['<pad>'] * (k - len(candidate))
            candidate = candidate + pad
            mask = [c != '<pad>' for c in candidate]

            idx_entity = candidate.index(entity)

            candidates.append(candidate)
            masks.append(mask)
            idx_entities.append(idx_entity)

            # print(candidates)
        
        return candidates, masks, idx_entities
    def search_candidate(self, m):
        candidate = set()
        str_search = re.sub(r'[\/\\~\!\@\#\$\%\^\&\*\(\)\-\+\[\]\"\'\:\`\{\}]', ' ', m)
        tmp = self.search.query('query_string', query=str_search, fields=['redirects^1','alias^1'])
        res = tmp.execute()
        for res_entity, res_hit in zip(res, res.hits):
            candidate.add(res_entity.entity)
        
        return candidate


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

num_mentions = 3
num_candidates = 15
max_length_seq_char = 56
max_length_seq_sence = 56
max_length_word = 10

batch_size, epoch = 32, 50
output_dim_word = 100 ## dim of ouput mention, candidate when that pass averge_mention

## argument encoder
d_model, n_layers, n_heads, d_ff, clip, pad_idx = 100, 2, 4, 512, 1.0, 0
dropout_rate = 0.2

## argument lstm
input_dim_lstm, hidden_size, bidirection, num_layers = 300, 50, True, 2 

## argument mlp embed
input_dim_mlp_embed, output_dim_mlp_embed, num_hidden_mlp_embed = max_length_word * (hidden_size*2)*2, 100, 256

## argument mlp score
input_dim_mlp_score, output_dim_mlp_score, num_hidden_mlp_score1, num_hidden_mlp_score2 = d_model*2 + output_dim_mlp_embed + max_length_word * output_dim_word * 3 + 1, 1, 256, 10

word2id, char2id, sentences, candidate_elsearch, candidate_prob, samples, samples_test, sentences_test, summary = load_data()
vocab_chars = len(char2id)
vocab_words = len(word2id)
dim_char = hidden_size * 2

encoder = Encoder(vocab_words, d_model, n_layers, n_heads, d_ff, pad_idx, dropout_rate)
char_embed = EmbedCharLayer(vocab_chars, max_length_seq_char, input_dim_lstm, hidden_size, num_layers, device, dropout_rate, bidirection)
sentence_embed = EmbedSentenceLayer(vocab_words, input_dim_lstm, hidden_size, num_layers, device, dropout_rate, bidirection)
mlp_embed = MLPEmbedingLayer(input_dim_mlp_embed, output_dim_mlp_embed, num_hidden_mlp_embed, dropout_rate)
mlp_score = MLPScoreLayer(input_dim_mlp_score, output_dim_mlp_score, num_hidden_mlp_score1, num_hidden_mlp_score2, dropout_rate)

Code cell <FeuEilW4nJZS>
#%% [code]
model = Entity_Linking(encoder, char_embed, sentence_embed, mlp_embed, mlp_score, num_mentions, num_candidates, max_length_seq_char, max_length_seq_sence, max_length_word, dim_char, output_dim_word, device)
loss = MyMarginLoss()
optimizer =  torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
model.to(device)

print("Load model ....")
model.load_state_dict(torch.load('../data/model_el_200k.pt'))
print("Complete load...")

Code cell <Eg2tRieAdRXl>
#%% [code]
test_dataset = CustomerDataset(samples_test, sentences_test, summary, num_mentions, num_candidates, max_length_seq_char, max_length_seq_sence, max_length_word, char2id, word2id, candidate_prob, candidate_elsearch, device, s, False)

test_dataloader = dataloader(test_dataset, batch_size=2, train=False)

loss_test, acc_test = evaluate(test_dataloader)
    
print(f"Accuracy test = {acc_test}, Loss Test = {loss_test}")