import torch
from torch.utils.data import Dataset

import nltk
from nltk.tokenize import word_tokenize
nltk.download('punkt')

class CustomerDataset(Dataset):
    
    def __init__(self, samples, sentences, summary, num_mention, num_candidate, 
                 max_length_seq_char, max_length_seq_sence, max_length_word,
                 char2id, word2id, candidate_prob, candidate_elsearch, device='cpu'):
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
        index_mentions += [[0] * (self.max_length_seq_char * self.max_length_word)] * (self.num_mention - len(mentions))

        index_mentions_word = self.mention2word(mentions, self.max_length_word)
        index_mentions_word += [[0] * self.max_length_word] * (self.num_mention - len(mentions))

        # generate candidate from mention, mask denote addtional padding
        # candidates shape: (num_mention, num_candidate)
        # masks shape: (num_mention, num_candidate)
        # idx_entities shape: (num_mention, 1)
        # index_candidates shape: (num_mention, num_candidate, max_num_words, max_length_char)
        candidates, mask_candidates, idx_entities = self.generate_candidate(mentions, entities, self.num_candidate)
        index_candidates = self.word2chars_candidate(candidates, mask_candidates, self.max_length_word, self.max_length_seq_char)
        index_candidates_word = self.candidate2word(candidates, mask_candidates, self.max_length_word)
        # index_candidates = torch.Tensor(index_candidates)

        # Padding mask candidates
        mask_candidates += [[False] * self.num_candidate] * (self.num_mention - len(mentions))

        # Padding idx_entities
        idx_entities += [-1] * (self.num_mention - len(mentions))

        # create vector candidate padding
        # mention_padding = torch.zeros((self.num_mention - len(mentions), self.num_candidate, self.max_length_word, self.max_length_seq_char))
        # index_candidates = torch.cat((index_candidates, mention_padding), dim=0)
        index_candidates += [[[0] * (self.max_length_seq_char * self.max_length_word)] * self.num_candidate] * (self.num_mention -len(mentions))
        index_candidates_word += [[[0] * self.max_length_word] * self.num_candidate] * (self.num_mention - len(mentions))

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
        index_mentions_word = torch.tensor(index_mentions_word).to(self.device)
        index_candidates_word = torch.tensor(index_candidates_word).to(self.device)

        # print(index_mentions_word.shape)
        # print(index_candidates_word.shape)

        return index_candidates, index_mentions, mask_mentions, mask_candidates, index_sentence, index_summary, start, end, idx_entities, index_mentions_word, index_candidates_word

    def word2chars_mention(self, mentions, max_length_word, max_length_seq_char):
        index_chars = []

        for mention in mentions:
            mention = mention.lower()
            id_chars = [self.char2id[c] if c in self.char2id else self.char2id['<unk>'] for c in list(mention)[:max_length_seq_char * max_length_word]]

            id_chars += [0] * ((max_length_seq_char * max_length_word) - len(id_chars))
            index_chars.append(id_chars)

        return index_chars
    
    def word2chars_candidate(self, candidates, masks, max_length_word, max_length_seq_char):
        index_chars = []
        for candidate, mask in zip(candidates, masks):
            index_char = []
            for cand, msk in zip(candidate, mask):
                if msk:
                    cand = cand.lower()
                    id_chars = [self.char2id[c] if c in self.char2id else self.char2id['<unk>'] for c in list(cand)[:max_length_seq_char * max_length_word]]
                    id_chars += [0] * (max_length_seq_char * max_length_word - len(id_chars))
                    index_char.append(id_chars)
                else:
                    id_chars = [0] * (max_length_seq_char * max_length_word)
                    index_char.append(id_chars)
            
            index_chars.append(index_char)
        
        return index_chars
    
    def mention2word(self, mentions, max_length_word):
        index_words = []
        for mention in mentions:
            mention = mention.lower()
            mention = word_tokenize(mention)[:max_length_word]

            id_word = [self.word2id[w] if w in self.word2id else self.word2id['<unk>'] for w in mention]
            id_word += [0] * (max_length_word - len(id_word))
        
            index_words.append(id_word)
        
        # mask_mention_word = [[tok == 0 for tok in men] for men in index_words]

        return index_words
    
    def candidate2word(self, candidates, masks, max_length_word):
        index_words = []
        for candidate, mask in zip(candidates, masks):
            index_word = []
            for cand, msk in zip(candidate, mask):
                if msk:
                    cand = cand.lower()
                    cand = word_tokenize(cand)[:max_length_word]
                    id_word = [self.word2id[w] if w in self.word2id else self.word2id['<unk>'] for w in cand]
                    id_word += [0] * (max_length_word - len(id_word))
                    index_word.append(id_word)
                else:
                    id_word = [0] * max_length_word
                    index_word.append(id_word)
            
            index_words.append(index_word)
        
        # mask_candidate_word = [[[c == 0 for c in cand] for cand in men] for men in index_words]
        
        return index_words

    def sence2word(self, sentence_id, max_length_seq_sence):

        # print(self.sentences[sentence_id])
        sentence = self.sentences[sentence_id]
        sentence = sentence.lower()
        sentence = word_tokenize(sentence)[:max_length_seq_sence-2]


        id_sence = [self.word2id['<s>']] + [self.word2id[s] if s in self.word2id else self.word2id['<unk>'] for s in sentence] + [self.word2id['</s>']]
        id_sence += [0] * (max_length_seq_sence - len(id_sence))

        # print(id_sence)
        return id_sence
    
    def summary2word(self, doc_id, max_length_seq_sence):
        
        # print(self.summary[doc_id])
        summary = self.summary[doc_id]
        summary = summary.lower()
        summary = word_tokenize(summary)[:max_length_seq_sence-2]


        id_sence = [self.word2id['<s>']] + [self.word2id[s] if s in self.word2id else self.word2id['<unk>'] for s in summary] + [self.word2id['</s>']]
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
                cand_elsearch = []

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
                candidate.append(entity)     

            pad = ['<pad>'] * (k - len(candidate))
            candidate = candidate + pad
            mask = [c != '<pad>' for c in candidate]

            idx_entity = candidate.index(entity)

            candidates.append(candidate)
            masks.append(mask)
            idx_entities.append(idx_entity)
        
        return candidates, masks, idx_entities