def generate_candidate(mentions, k=10):

    candidates = []
    masks = []
    idx_entities = []
    for i, mention in enumerate(mentions):
        ## Get candidate from elasticsearch and probility
        candidate = []
        mention = mentions[i].lower()

        ## Get candidate:
        if mention in self.candidate_prob:
            cand_prob = [e[1] for e in self.candidate_prob[mention]]
        else:
            cand_prob = []
        
        if mention in self.candidate_elsearch:
            cand_elsearch = [e[1] for e in self.candidate_elsearch[mention]]
        else:
            if self.search is not None:
                cand_elsearch = [e for e in self.search_candidate(mention)][:k-1]
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
  
        pad = ['<pad>'] * (k - len(candidate))
        candidate = candidate + pad
        mask = [c != '<pad>' for c in candidate]

        candidates.append(candidate)
        masks.append(mask)
        
    return candidates, masks

def search_candidate(m):
    candidate = set()
    str_search = re.sub(r'[\/\\~\!\@\#\$\%\^\&\*\(\)\-\+\[\]\"\'\:\`\{\}]', ' ', m)
    tmp = search.query('query_string', query=str_search, fields=['redirects^1','alias^1'])
    res = tmp.execute()
    for res_entity, res_hit in zip(res, res.hits):
        candidate.add(res_entity.entity)
    
    return candidate

def word2chars_mention(mentions):
    index_chars = []

    for mention in mentions:
        mention = mention.lower()
        id_chars = [self.char2id[c] if c in self.char2id else self.char2id['<unk>'] for c in list(mention)[:max_length_seq_char * max_length_word]]

        id_chars += [0] * ((max_length_seq_char * max_length_word) - len(id_chars))
        index_chars.append(id_chars)

    return index_chars

def word2chars_candidate(candidates, masks):
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
    
    def candidate2word(candidates, masks):
        index_words = []
        for candidate, mask in zip(candidates, masks):
            index_word = []
            for cand, msk in zip(candidate, mask):
                if msk:
                    cand = cand.lower()
                    cand = word_tokenize(cand)[:max_length_word]
                    id_word = [word2id[w] if w in word2id else word2id['<unk>'] for w in cand]
                    id_word += [0] * (max_length_word - len(id_word))
                    index_word.append(id_word)
                else:
                    id_word = [0] * max_length_word
                    index_word.append(id_word)
            
            index_words.append(index_word)
        
        # mask_candidate_word = [[[c == 0 for c in cand] for cand in men] for men in index_words]
        
        return index_words

    def sence2word(sentence):

        sentence = sentence.lower()
        sentence = word_tokenize(sentence)[:max_length_seq_sence-2]


        id_sence = [word2id['<s>']] + [self.word2id[s] if s in word2id else word2id['<unk>'] for s in sentence] + [word2id['</s>']]
        id_sence += [0] * (max_length_seq_sence - len(id_sence))

        return id_sence
    
    def summary2word(entity, title2id):
        
        entity = entity.lower()
        if entity in title2id:
            doc_id = title2id[entity]
            summary = summary[doc_id]
        else:
            summary = entity

        summary = summary.lower()
        summary = word_tokenize(summary)[:max_length_seq_sence-2]

        id_sence = [word2id['<s>']] + [word2id[s] if s in word2id else word2id['<unk>'] for s in summary] + [self.word2id['</s>']]
        id_sence += [0] * (max_length_seq_sence - len(id_sence))

        return id_sence

def inference(input, num_candidate):
    sentence = input['sentence']
    offset = input['offset']

    with open('../data/title2id.pkl', 'rb') as f:
        title2id = pickle.load(f)
    
    sentence_tok = word_tokenize(sentence)
    mentions = []

    for s, e in offset:
        mentions.append(sentence[s:e])
    
    mentions = mentions[:3]

    candidates, masks = generate_candidate(mentions, num_candidate)

    index_mentions = word2chars_mention(mentions, max_length_word, max_length_seq_char)
    index_mentions += [[0] * (max_length_seq_char * max_length_word)] * (num_mention - len(mentions))

    index_mentions_word = mention2word(mentions, max_length_word)
    index_mentions_word += [[0] * max_length_word] * (num_mention - len(mentions))