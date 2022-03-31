import torch
import torch.nn as nn


class Entity_Linking(nn.Module):
    def __init__(self, encoder,char_embed, sentence_embed, mlp_embed, mlp_score, 
                 num_mention, num_candidate, max_length_seq_char, max_length_seq_sence, 
                 max_length_word, dim_char, output_dim_word, device='cpu'):
        super(Entity_Linking, self).__init__()
        self.encoder = encoder
        self.num_mention = num_mention
        self.num_candidate = num_candidate
        self.max_length_seq_char = max_length_seq_char
        self.max_length_seq_sence = max_length_seq_sence
        self.max_length_word = max_length_word
        self.char_embed = char_embed
        self.sentence_embed = sentence_embed
        self.mlp_embed = mlp_embed
        self.mlp_score = mlp_score
        self.dim_char = dim_char
        self.output_dim_word = output_dim_word 
        self.device = device
        self.cosin = nn.CosineSimilarity(dim=3, eps=1e-6)
        self.averge_mention = nn.Sequential(
            nn.Linear(max_length_seq_char * dim_char, 1024),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),
            nn.Linear(1024, output_dim_word)
        )
    
    def forward(self, index_candidates, index_mentions, index_sentence, index_summary):

        ## Shape: (batch_size * num_mention * num_candidate * max_length_word, max_length_seq_char, dim_char)
        batch_size = index_candidates.shape[0]

        feature_candidates = self.char_embed(index_candidates)
        feature_mentions = self.char_embed(index_mentions)
        feature_sentence = self.encoder(index_sentence)
        feature_summary = self.encoder(index_summary)

        # feature_candidates = self.averge_embed(feature_candidates, dim=1)
        # feature_mentions = self.averge_embed(feature_mentions, dim=1)

        feature_candidates = feature_candidates.reshape(batch_size, self.num_mention, self.num_candidate, self.max_length_word, -1)
        feature_mentions = feature_mentions.reshape(batch_size, self.num_mention, self.max_length_word, -1)

        feature_candidates = self.averge_mention(feature_candidates)
        feature_mentions = self.averge_mention(feature_mentions)

        feature_candidates = feature_candidates.reshape(batch_size, self.num_mention, self.num_candidate, -1)
        feature_mentions = feature_mentions.reshape(batch_size, self.num_mention, -1)

        ## Repeat ==> feature mention.shape = feature_candidates.shape
        feature_mentions = torch.unsqueeze(feature_mentions, 2)
        feature_mentions = feature_mentions.repeat(1, 1, self.num_candidate, 1)
        # print(feature_mentions.shape, '\t', feature_candidates.shape)

        ## Caculate cosin of vectors
        ### Shape: (batch_size, num_mention, num_candidate, 1)
        score_cosin = self.cosine_similar(feature_mentions, feature_candidates, self.cosin)
        score_cosin = score_cosin.reshape(-1, self.num_mention, self.num_candidate, 1)

        ## Pass MLPEmbedLayer()
        ### Shape: (batch_size, num_mention, num_candidate, output_dim)
        feature_mlp = self.mlp_embed(torch.cat((feature_mentions, feature_candidates), dim=-1))
        ### Shape: (batch_size, num_mention, output_dim)
        # feature_mentions = self.mlp_embed(feature_mentions)

        ## unsqueeze dim feature_mention
        ## Shape: (batch_size, num_mention, num_candidate, dim_cand=800)
        # feature_mentions = torch.unsqueeze(feature_mentions, 2)
        # feature_mentions = feature_mentions.repeat(1, 1, self.num_candidate, 1)

        ## Feature dot product
        feature_dot_product = feature_mentions * feature_candidates

        ## Feature sum
        feature_sum = feature_mentions + feature_candidates

        ## Feature sub
        feature_sub = feature_mentions - feature_candidates
        
        ## Repeat feature_sentence
        feature_sentence = torch.unsqueeze(feature_sentence, 1)
        feature_sentence = torch.unsqueeze(feature_sentence, 2)
        feature_sentence = feature_sentence.repeat(1, self.num_mention, self.num_candidate, 1)

        ## Repeat feature summary
        feature_summary = torch.unsqueeze(feature_summary, 1)
        feature_summary = torch.unsqueeze(feature_summary, 2)
        feature_summary = feature_summary.repeat(1, self.num_mention, self.num_candidate, 1)
        # print(feature_sentence.shape)
        ## Concat score_cosin, feature_mentions. feature_candidates
        ### Shape: (batch_size, num_mention, num_candidate, dim=201)
        features = torch.cat((feature_sentence, feature_mlp, feature_dot_product, feature_sum, feature_sub, score_cosin, feature_summary), dim=-1)
        # print("Feature Shape:", features.shape)

        ## Pass MLPScore
        ### Shape: batch_size, num_mention, num_candidate, 1
        score_candidate = self.mlp_score(features)
        score_candidate = score_candidate.reshape(batch_size, self.num_mention, -1)

        return score_candidate
    
    def cosine_similar(self, feature_mentions, feature_candidates, cosin):

        score = cosin(feature_mentions, feature_candidates)
        return score

    def averge_embed(self, x, dim=0, keepdim=True):
        return torch.mean(x, dim=dim, keepdim=keepdim)

class Entity_Linking_Base(nn.Module):
    def __init__(self, char_embed, sentence_embed, mlp_embed, mlp_score, num_mention,
                num_candidate, max_length_seq_char, max_length_seq_sence, 
                max_length_word, dim_char, device='cpu'):
        super(Entity_Linking_Base, self).__init__()
        self.char_embed = char_embed
        self.sentence_embed = sentence_embed
        self.mlp_embed = mlp_embed
        self.mlp_score = mlp_score
        self.num_candidate = num_candidate
        self.num_mention = num_mention
        self.max_length_seq_char = max_length_seq_char
        self.max_length_seq_sence = max_length_seq_sence
        self.max_length_word = max_length_word
        self.cosin = nn.CosineSimilarity(dim=3, eps=1e-6)
        # self.averge_mention = nn.Linear(max_length_seq_char * dim_char, output_dim_word)
    
    def forward(self, index_candidates, index_mentions, index_sentence):
        ## Shape: (batch_size * num_mention * num_candidate * max_length_word, max_length_seq_char, dim_char)
        batch_size = index_sentence.shape[0]

        feature_candidates = self.char_embed(index_candidates)
        feature_mentions = self.char_embed(index_mentions)
        feature_sentence = self.sentence_embed(index_sentence)

        feature_candidates = self.averge_embed(feature_candidates, dim=-2)
        feature_mentions = self.averge_embed(feature_mentions, dim=-2)

        feature_candidates = feature_candidates.reshape(batch_size, self.num_mention, self.num_candidate, -1)
        feature_mentions = feature_mentions.reshape(batch_size, self.num_mention, -1)
        
        ## Repeat ==> feature mention.shape = feature_candidates.shape
        feature_mentions = torch.unsqueeze(feature_mentions, -2)

        feature_mentions = feature_mentions.repeat(1, 1, self.num_candidate, 1)
        # feature_mentions = feature_mentions.repeat(1, 1, self.num_candidate, 1)
        ## Caculate cosin of vectors
        ### Shape: (batch_size, num_mention, num_candidate, 1)
        score_cosin = self.cosine_similar(feature_mentions, feature_candidates, self.cosin)
        score_cosin = score_cosin.reshape(-1, self.num_mention, self.num_candidate, 1)

        feature_mlp = self.mlp_embed(torch.cat((feature_mentions, feature_candidates), dim=-1))
        ## Pass MLPEmbedLayer()
        ### Shape: (batch_size, num_mention, num_candidate, output_dim)
        # feature_candidates = self.mlp_embed(feature_candidates)
        ### Shape: (batch_size, num_mention, output_dim)
        # feature_mentions = self.mlp_embed(feature_mentions)

        ## unsqueeze dim feature_mention
        ## Shape: (batch_size, num_mention, num_candidate, dim_cand=800)
        # feature_mentions = torch.unsqueeze(feature_mentions, 2)
        # feature_mentions = feature_mentions.repeat(1, 1, self.num_candidate, 1)
        
        ## Repeat feature_sentence
        feature_sentence = torch.unsqueeze(feature_sentence, 1)
        feature_sentence = torch.unsqueeze(feature_sentence, 2)
        feature_sentence = feature_sentence.repeat(1, self.num_mention, self.num_candidate, 1)


        ## Concat score_cosin, feature_mentions. feature_candidates
        ### Shape: (batch_size, num_mention, num_candidate, dim=201)
        features = torch.cat((feature_sentence, feature_mlp, score_cosin), dim=3)
        # print("Feature Shape:", features.shape)

        ## Pass MLPScore
        ### Shape: batch_size, num_mention, num_candidate, 1
        score_candidate = self.mlp_score(features)
        score_candidate = score_candidate.reshape(batch_size, self.num_mention, -1)

        return score_candidate
    
    def cosine_similar(self, feature_mentions, feature_candidates, cosin):

        # feature_mentions = torch.unsqueeze(feature_mentions, 2)
        # feature_mentions = feature_mentions.repeat(1, 1, self.num_candidate, 1)
        score = cosin(feature_mentions, feature_candidates)
        return score

    def averge_embed(self, x, dim=0, keepdim=True):
        return torch.mean(x, dim=dim, keepdim=keepdim)
        