import torch
import torch.nn as nn


class Entity_Linking_CNN(nn.Module):
    def __init__(self, encoder, cnn_embed, mlp_embed, mlp_score, num_mention,
                num_candidate, max_length_seq_char, max_length_seq_sence, 
                max_length_word, device='cpu'):
        super(Entity_Linking_CNN, self).__init__()
        self.encoder = encoder
        self.cnn_embed = cnn_embed
        self.mlp_embed = mlp_embed
        self.mlp_score = mlp_score
        self.num_candidate = num_candidate
        self.num_mention = num_mention
        self.max_length_seq_char = max_length_seq_char
        self.max_length_seq_sence = max_length_seq_sence
        self.max_length_word = max_length_word
        self.cosin = nn.CosineSimilarity(dim=3, eps=1e-6)
        # self.averge_mention = nn.Linear(max_length_seq_char * dim_char, output_dim_word)
    
    def forward(self, index_candidates, index_mentions, index_sentence, index_summary, index_mentions_word, index_candidates_word):
        ## Shape: (batch_size * num_mention * num_candidate * max_length_word, max_length_seq_char, dim_char)
        batch_size = index_sentence.shape[0]

        feature_candidates = self.cnn_embed(index_candidates, index_candidates_word)
        feature_mentions = self.cnn_embed(index_mentions, index_mentions_word)
        feature_sentence = self.encoder(index_sentence)
        feature_summary = self.encoder(index_summary)


        # feature_candidates = self.averge_embed(feature_candidates, dim=-2)
        # feature_mentions = self.averge_embed(feature_mentions, dim=-2)

        feature_candidates = feature_candidates.reshape(batch_size, self.num_mention, self.num_candidate, -1)
        # print(feature_candidates.shape)
        feature_mentions = feature_mentions.reshape(batch_size, self.num_mention, -1)
        
        ## Repeat ==> feature mention.shape = feature_candidates.shape
        feature_mentions = torch.unsqueeze(feature_mentions, -2)

        feature_mentions = feature_mentions.repeat(1, 1, self.num_candidate, 1)
        
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

        ### Feature agument
        feature_dot = feature_mentions * feature_candidates
        # feature_sum = feature_mentions + feature_candidates
        feature_sub = feature_mentions - feature_candidates

        ## unsqueeze dim feature_mention
        ## Shape: (batch_size, num_mention, num_candidate, dim_cand=800)
        # feature_mentions = torch.unsqueeze(feature_mentions, 2)
        # feature_mentions = feature_mentions.repeat(1, 1, self.num_candidate, 1)
        
        ## Repeat feature_sentence
        feature_sentence = torch.unsqueeze(feature_sentence, 1)
        feature_sentence = torch.unsqueeze(feature_sentence, 2)
        feature_sentence = feature_sentence.repeat(1, self.num_mention, self.num_candidate, 1)

        ## Repeat feature summary
        feature_summary = torch.unsqueeze(feature_summary, 1)
        feature_summary = torch.unsqueeze(feature_summary, 2)
        feature_summary = feature_summary.repeat(1, self.num_mention, self.num_candidate, 1)


        ## Concat score_cosin, feature_mentions. feature_candidates
        ### Shape: (batch_size, num_mention, num_candidate, dim=201)
        features = torch.cat(( feature_dot, feature_sub, feature_sentence, feature_mlp, feature_summary, score_cosin), dim=-1)
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
        