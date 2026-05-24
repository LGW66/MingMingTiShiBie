import torch
import torch.nn as nn
import torch.nn.functional as F


class CRF(nn.Module):
    def __init__(self, num_tags):
        super().__init__()
        self.num_tags = num_tags
        self.transitions = nn.Parameter(torch.randn(num_tags, num_tags))
        self.start_transitions = nn.Parameter(torch.randn(num_tags))
        self.end_transitions = nn.Parameter(torch.randn(num_tags))

    def forward(self, emissions, tags, mask=None):
        if mask is None:
            mask = torch.ones(emissions.shape[:2], dtype=torch.bool, device=emissions.device)
        
        numerator = self._compute_score(emissions, tags, mask)
        denominator = self._compute_normalizer(emissions, mask)
        return torch.sum(numerator - denominator)

    def decode(self, emissions, mask=None):
        if mask is None:
            mask = torch.ones(emissions.shape[:2], dtype=torch.bool, device=emissions.device)
        
        return self._viterbi_decode(emissions, mask)

    def _compute_score(self, emissions, tags, mask):
        batch_size, seq_len = tags.shape
        score = self.start_transitions[tags[:, 0]]
        
        for i in range(seq_len):
            score += emissions[torch.arange(batch_size), i, tags[:, i]] * mask[:, i]
            if i < seq_len - 1:
                score += self.transitions[tags[:, i], tags[:, i + 1]] * mask[:, i + 1]
        
        last_mask = mask.long().sum(dim=1) - 1
        last_tags = tags.gather(1, last_mask.unsqueeze(1)).squeeze(1)
        score += self.end_transitions[last_tags]
        
        return score

    def _compute_normalizer(self, emissions, mask):
        batch_size, seq_len, num_tags = emissions.shape
        alpha = self.start_transitions + emissions[:, 0]
        
        for i in range(1, seq_len):
            alpha = self._log_sum_exp(
                alpha.unsqueeze(2) + self.transitions.unsqueeze(0) + emissions[:, i].unsqueeze(1), 
                dim=1
            )
            alpha = torch.where(mask[:, i].unsqueeze(1), alpha, alpha)
        
        return self._log_sum_exp(alpha + self.end_transitions, dim=1)

    def _viterbi_decode(self, emissions, mask):
        batch_size, seq_len, num_tags = emissions.shape
        backpointers = []
        
        viterbi = self.start_transitions + emissions[:, 0]
        
        for i in range(1, seq_len):
            viterbi, backpointer = torch.max(viterbi.unsqueeze(2) + self.transitions.unsqueeze(0), dim=1)
            viterbi += emissions[:, i]
            backpointers.append(backpointer)
        
        viterbi += self.end_transitions
        best_tags = []
        
        for b in range(batch_size):
            best_tag = viterbi[b].argmax().item()
            best_tags.append([best_tag])
            
            for backpointer in reversed(backpointers):
                best_tag = backpointer[b, best_tag].item()
                best_tags[b].append(best_tag)
            
            best_tags[b].reverse()
            actual_len = int(mask[b].sum().item())
            best_tags[b] = best_tags[b][:actual_len]
        
        return best_tags

    def _log_sum_exp(self, x, dim):
        max_val, _ = torch.max(x, dim=dim)
        return max_val + torch.log(torch.sum(torch.exp(x - max_val.unsqueeze(dim)), dim=dim))


class BiLSTM_CRF(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_labels, num_layers=2, dropout=0.5):
        super(BiLSTM_CRF, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.num_labels = num_labels
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim // 2,
            num_layers=num_layers,
            bidirectional=True,
            dropout=dropout,
            batch_first=True
        )
        
        self.fc = nn.Linear(hidden_dim, num_labels)
        
        self.crf = CRF(num_labels)
    
    def forward(self, char_ids, mask=None):
        batch_size, seq_len = char_ids.shape
        
        embeds = self.embedding(char_ids)
        
        lstm_out, _ = self.lstm(embeds)
        
        emissions = self.fc(lstm_out)
        
        if mask is not None:
            emissions = emissions * mask.unsqueeze(-1)
        
        return emissions
    
    def compute_loss(self, emissions, tags, mask=None):
        return -self.crf(emissions, tags, mask=mask) / emissions.shape[0]
    
    def predict(self, char_ids, mask=None):
        emissions = self.forward(char_ids, mask)
        return self.crf.decode(emissions, mask=mask)
