import math
import torch
from torch import nn
from torch.nn import functional as F


class AsterHead(nn.Module):
    """
    ASTER: Attentional Scene Text Recognizer with Flexible Rectification
    https://github.com/ayumiymk/aster.pytorch/blob/master/lib/models/attention_recognition_head.py

    input: [b x 16 x 64 x in_planes]
    output: probability sequence: [b x T x num_classes]
    """
    def __init__(self, num_classes, in_planes, s_dim, att_dim, max_len_labels):
        super(AsterHead, self).__init__()
        self.num_classes = num_classes # this is the output classes. So it includes the <EOS>.
        self.in_planes = in_planes
        self.s_dim = s_dim
        self.att_dim = att_dim
        self.max_len_labels = max_len_labels

        self.decoder = DecoderUnit(s_dim=s_dim, x_dim=in_planes, y_dim=num_classes, att_dim=att_dim)

    def forward(self, x):
        x, targets, lengths = x
        batch_size = x.size(0)
        # Decoder
        state = torch.zeros(1, batch_size, self.s_dim)
        outputs = []

        for i in range(max(lengths)):
            if i == 0:
                y_prev = torch.zeros((batch_size)).fill_(self.num_classes) # the last one is used as the <BOS>.
            else:
                y_prev = targets[:,i-1]

            output, state = self.decoder(x, state, y_prev)
            outputs.append(output)
        outputs = torch.cat([_.unsqueeze(1) for _ in outputs], 1)
        return outputs

    def sample(self, x):    # inference stage.
        x, _, _ = x
        batch_size = x.size(0)

        # Decoder
        state = torch.zeros(1, batch_size, self.s_dim)

        predicted_ids, predicted_scores = [], []
        for i in range(self.max_len_labels):
            if i == 0:
                y_prev = torch.zeros((batch_size)).fill_(self.num_classes)
            else:
                y_prev = predicted

            output, state = self.decoder(x, state, y_prev)
            output = F.softmax(output, dim=1)
            score, predicted = output.max(1)
            predicted_ids.append(predicted.unsqueeze(1))
            predicted_scores.append(score.unsqueeze(1))
        predicted_ids = torch.cat(predicted_ids, 1)
        predicted_scores = torch.cat(predicted_scores, 1)

        return predicted_ids, predicted_scores

    def beam_search(self, x, beam_width, eos):

        def _inflate(tensor, times, dim):
            repeat_dims = [1] * tensor.dim()
            repeat_dims[dim] = times
            return tensor.repeat(*repeat_dims)

        # https://github.com/IBM/pytorch-seq2seq/blob/fede87655ddce6c94b38886089e05321dc9802af/seq2seq/models/TopKDecoder.py
        batch_size, l, d = x.size()

        # inflated_encoder_feats = _inflate(encoder_feats, beam_width, 0) # ABC --> AABBCC -/-> ABCABC
        inflated_encoder_feats = x.unsqueeze(1).permute((1,0,2,3)).repeat((beam_width,1,1,1)).permute((1,0,2,3)).contiguous().view(-1, l, d)

        # Initialize the decoder
        state = torch.zeros(1, batch_size * beam_width, self.s_dim)
        pos_index = (torch.Tensor(range(batch_size)) * beam_width).long().view(-1, 1)

        # Initialize the scores
        sequence_scores = torch.Tensor(batch_size * beam_width, 1)
        sequence_scores.fill_(-float('Inf'))
        sequence_scores.index_fill_(0, torch.Tensor([i * beam_width for i in range(0, batch_size)]).long(), 0.0)
        # sequence_scores.fill_(0.0)

        # Initialize the input vector
        y_prev = torch.zeros((batch_size * beam_width)).fill_(self.num_classes)

        # Store decisions for backtracking
        stored_scores          = list()
        stored_predecessors    = list()
        stored_emitted_symbols = list()

        for i in range(self.max_len_labels):
            output, state = self.decoder(inflated_encoder_feats, state, y_prev)
            log_softmax_output = F.log_softmax(output, dim=1)

            sequence_scores = _inflate(sequence_scores, self.num_classes, 1)
            sequence_scores += log_softmax_output
            scores, candidates = sequence_scores.view(batch_size, -1).topk(beam_width, dim=1)

            # Reshape input = (bk, 1) and sequence_scores = (bk, 1)
            y_prev = (candidates % self.num_classes).view(batch_size * beam_width)
            sequence_scores = scores.view(batch_size * beam_width, 1)

            # Update fields for next timestep
            predecessors = (candidates / self.num_classes + pos_index.expand_as(candidates)).view(batch_size * beam_width, 1)
            state = state.index_select(1, predecessors.squeeze())

            # Update sequence socres and erase scores for <eos> symbol so that they aren't expanded
            stored_scores.append(sequence_scores.clone())
            eos_indices = y_prev.view(-1, 1).eq(eos)
            if eos_indices.nonzero().dim() > 0:
                sequence_scores.masked_fill_(eos_indices, -float('inf'))

            # Cache results for backtracking
            stored_predecessors.append(predecessors)
            stored_emitted_symbols.append(y_prev)

        # Do backtracking to return the optimal values
        # Initialize return variables given different types
        p = list()
        l = [[self.max_len_labels] * beam_width for _ in range(batch_size)]  # Placeholder for lengths of top-k sequences

        # the last step output of the beams are not sorted
        # thus they are sorted here
        sorted_score, sorted_idx = stored_scores[-1].view(batch_size, beam_width).topk(beam_width)

        # initialize the sequence scores with the sorted last step beam scores
        s = sorted_score.clone()

        batch_eos_found = [0] * batch_size  # number of EOS found in backward loop below for each batch
        t = self.max_len_labels - 1
        # initialize the back pointer with the sorted order of the last step beams.
        # add pos_index for indexing variable with b*k as the first dimension.
        t_predecessors = (sorted_idx + pos_index.expand_as(sorted_idx)).view(batch_size * beam_width)
        while t >= 0:
            # Re-order the variables with the back pointer
            current_symbol = stored_emitted_symbols[t].index_select(0, t_predecessors)
            t_predecessors = stored_predecessors[t].index_select(0, t_predecessors).squeeze()
            eos_indices = stored_emitted_symbols[t].eq(eos).nonzero()
            if eos_indices.dim() > 0:
                for i in range(eos_indices.size(0)-1, -1, -1):
                    # Indices of the EOS symbol for both variables
                    # with b*k as the first dimension, and b, k for
                    # the first two dimensions
                    idx = eos_indices[i]
                    b_idx = int(idx[0] / beam_width)
                    # The indices of the replacing position
                    # according to the replacement strategy noted above
                    res_k_idx = beam_width - (batch_eos_found[b_idx] % beam_width) - 1
                    batch_eos_found[b_idx] += 1
                    res_idx = b_idx * beam_width + res_k_idx

                    # Replace the old information in return variables
                    # with the new ended sequence information
                    t_predecessors[res_idx] = stored_predecessors[t][idx[0]]
                    current_symbol[res_idx] = stored_emitted_symbols[t][idx[0]]
                    s[b_idx, res_k_idx] = stored_scores[t][idx[0], [0]]
                    l[b_idx][res_k_idx] = t + 1

            # record the back tracked results
            p.append(current_symbol)

            t -= 1

        # Sort and re-order again as the added ended sequences may change
        # the order (very unlikely)
        s, re_sorted_idx = s.topk(beam_width)
        for b_idx in range(batch_size):
            l[b_idx] = [l[b_idx][k_idx.item()] for k_idx in re_sorted_idx[b_idx,:]]

        re_sorted_idx = (re_sorted_idx + pos_index.expand_as(re_sorted_idx)).view(batch_size*beam_width)

        # Reverse the sequences and re-order at the same time
        # because the backtracking happens in reverse time order
        p = [step.index_select(0, re_sorted_idx).view(batch_size, beam_width, -1) for step in reversed(p)]
        p = torch.cat(p, -1)[:,0,:]
        return p, torch.ones_like(p)


class AttentionUnit(nn.Module):

    def __init__(self, s_dim, x_dim, att_dim):
        super(AttentionUnit, self).__init__()

        self.s_dim = s_dim
        self.x_dim = x_dim
        self.att_dim = att_dim

        self.sEmbed = nn.Linear(s_dim, att_dim)
        self.xEmbed = nn.Linear(x_dim, att_dim)
        self.wEmbed = nn.Linear(att_dim, 1)

        self.init_weights()

    def init_weights(self):
        nn.init.normal_(self.sEmbed.weight, std=0.01)
        nn.init.constant_(self.sEmbed.bias, 0)
        nn.init.normal_(self.xEmbed.weight, std=0.01)
        nn.init.constant_(self.xEmbed.bias, 0)
        nn.init.normal_(self.wEmbed.weight, std=0.01)
        nn.init.constant_(self.wEmbed.bias, 0)

    def forward(self, x, s_prev):
        batch_size, T, _ = x.size()                      # [ b x T  x   x_dim]
        x = x.view(-1, self.x_dim)                       # [(b x T) x   x_dim]
        x_proj = self.xEmbed(x)                           # [(b x T) x att_dim]
        x_proj = x_proj.view(batch_size, T, -1)            # [ b x T  x att_dim]

        s_prev = s_prev.squeeze(0)
        s_proj = self.sEmbed(s_prev)                        # [b     x att_dim]
        s_proj = torch.unsqueeze(s_proj, 1)                 # [b x 1 x att_dim]
        s_proj = s_proj.expand(batch_size, T, self.att_dim) # [b x T x att_dim]

        sum = torch.tanh(s_proj + x_proj).view(-1, self.att_dim)

        v_proj = self.wEmbed(sum) # [(b x T) x 1]
        v_proj = v_proj.view(batch_size, T)

        alpha = F.softmax(v_proj, dim=1) # attention weights for each sample in the minibatch

        return alpha


class DecoderUnit(nn.Module):
    def __init__(self, s_dim, x_dim, y_dim, att_dim):
        super(DecoderUnit, self).__init__()
        self.s_dim = s_dim
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.att_dim = att_dim
        self.emd_dim = att_dim

        self.attention_unit = AttentionUnit(s_dim, x_dim, att_dim)
        self.tgt_embedding = nn.Embedding(y_dim+1, self.emd_dim) # the last is used for <BOS> 
        self.gru = nn.GRU(input_size=x_dim+self.emd_dim, hidden_size=s_dim, batch_first=True)
        self.fc = nn.Linear(s_dim, y_dim)

        self.init_weights()

    def init_weights(self):
        nn.init.normal_(self.tgt_embedding.weight, std=0.01)
        nn.init.normal_(self.fc.weight, std=0.01)
        nn.init.constant_(self.fc.bias, 0)

    def forward(self, x, s_prev, y_prev):
        # x: feature sequence from the image decoder.
        B, T, _ = x.size()
        alpha = self.attention_unit(x, s_prev)
        context = torch.bmm(alpha.unsqueeze(1), x).squeeze(1)
        yProj = self.tgt_embedding(y_prev.long())

        # self.gru.flatten_parameters()
        output, state = self.gru(torch.cat([yProj, context], 1).unsqueeze(1), s_prev)
        output = output.squeeze(1)

        output = self.fc(output)
        return output, state


