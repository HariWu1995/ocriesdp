"""
Processing Key Information Extraction from Documents using Improved Graph Learning - Convolutional Networks
    Paper: https://arxiv.org/pdf/2004.07464.pdf
    Code: https://github.com/wenwenyu/PICK-pytorch/tree/master/model
"""

from typing import *

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.ops import roi_align
from torchvision.ops import roi_pool

from src.models.backbones.resnet import build_resnet
from src.models.graphs import GraphLearningConvolutionNetwork as GLCN
from src.models.layers import BidirLSTM
from src.models.heads.classification.CRF.model import ConditionalRandomField as CRF
from src.models.heads.extraction.PICK.utils import keys_vocab_cls, entities_iob_vocab_cls


class Encoder(nn.Module):

    def __init__(self, textual_dim: int, out_dim: int, visual_dim: int = 512,
                 n_heads: int = 8, n_layers: int = 6, hidden_dim: int = 2048,
                 dropout: float = 0.1, max_seq_len: int = 100, learn_positions: bool = True,
                 visual_encoder: str = 'resnet50', load_visual_encoder: bool = False,
                 roi_pooling_mode: str = 'roi_align',
                 roi_pooling_size: Tuple[int, int] = (7, 7)):
        """
        Convert image segments and text segments to node embedding.
        """
        super().__init__()

        self.dropout = dropout
        assert roi_pooling_mode in ['roi_align', 'roi_pool'], f'roi pooling model: {roi_pooling_mode} not support.'
        assert roi_pooling_size and len(roi_pooling_size) == 2, f'roi_pooling_size = {roi_pooling_size} not be set properly.'
        self.roi_pooling_size = tuple(roi_pooling_size)  # (h, w)
        self.roi_pooling_mode = roi_pooling_mode

        multimodal_encoder_layer = nn.TransformerEncoderLayer(d_model=textual_dim, nhead=n_heads,
                                                        dim_feedforward=hidden_dim, dropout=dropout)
        self.multimodal_encoder = nn.TransformerEncoder(multimodal_encoder_layer, num_layers=n_layers)

        visual_encoder = int(visual_encoder.lower().replace('resnet', ''))
        self.visual_encoder = build_resnet(output_channels=visual_dim, variant_depth=visual_encoder, load_pretrained=load_visual_encoder)

        self.conv = nn.Conv2d(visual_dim, out_dim, self.roi_pooling_size)
        self.bn = nn.BatchNorm2d(out_dim)
        self.proj = nn.Linear(2*out_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)

        # Compute the positional encodings once in log space.
        if learn_positions:
            from models.layers import PositionalEncodingLearnable as PositionalEncoding
        else:
            from models.layers import PositionalEncoding
            
        position_embedding = PositionalEncoding(d_model=textual_dim, max_seq_len=max_seq_len)
        position_embedding = position_embedding.unsqueeze(0).unsqueeze(0)  # (1, 1, max_seq_len, textual_dim)
        self.position_embedding = position_embedding
        # self.register_buffer('position_embedding', position_embedding) # only be used to non-learnable parameters

        self.pe_dropout = nn.Dropout(self.dropout)
        self.out_dropout = nn.Dropout(self.dropout)

    def forward(self, images: torch.Tensor, boxes_coordinate: torch.Tensor, transcripts: torch.Tensor,
                                        src_key_padding_mask: torch.Tensor,):
        """
        Parameters
        ----------
        images: shape is (B, N, H, W, C), where B is batch size, 
                                                N is number of segments of documents, 
                                                H is height of image, 
                                                W is width of image, 
                                                C is channel of image (default is 3).
        boxes_coordinate: shape is (B, N, 8), 
                                    where 8 is coordinates (x1, y1, x2, y2, x3, y3, x4, y4).
        transcripts: text segments, shape is (B, N, T, D), where T is max length of transcripts,
                                                                 D is dimension of model.
        src_key_padding_mask: text padding mask, shape is (B*N, T), True for padding value.
            if provided, specified padding elements in the key will be ignored by  attention.
            When value is True, the corresponding value on attention layer will be filled with -inf.

        Returns
        -------
        set of nodes X, shape is (B*N, T, D)
        """
        B, N, T, D = transcripts.shape

        # get image embedding 
        _, _, origin_H, origin_W = images.shape # (B, 3, H, W)
        visual_features = self.visual_encoder(images)
        _, C, H, W = visual_features.shape

        # generate RoIs for roi_pooling, RoIs shape is (B, N, 5), 5 means (batch_index, x0, y0, x1, y1)
        rois_batch = torch.zeros(B, N, 5, device=visual_features.device)
        for i in range(B):  # (B, N, 8)
            doc_boxes = boxes_coordinate[i] # (N, 8)       
            pos = torch.stack([doc_boxes[:, 0], doc_boxes[:, 1], doc_boxes[:, 4], doc_boxes[:, 5]], dim=1) # (N, 4)
            rois_batch[i, :, 1:5] = pos
            rois_batch[i, :, 0] = i

        spatial_scale = float(H / origin_H)
        # use roi pooling to get visual segments -> (B*N, C, roi_pooling_size, roi_pooling_size)
        if self.roi_pooling_mode == 'roi_align':
            visual_segments = roi_align(visual_features, rois_batch.view(-1, 5), self.roi_pooling_size, spatial_scale)
        else:
            visual_segments = roi_pool(visual_features, rois_batch.view(-1, 5), self.roi_pooling_size, spatial_scale)

        visual_segments = self.conv(visual_segments)
        visual_segments = self.bn(visual_segments)
        visual_segments = F.relu(visual_segments)           # (B*N, D, 1, 1)
        visual_segments = visual_segments.squeeze()         # (B*N, D,)
        visual_segments = visual_segments.unsqueeze(dim=1)  # (B*N, 1, D)

        # add positional embedding
        textual_segments = transcripts + self.position_embedding[:, :, :transcripts.size(2), :]
        textual_segments = self.pe_dropout(textual_segments)
        textual_segments = textual_segments.reshape(B * N, T, D) # (B*N, T, D)
        
        visual_segments = visual_segments.expand_as(textual_segments) # (B*N, 1, D) --> (B*N, T, D)

        # here we first aggregate visual embedding and textual embedding,
        # then get a non-local fusion features, different from paper process.
        out = visual_segments + textual_segments

        out = out.transpose(0, 1).contiguous() # (T, B*N, D)
        out = self.multimodal_encoder(out, src_key_padding_mask=src_key_padding_mask) # (T, B*N, D)
    
        out = out.transpose(0, 1).contiguous() # (B*N, T, D)
        out = self.norm(out)
        out = self.out_dropout(out)

        return out


class UnionLayer(nn.Module):

    def __init__(self, padding_value: int):
        self.padding_value = padding_value  # iob_labels_vocab_cls.stoi['<pad>']
        super().__init__()

    def forward(self, x: torch.Tensor, x_gcn: torch.Tensor, mask: torch.Tensor, length: torch.Tensor, tags):
        """
        For a document, we aggregate non-paddding (valid) x and x_gcn in a document-level format,
                        then feed it into CRF layer.
        
        Parameters
        ----------
        x: set of nodes, the output of encoder, (B, N, T, D)
        x_gcn: node embedding, the output of graph module, (B, N, D)
        mask: whether is non-padding value at i-th position of segments, (B, N, T)
        length: the length of every segments (boxes) of documents, (B, N)
        tags: IBO labels for every segments (boxes) of documents, (B, N, T)

        Returns
        -------
        new_x       (B, max_doc_seq_len, D)
        new_mask    (B, max_doc_seq_len)
        doc_seq_len (B,)
        new_tags    (B, max_doc_seq_len)
        """
        B, N, T, D = x.shape
        x = x.reshape(B, N * T, -1)
        mask = mask.reshape(B, N * T)

        # combine x and x_gcn
        x_gcn = x_gcn.unsqueeze(2).expand(B, N, T, -1) # (B, N, T, D)
        x_gcn = x_gcn.reshape(B, N * T, -1) # (B, N*T, D)
        x = x_gcn + x # (B, N*T, D)

        doc_seq_len = length.sum(dim=-1) # (B,)

        # dynamically calculate max_doc_seq_len
        max_doc_seq_len = doc_seq_len.max()

        # statically calculate max_doc_seq_len
        # max_doc_seq_len = MAX_BOXES_NUM * MAX_TRANSCRIPT_LEN

        # init x, mask, tags value
        new_x = torch.zeros_like(x, device=x.device)        # (B, N*T, D)
        new_mask = torch.zeros_like(mask, device=x.device)  # (B, N*T)
        if self.training:
            tags = tags.reshape(B, N * T) # (B, N*T)
            new_tags = torch.full_like(tags, self.padding_value, device=x.device)
            new_tags = new_tags[:, :max_doc_seq_len]

        # merge all non-padding value in document-level
        for i in range(B):
            doc_x = x[i]        # (N*T, D)
            doc_mask = mask[i]  # (N*T,)
            valid_doc_x = doc_x[doc_mask == 1]      # (num_valids, D)
            num_valids = valid_doc_x.size(0)
            new_x[i, :num_valids] = valid_doc_x     # (B, N*T, D)
            new_mask[i, :doc_seq_len[i]] = 1        # (B, N*T)

            if self.training:
                valid_tags = tags[i][doc_mask == 1]
                new_tags[i, :num_valids] = valid_tags

        new_x = new_x[:, :max_doc_seq_len, :]       # (B, max_doc_seq_len, D)
        new_mask = new_mask[:, :max_doc_seq_len]    # (B, max_doc_seq_len)

        if self.training:
            return new_x, new_mask, doc_seq_len, new_tags
        else:
            return new_x, new_mask, doc_seq_len, None


class Decoder(nn.Module):

    def __init__(self, padding_values: Dict[int, int], bilstm_config, mlp_config, crf_config):
        super().__init__()
        self.union_layer = UnionLayer(padding_values['iob_vocab'])  # iob_labels_vocab_cls.stoi['<pad>']
        self.bilstm_layer = BidirLSTM(padding_values['keys_vocab'],       # keys_vocab_cls.stoi['<pad>']
                                        bilstm_config, mlp_config,) 
        self.crf_layer = CRF(**crf_config)

    def forward(self, x: torch.Tensor, x_gcn: torch.Tensor, mask: torch.Tensor, 
                 length: torch.Tensor, tags: torch.Tensor,):
        """
        Parameters
        ----------        
        x: set of nodes, the output of encoder, (B, N, T, D)
        x_gcn: node embedding, the output of graph module, (B, N, D)
        mask: whether is non-padding value at i-th position of segments, (B, N, T)
        length: the length of every segments (boxes) of documents, (B, N)
        tags: IBO labels for every segments (boxes) of documents, (B, N, T)

        Returns
        -------
        new_x       (B, max_doc_seq_len, D)
        new_mask    (B, max_doc_seq_len)
        doc_seq_len (B,)
        new_tags    (B, max_doc_seq_len)
        """
        new_x, new_mask, doc_seq_len, new_tags = self.union_layer(x, x_gcn, mask, length, tags)

        logits = self.bilstm_layer(new_x, doc_seq_len, (None, None)) # (B, N*T, out_dim)

        if self.training:
            log_likelihood = self.crf_layer(logits, new_tags, mask=new_mask, input_batch_first=True, keepdim=True) # (B,)
        else:
            log_likelihood = None

        return logits, new_mask, log_likelihood


class PICK(nn.Module):
    
    def __init__(self, **kwargs):
        super().__init__()
        embedding_config = kwargs['embedding_config']
        encoder_config = kwargs['encoder_config']
        graph_config = kwargs['graph_config']
        decoder_config = kwargs['decoder_config']
        self.make_model(embedding_config, encoder_config, graph_config, decoder_config)

    def make_model(self, embedding_config, encoder_config, graph_config, decoder_config):
        # Given the params of each component, creates components.
        # embedding_config -> word_emb
        embedding_config['num_embeddings'] = len(keys_vocab_cls)
        self.word_emb = nn.Embedding(**embedding_config)

        encoder_config['char_embedding_dim'] = embedding_config['embedding_dim']
        self.encoder = Encoder(**encoder_config)

        graph_config['in_dim'] = encoder_config['out_dim']
        graph_config['out_dim'] = encoder_config['out_dim']
        self.graph = GLCN(**graph_config)

        decoder_config['bilstm_config']['input_size'] = encoder_config['out_dim']
        if decoder_config['bilstm_config']['bidirectional']:
            decoder_config['mlp_config']['in_dim'] = decoder_config['bilstm_config']['hidden_size'] * 2
        else:
            decoder_config['mlp_config']['in_dim'] = decoder_config['bilstm_config']['hidden_size']
        decoder_config['mlp_config']['out_dim'] = len(entities_iob_vocab_cls)
        decoder_config['crf_config']['num_tags'] = len(entities_iob_vocab_cls)
        self.decoder = Decoder(**decoder_config)

    def _aggregate_avg_pooling(self, text_input, text_mask):
        """
        Apply mean pooling over time (text length), (B*N, T, D) -> (B*N, D)

        Parameters
        ----------
        text_input : (B*N, T, D)
        text_mask  : (B*N, T)
        
        Returns
        -------
        output : (B*N, D)
        """
        # filter out padding value, (B*N, T, D)
        text_input = text_input * text_mask.detach().unsqueeze(2).float()
        
        sum_out = torch.sum(text_input, dim=1)   # (B*N, D)
        
        text_len = text_mask.float().sum(dim=1)                 # (B*N, )
        text_len = text_len.unsqueeze(1).expand_as(sum_out)     # (B*N, D)
        text_len = text_len + text_len.eq(0).float()  # avoid divide zero denominator
        
        mean_out = sum_out.div(text_len) # (B*N, D)
        return mean_out

    @staticmethod
    def compute_mask(mask: torch.Tensor):
        """
        Parameters
        ----------
        mask: (B, N, T)
        
        Returns
        -------
        graph_node_mask
        src_key_padding_mask
        """
        B, N, T = mask.shape
        mask = mask.reshape(B * N, T)
        mask_sum = mask.sum(dim=-1)         # (B*N,)

        graph_node_mask = mask_sum != 0     # (B*N,)
        graph_node_mask = graph_node_mask.unsqueeze(-1).expand(B * N, T)  # True for valid node

        # If src key are all be masked (indicting text segments is null), attention_weight will be nan after softmax
        # So we do not mask all padded sample. Instead, we mask it after Transformer encoding.
        src_key_padding_mask = torch.logical_not(mask.bool()) & graph_node_mask  # True for padding mask position
        return src_key_padding_mask, graph_node_mask

    def forward(self, **config):
        # input
        whole_image = config['whole_image']  # (B, 3, H, W)
        relation_features = config['relation_features']  # initial relation embedding (B, N, N, 6)
        text_segments = config['text_segments']  # text segments (B, N, T)
        text_length = config['text_length']  # (B, N)
        iob_tags_label = config['iob_tags_label'] if self.training else None  # (B, N, T)
        mask = config['mask']  # (B, N, T)
        boxes_coordinate = config['boxes_coordinate']  # (B, num_boxes, 8)

        """ 
        Encoder module 
        """
        # word embedding
        text_emb = self.word_emb(text_segments)

        # src_key_padding_mask is text padding mask, True is padding value (B*N, T)
        # graph_node_mask is mask for graph, True is valid node, (B*N, T)
        src_key_padding_mask, graph_node_mask = self.compute_mask(mask)

        # set of nodes, (B*N, T, D)
        x = self.encoder(images=whole_image, boxes_coordinate=boxes_coordinate, transcripts=text_emb,
                         src_key_padding_mask=src_key_padding_mask)

        """ 
        Graph module 
        """
        # text_mask, True for valid, (including all not valid node), (B*N, T)
        text_mask = torch.logical_not(src_key_padding_mask).byte()

        # (B*N, T, D) -> (B*N, D)
        x_gcn = self._aggregate_avg_pooling(x, text_mask)
        
        # (B*N, 1)，True is valid node
        graph_node_mask = graph_node_mask.any(dim=-1, keepdim=True)
        
        # (B*N, D), filter out not valid node
        x_gcn = x_gcn * graph_node_mask.byte()

        # initial adjacent matrix (B, N, N)
        B, N, T = mask.shape
        init_adj = torch.ones((B, N, N), device=text_emb.device)
        boxes_num = mask[:, :, 0].sum(dim=1, keepdim=True)  # (B, 1)
        
        # (B, N, D)
        x_gcn = x_gcn.reshape(B, N, -1)
        
        # (B, N, D), (B, N, N), (B,)
        x_gcn, soft_adj, gl_loss = self.graph(x_gcn, relation_features, init_adj, boxes_num)
        adj = soft_adj * init_adj

        """ 
        Decoder module 
        """
        logits, new_mask, log_likelihood = self.decoder(x.reshape(B, N, T, -1), x_gcn, mask, text_length, iob_tags_label)

        output = {"logits": logits, "new_mask": new_mask, "adj": adj}
        if self.training:
            output['gl_loss'] = gl_loss
            crf_loss = -log_likelihood
            output['crf_loss'] = crf_loss
        return output

    def __str__(self):
        """ Model prints with number of trainable parameters """
        return super().__str__() + '\n Trainable parameters: {}'.format(self.count_parameters())

    def count_parameters(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return params



