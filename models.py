import torch.nn as nn
import torch.nn.functional as F
import torch
import math
from dgllife.model.gnn import GCN
from ban import BANLayer
from torch.nn.utils.weight_norm import weight_norm
import dgl
import numpy as np
import os
import torch.nn.init as init
import warnings
from torch import Tensor
from typing import Optional
from typing import Tuple



def binary_cross_entropy(pred_output, labels):
    loss_fct = torch.nn.BCELoss()
    m = nn.Sigmoid()
    n = torch.squeeze(m(pred_output), 1)
    loss = loss_fct(n, labels)
    return n, loss



def cross_entropy_logits(linear_output, label, weights=None):
    class_output = F.log_softmax(linear_output, dim=1)
    n = F.softmax(linear_output, dim=1)[:, 1]
    max_class = class_output.max(1)
    y_hat = max_class[1]  # get the index of the max log-probability
    if weights is None:
        loss = nn.NLLLoss()(class_output, label.type_as(y_hat).view(label.size(0)))
    else:
        losses = nn.NLLLoss(reduction="none")(class_output, label.type_as(y_hat).view(label.size(0)))
        loss = torch.sum(weights * losses) / torch.sum(weights)
    return n, loss



def entropy_logits(linear_output):
    p = F.softmax(linear_output, dim=1)
    loss_ent = -torch.sum(p * (torch.log(p + 1e-5)), dim=1)
    return loss_ent



class GBAN_DA(nn.Module):
    def __init__(self, **config):
        super(GBAN_DA, self).__init__()
        drug_in_feats = config["DRUG"]["NODE_IN_FEATS"]
        drug_embedding = config["DRUG"]["NODE_IN_EMBEDDING"]
        drug_hidden_feats = config["DRUG"]["HIDDEN_LAYERS"]
        protein_emb_dim = config["PROTEIN"]["EMBEDDING_DIM"]
        num_filters = config["PROTEIN"]["NUM_FILTERS"]
        kernel_size = config["PROTEIN"]["KERNEL_SIZE"]
        mlp_in_dim = config["DECODER"]["IN_DIM"]
        mlp_hidden_dim = config["DECODER"]["HIDDEN_DIM"]
        mlp_out_dim = config["DECODER"]["OUT_DIM"]
        drug_padding = config["DRUG"]["PADDING"]
        protein_padding = config["PROTEIN"]["PADDING"]
        out_binary = config["DECODER"]["BINARY"]
        ban_heads = config["BCN"]["HEADS"]

        # 新增特征注意力模块
        drug_hidden = config["DRUG"]["HIDDEN_LAYERS"][-1]

        self.drug_att = FeatureAttention(drug_hidden)


        self.drug_extractor = MolecularGCN(in_feats=drug_in_feats, dim_embedding=drug_embedding,
                                           padding=drug_padding,
                                           hidden_feats=drug_hidden_feats)


        self.protein_embed = nn.Embedding(26, protein_emb_dim, padding_idx=0)

        self.protein_extractor = ProteinEncoder(
            max_len=1000,
            encoder_dim=protein_emb_dim,
            num_layers=3,
            num_attention_heads=4,
            feed_forward_expansion_factor=4,
            feed_forward_dropout_p=0.1,
            attention_dropout_p=0.1,
        )

        self.protein_extractor1 = ProteinCNN(protein_emb_dim, num_filters, kernel_size, protein_padding)

        self.pro_fusion = proFusion()

        self.bcn = weight_norm(
            BANLayer(v_dim=drug_hidden_feats[-1], q_dim=num_filters[-1], h_dim=mlp_in_dim, h_out=ban_heads),
            name='h_mat', dim=None)

        self.mlp_classifier = MLPDecoder(mlp_in_dim, mlp_hidden_dim, mlp_out_dim, binary=out_binary)

    def forward(self, bg_d, v_p, protein_mask, mode="train"):

        v_d = self.drug_extractor(bg_d)


        v_p0 = self.protein_embed(v_p.long())
        v_p1 = self.protein_extractor(v_p0, protein_mask)  # 传入mask
        v_p_cnn = self.protein_extractor1(v_p)
        v_p = self.pro_fusion(v_p_cnn, v_p1)


        v_d = self.drug_att(v_d)

        f, att = self.bcn(v_d, v_p)

        score = self.mlp_classifier(f)

        if mode == "train":
            return v_d, v_p, f, score
        elif mode == "eval":
            return v_d, v_p, score, att

class MolecularGCN(nn.Module):
    def __init__(self, in_feats, dim_embedding=128, padding=True, hidden_feats=None, activation=None):
        super(MolecularGCN, self).__init__()
        self.init_transform = nn.Linear(in_feats, dim_embedding, bias=False)
        if padding:
            with torch.no_grad():
                self.init_transform.weight[-1].fill_(0)
        self.gnn = GCN(in_feats=dim_embedding, hidden_feats=hidden_feats, activation=activation)
        self.output_feats = hidden_feats[-1]

    def forward(self, batch_graph):
        node_feats = batch_graph.ndata.pop('h')
        node_feats = self.init_transform(node_feats)
        node_feats = self.gnn(batch_graph, node_feats)
        batch_size = batch_graph.batch_size
        node_feats = node_feats.view(batch_size, -1, self.output_feats)
        return node_feats



class ProteinEncoder(nn.Module):
    def __init__(
        self,
        max_len: int = 1000,
        encoder_dim: int = 512,
        num_layers: int = 3,
        num_attention_heads: int = 8,
        feed_forward_expansion_factor: int = 4,
        feed_forward_dropout_p: float = 0.1,
        attention_dropout_p: float = 0.1,

    ):
        super(ProteinEncoder, self).__init__()

        self.attention_blocks = nn.ModuleList([AttentionFFNBlock(
            encoder_dim=encoder_dim,
            num_attention_heads=num_attention_heads,
            feed_forward_expansion_factor=feed_forward_expansion_factor,
            feed_forward_dropout_p=feed_forward_dropout_p,
            attention_dropout_p=attention_dropout_p,
            max_len=max_len
        ) for _ in range(num_layers)])

    def forward(self, inputs: Tensor, mask: Tensor) -> Tensor:
        outputs = inputs
        for block in self.attention_blocks:
            outputs = block(outputs, mask)
        return outputs

class AttentionFFNBlock(nn.Module):

    def __init__(
        self,
        encoder_dim: int = 512,
        num_attention_heads: int = 8,
        feed_forward_expansion_factor: int = 4,
        feed_forward_dropout_p: float = 0.1,
        attention_dropout_p: float = 0.1,
        max_len: int = 1000
    ):
        super(AttentionFFNBlock, self).__init__()
        self.MHSA_model = MultiHeadedSelfAttentionModule(
            d_model=encoder_dim,
            num_heads=num_attention_heads,
            dropout_p=attention_dropout_p,
            max_len=max_len
        )

        self.FF_model = FeedForwardModule(
            encoder_dim=encoder_dim,
            expansion_factor=feed_forward_expansion_factor,
            dropout_p=feed_forward_dropout_p,
        )
        self.layer_norm = nn.LayerNorm(encoder_dim)

    def forward(self, inputs: Tensor, mask: Tensor) -> Tensor:

        attn_output = self.MHSA_model(inputs, mask)
        attn_output = self.layer_norm(attn_output + inputs)
        ffn_output = self.FF_model(attn_output)
        outputs = self.layer_norm(ffn_output + attn_output)
        return outputs

class MultiHeadedSelfAttentionModule(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout_p: float = 0.1, max_len: int = 1000):
        super(MultiHeadedSelfAttentionModule, self).__init__()
        self.positional_encoding = PositionalEncoding(d_model, max_len=max_len)
        self.layer_norm = nn.LayerNorm(d_model)
        self.attention = RelativeMultiHeadAttention(d_model, num_heads)
        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, inputs: Tensor, mask: Optional[Tensor] = None):
        batch_size, seq_length, _ = inputs.size()
        pos_embedding = self.positional_encoding(seq_length)
        pos_embedding = pos_embedding.repeat(batch_size, 1, 1)
        inputs = self.layer_norm(inputs)
        outputs = self.attention(inputs, inputs, inputs, pos_embedding=pos_embedding, mask=mask)
        return self.dropout(outputs)


class RelativeMultiHeadAttention(nn.Module):
    def __init__(self, d_model: int = 512, num_heads: int = 8):
        super(RelativeMultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model % num_heads should be zero."
        self.d_model = d_model
        self.d_head = int(d_model / num_heads)
        self.num_heads = num_heads
        self.sqrt_dim = math.sqrt(d_model)
        self.query_proj = Linear(d_model, d_model)
        self.key_proj = Linear(d_model, d_model)
        self.value_proj = Linear(d_model, d_model)
        self.pos_proj = Linear(d_model, d_model, bias=False)
        self.u_bias = nn.Parameter(torch.Tensor(self.num_heads, self.d_head))
        self.v_bias = nn.Parameter(torch.Tensor(self.num_heads, self.d_head))
        torch.nn.init.xavier_uniform_(self.u_bias)
        torch.nn.init.xavier_uniform_(self.v_bias)
        self.out_proj = Linear(d_model, d_model)

    def forward(self, query: Tensor, key: Tensor, value: Tensor, pos_embedding: Tensor, mask: Tensor) -> Tensor:
        batch_size = value.size(0)
        query = self.query_proj(query).view(batch_size, -1, self.num_heads, self.d_head)
        key = self.key_proj(key).view(batch_size, -1, self.num_heads, self.d_head).permute(0, 2, 1, 3)
        value = self.value_proj(value).view(batch_size, -1, self.num_heads, self.d_head).permute(0, 2, 1, 3)
        pos_embedding = self.pos_proj(pos_embedding).view(batch_size, -1, self.num_heads, self.d_head)
        content_score = torch.matmul((query + self.u_bias).transpose(1, 2), key.transpose(2, 3))
        pos_score = torch.matmul((query + self.v_bias).transpose(1, 2), pos_embedding.permute(0, 2, 3, 1))
        pos_score = self._relative_shift(pos_score)
        score = (content_score + pos_score) / self.sqrt_dim
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)
            score.masked_fill(mask == 0, -1e9)
        attn = F.softmax(score, -1)
        context = torch.matmul(attn, value).transpose(1, 2)
        context = context.contiguous().view(batch_size, -1, self.d_model)
        return self.out_proj(context)

    def _relative_shift(self, pos_score: Tensor) -> Tensor:
        batch_size, num_heads, seq_length1, seq_length2 = pos_score.size()
        zeros = pos_score.new_zeros(batch_size, num_heads, seq_length1, 1)
        padded_pos_score = torch.cat([zeros, pos_score], dim=-1)
        padded_pos_score = padded_pos_score.view(batch_size, num_heads, seq_length2 + 1, seq_length1)
        pos_score = padded_pos_score[:, :, 1:].view_as(pos_score)
        return pos_score


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int = 512, max_len: int = 1000) -> None:
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model, requires_grad=False)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, length: int) -> Tensor:
        return self.pe[:, :length]


class FeedForwardModule(nn.Module):
    def __init__(self, encoder_dim: int = 512, expansion_factor: int = 4, dropout_p: float = 0.1):
        super(FeedForwardModule, self).__init__()
        self.sequential = nn.Sequential(
            nn.LayerNorm(encoder_dim),
            Linear(encoder_dim, encoder_dim * expansion_factor, bias=True),
            Swish(),
            nn.Dropout(p=dropout_p),
            Linear(encoder_dim * expansion_factor, encoder_dim, bias=True),
            nn.Dropout(p=dropout_p),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.sequential(inputs)


class ConvModule(nn.Module):
    def __init__(self, in_channels: int, kernel_size: int = 5, dropout_p: float = 0.1):
        super(ConvModule, self).__init__()
        self.sequential = nn.Sequential(
            nn.LayerNorm(in_channels),
            Transpose(shape=(1, 2)),
            nn.Conv1d(in_channels=in_channels, out_channels=in_channels * 2, kernel_size=1, stride=1, padding=0),
            Swish(),
            nn.Conv1d(
                in_channels=in_channels * 2,
                out_channels=in_channels * 2,
                kernel_size=kernel_size,
                stride=1,
                padding=(kernel_size - 1) // 2,
                groups=in_channels * 2,
            ),
            GLU(dim=1),
            nn.BatchNorm1d(in_channels),
            nn.Conv1d(in_channels=in_channels, out_channels=in_channels, kernel_size=1, stride=1, padding=0),
            nn.Dropout(p=dropout_p),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.sequential(inputs).transpose(1, 2)


class GLU(nn.Module):
    def __init__(self, dim: int) -> None:
        super(GLU, self).__init__()
        self.dim = dim

    def forward(self, inputs: Tensor) -> Tensor:
        outputs, gate = inputs.chunk(2, dim=self.dim)
        return outputs * gate.sigmoid()


class Swish(nn.Module):
    def forward(self, inputs: Tensor) -> Tensor:
        return inputs * inputs.sigmoid()


class Transpose(nn.Module):
    def __init__(self, shape: tuple):
        super(Transpose, self).__init__()
        self.shape = shape

    def forward(self, x: Tensor) -> Tensor:
        return x.transpose(*self.shape)


class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super(Linear, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        init.xavier_uniform_(self.linear.weight)
        if bias:
            init.zeros_(self.linear.bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.linear(x)


class MLPDecoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, binary=1):
        super(MLPDecoder, self).__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_dim)
        self.bn3 = nn.BatchNorm1d(out_dim)
        self.fc4 = nn.Linear(out_dim, binary)

    def forward(self, x):
        x = self.bn1(F.relu(self.fc1(x)))
        x = self.bn2(F.relu(self.fc2(x)))
        x = self.bn3(F.relu(self.fc3(x)))
        x = self.fc4(x)
        return x


class SimpleClassifier(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, dropout):
        super(SimpleClassifier, self).__init__()
        layers = [
            weight_norm(nn.Linear(in_dim, hid_dim), dim=None),
            nn.ReLU(),
            nn.Dropout(dropout, inplace=True),
            weight_norm(nn.Linear(hid_dim, out_dim), dim=None)
        ]
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        logits = self.main(x)
        return logits


class RandomLayer(nn.Module):
    def __init__(self, input_dim_list, output_dim=256):
        super(RandomLayer, self).__init__()
        self.input_num = len(input_dim_list)
        self.output_dim = output_dim
        self.random_matrix = [torch.randn(input_dim_list[i], output_dim) for i in range(self.input_num)]

    def forward(self, input_list):
        return_list = [torch.mm(input_list[i], self.random_matrix[i]) for i in range(self.input_num)]
        return_tensor = return_list[0] / math.pow(float(self.output_dim), 1.0 / len(return_list))
        for single in return_list[1:]:
            return_tensor = torch.mul(return_tensor, single)
        return return_tensor

    def cuda(self):
        super(RandomLayer, self).cuda()
        self.random_matrix = [val.cuda() for val in self.random_matrix]

class proFusion(nn.Module):
    def forward(self, A, B):
        result_1 = torch.matmul(A, B.transpose(1, 2))
        result_2 = torch.matmul(result_1, B)
        return torch.add(result_2, A)

class ProteinCNN(nn.Module):
    def __init__(self, embedding_dim, num_filters, kernel_size, padding=True):
        super(ProteinCNN, self).__init__()
        if padding:
            self.embedding = nn.Embedding(26, embedding_dim, padding_idx=0)
        else:
            self.embedding = nn.Embedding(26, embedding_dim)
        in_ch = [embedding_dim] + num_filters
        self.in_ch = in_ch[-1]
        kernels = kernel_size
        self.conv1 = nn.Conv1d(in_channels=in_ch[0], out_channels=in_ch[1], kernel_size=kernels[0])
        self.bn1 = nn.BatchNorm1d(in_ch[1])
        self.conv2 = nn.Conv1d(in_channels=in_ch[1], out_channels=in_ch[2], kernel_size=kernels[1])
        self.bn2 = nn.BatchNorm1d(in_ch[2])
        self.conv3 = nn.Conv1d(in_channels=in_ch[2], out_channels=in_ch[3], kernel_size=kernels[2])
        self.bn3 = nn.BatchNorm1d(in_ch[3])

    def forward(self, v):
        v = self.embedding(v.long())
        v = v.transpose(2, 1)
        v = self.bn1(F.relu(self.conv1(v)))
        v = self.bn2(F.relu(self.conv2(v)))
        v = self.bn3(F.relu(self.conv3(v)))
        v = v.view(v.size(0), v.size(2), -1)
        return v



class FeatureAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16, use_spatial=True):
        super(FeatureAttention, self).__init__()
        self.use_spatial = use_spatial


        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(in_channels, in_channels // reduction_ratio, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels // reduction_ratio, in_channels, 1),
            nn.Sigmoid()
        )


        if use_spatial:
            self.spatial_att = nn.Sequential(
                nn.Conv1d(in_channels, in_channels // reduction_ratio, 1),
                nn.ReLU(inplace=True),
                nn.Conv1d(in_channels // reduction_ratio, 1, 1),
                nn.Sigmoid()
            )

    def forward(self, x):

        x = x.permute(0, 2, 1)


        channel_weights = self.channel_att(x)
        x_channel = x * channel_weights


        if self.use_spatial:
            spatial_weights = self.spatial_att(x_channel)
            x_spatial = x_channel * spatial_weights
            x_out = x_spatial.permute(0, 2, 1)
        else:
            x_out = x_channel.permute(0, 2, 1)

        return x_out








