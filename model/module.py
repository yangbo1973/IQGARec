import torch
import math
import torch.nn.functional as F
from torch import nn


class PointWiseFeedForward(torch.nn.Module):
    def __init__(self, hidden_size, dropout_rate):

        super(PointWiseFeedForward, self).__init__()

        self.conv1 = torch.nn.Conv1d(hidden_size, hidden_size, kernel_size=1)
        self.dropout1 = torch.nn.Dropout(p=dropout_rate)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(hidden_size, hidden_size, kernel_size=1)
        self.dropout2 = torch.nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        outputs = outputs.transpose(-1, -2) # as Conv1D requires (N, C, Length)
        outputs += inputs
        return outputs
    

class SinusoidalPositionEmbeddings(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings
    

class ConditionalLayerNorm(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.layernorm = torch.nn.LayerNorm(dim, elementwise_affine=False)
        self.mlp_g = torch.nn.Sequential(
            torch.nn.Linear(dim, dim),
            torch.nn.SiLU(),
            torch.nn.Linear(dim, dim)
        )
        self.mlp_b = torch.nn.Sequential(
            torch.nn.Linear(dim, dim),
            torch.nn.SiLU(),
            torch.nn.Linear(dim, dim)
        )

    def forward(self, s, t, h_0):
        
        # s: [batch, len + 1, dim]
        # t: [batch, dim]
        # h_0: [batch, len, dim]

        mlp_input = h_0 + t
        g = self.mlp_g(mlp_input)
        b = self.mlp_b(mlp_input)
        return g * self.layernorm(s) + b


class SiLU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, hidden_size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, hidden_size, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(hidden_size, hidden_size*4)
        self.w_2 = nn.Linear(hidden_size*4, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_normal_(self.w_1.weight)
        nn.init.xavier_normal_(self.w_2.weight)

    def forward(self, hidden):
        hidden = self.w_1(hidden)
        activation = 0.5 * hidden * (1 + torch.tanh(math.sqrt(2 / math.pi) * (hidden + 0.044715 * torch.pow(hidden, 3))))
        return self.w_2(self.dropout(activation))


class MultiHeadedAttention(nn.Module):
    def __init__(self, heads, hidden_size, dropout):
        super().__init__()
        assert hidden_size % heads == 0
        self.size_head = hidden_size // heads
        self.num_heads = heads
        self.linear_layers = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(3)])
        self.w_layer = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(p=dropout)
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_normal_(self.w_layer.weight)

    def forward(self, q, k, v, mask=None):
        batch_size = q.shape[0]
        q, k, v = [l(x).view(batch_size, -1, self.num_heads, self.size_head).transpose(1, 2) for l, x in zip(self.linear_layers, (q, k, v))]
        corr = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.size(-1))
        
        if mask is not None:
            mask = mask.unsqueeze(1).repeat([1, corr.shape[1], 1]).unsqueeze(-1).repeat([1,1,1,corr.shape[-1]])
            corr = corr.masked_fill(mask == 0, -1e9)
        prob_attn = F.softmax(corr, dim=-1)
        if self.dropout is not None:
            prob_attn = self.dropout(prob_attn)
        hidden = torch.matmul(prob_attn, v)
        hidden = self.w_layer(hidden.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.size_head))
        return hidden
    


class TransformerBlock(nn.Module):
    def __init__(self, hidden_size, attn_heads, dropout):
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadedAttention(heads=attn_heads, hidden_size=hidden_size, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(hidden_size=hidden_size, dropout=dropout)
        self.input_sublayer = SublayerConnection(hidden_size=hidden_size, dropout=dropout)
        self.output_sublayer = SublayerConnection(hidden_size=hidden_size, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, hidden, mask):
        hidden = self.input_sublayer(hidden, lambda _hidden: self.attention.forward(_hidden, _hidden, _hidden, mask=mask))
        hidden = self.output_sublayer(hidden, self.feed_forward)
        return self.dropout(hidden)
    


class Transformer_rep(nn.Module):
    def __init__(self, args):
        super(Transformer_rep, self).__init__()
        self.hidden_size = args.hidden_size
        self.heads = 4
        self.dropout = args.dropout
        self.n_blocks = args.num_blocks
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(self.hidden_size, self.heads, self.dropout) for _ in range(self.n_blocks)])

    def forward(self, hidden, mask):
        for transformer in self.transformer_blocks:
            hidden = transformer.forward(hidden, mask)
        return hidden


class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, dropout_rate, commitment_cost=4):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.dropout_rate = dropout_rate
        self.commitment_cost = commitment_cost
        
        # 初始化 Codebook
        self.embeddings = nn.Embedding(self.num_embeddings, self.embedding_dim)
        nn.init.uniform_(self.embeddings.weight, -1/self.num_embeddings, 1/self.num_embeddings)
        self.dropout = nn.Dropout(self.dropout_rate)
        self.layernorm = nn.LayerNorm(self.embedding_dim, elementwise_affine=False)

    def forward(self, z_e, mask):
        # 输入：z_e: [B, L, D]
        # 输入：mask: [B, L]
        # 输出：z_q: [B, L, D]

        # [BL, D]
        z_e_flat = z_e.view(-1, self.embedding_dim) 
        mask_flat = mask.view(-1)

        # 计算距离 [BL, num_embeddings]
        distances = (
            torch.sum(z_e_flat**2, dim=1, keepdim=True) +
            torch.sum(self.embeddings.weight**2, dim=1) -
            2 * torch.matmul(z_e_flat, self.embeddings.weight.t()))
        
        # 找到最近邻的 Codebook 索引
        # [BL]
        encoding_indices = torch.argmin(distances, dim=1)
        # [BL, D]
        quantized_flat = self.embeddings(encoding_indices)
        #quantized_flat = self.dropout(quantized_flat)

        # 计算损失
        e_loss = F.mse_loss(quantized_flat[mask_flat!=0].detach(), z_e_flat[mask_flat!=0])  # Commitment Loss
        q_loss = F.mse_loss(z_e_flat[mask_flat!=0].detach(), quantized_flat[mask_flat!=0])  # 码本Loss
        loss = q_loss + self.commitment_cost * e_loss

        # 直通梯度
        #quantized_flat = z_e_flat + (quantized_flat - z_e_flat).detach()

        # 恢复原始维度
        quantized = quantized_flat.view(z_e.shape)

        # 计算困惑度（Codebook 使用均匀性），码书使用频率的熵
        avg_probs = torch.histc(encoding_indices.float(), bins=self.num_embeddings, min=0, max=self.num_embeddings-1)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return quantized, loss, encoding_indices


class ResidualQuantizer(nn.Module):
    def __init__(self, n_stages, num_embeddings, embedding_dim, dropout_rate, commitment_cost=4):
        super().__init__()
        self.n_stages = n_stages
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.dropout_rate = dropout_rate
        self.commitment_cost = commitment_cost

        self.vq_layers = nn.ModuleList([VectorQuantizer(self.num_embeddings, self.embedding_dim, self.dropout_rate
                                                        , self.commitment_cost) for _ in range(n_stages)])

    def forward(self, z_e, mask):
        # 初始化残差和量化输出
        residual = z_e
        quantized_out = 0
        total_loss = 0
        encoding_indices = []

        for i in range(self.n_stages):
            # 量化当前残差
            quantized_i, loss_i, encoding_indices_i = self.vq_layers[i](residual, mask)

            # 更新量化输出和残差
            quantized_out = quantized_out + quantized_i
            residual = z_e - quantized_out

            # 累加损失和困惑度
            total_loss += loss_i
            encoding_indices.append(encoding_indices_i)

        return quantized_out, total_loss, encoding_indices
    

class SequenceVectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, max_len, embedding_dim, dropout_rate, commitment_cost=0.25):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.max_len = max_len
        self.num_embeddings = num_embeddings
        self.dropout_rate = dropout_rate
        self.commitment_cost = commitment_cost
        
        # 初始化 Codebook
        self.embeddings = nn.Embedding(self.num_embeddings, self.max_len * self.embedding_dim)
        nn.init.uniform_(self.embeddings.weight, -1/self.num_embeddings, 1/self.num_embeddings)
        self.dropout = nn.Dropout(self.dropout_rate)
        self.layernorm = nn.LayerNorm(self.max_len * self.num_embeddings)

    def forward(self, z_e, mask):
        # 输入：z_e: [B, L, D]
        # 输入：mask: [B, L]
        # 输出：z_q: [B, L, D]

        # [B, LD]
        z_e_flat = z_e.view(z_e.shape[0], -1) 

        # 计算距离 [B, num_embeddings]
        distances = (
            torch.sum(z_e_flat**2, dim=1, keepdim=True) +
            torch.sum(self.embeddings.weight**2, dim=1) -
            2 * torch.matmul(z_e_flat, self.embeddings.weight.t()))

        # 找到最近邻的 Codebook 索引
        # [B]
        encoding_indices = torch.argmin(distances, dim=1)
        # [B, LD]
        quantized_flat = self.embeddings(encoding_indices)
        quantized_flat = self.layernorm(self.dropout(quantized_flat))

        # 计算损失
        q_loss = F.mse_loss(quantized_flat.detach(), z_e_flat)  # 码本Loss
        e_loss = F.mse_loss(z_e_flat.detach(), quantized_flat)  # Commitment Loss
        loss = q_loss + self.commitment_cost * e_loss

        # 直通梯度
        #quantized_flat = z_e_flat + (quantized_flat - z_e_flat).detach()

        # 恢复原始维度
        quantized = quantized_flat.view(z_e.shape[0], z_e.shape[1], self.embedding_dim)

        # 计算困惑度（Codebook 使用均匀性），码书使用频率的熵
        avg_probs = torch.histc(encoding_indices.float(), bins=self.num_embeddings, min=0, max=self.num_embeddings-1)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return quantized, loss, perplexity
    




