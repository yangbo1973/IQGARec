import torch
from .module import *
import torch.nn.functional as F
from utils import *
from sklearn.cluster import KMeans

class IQGARec(torch.nn.Module):
    def __init__(self, hidden_size, item_num, state_size, emb_dropout_rate, dropout_rate, delta, 
                 lam_history, lam_intent, lam_rqloss, lam_cl_loss, ratio_substitute, n_stages, 
                 device, timesteps, beta_start, beta_end, beta_sche, num_clusters, num_heads=1, num_blocks=1):
        super(IQGARec, self).__init__()

        self.device = device
        self.hidden_size = hidden_size
        self.item_num = item_num
        self.state_size = state_size
        self.dropout_rate = dropout_rate
        self.emb_dropout_rate = emb_dropout_rate
        self.num_clusters = num_clusters
        self.num_heads = num_heads
        self.num_blocks = num_blocks
        self.delta = torch.tensor(delta).to(device)
        self.lam_history = lam_history
        self.lam_intent = lam_intent
        self.lam_rqloss = lam_rqloss
        self.lam_cl_loss = lam_cl_loss
        self.ratio_substitute = ratio_substitute
        self.n_stages = n_stages

        self.timesteps = timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta_sche = beta_sche

        self.item_embeddings = torch.nn.Embedding(self.item_num + 1, self.hidden_size, padding_idx = 0)
        torch.nn.init.normal_(self.item_embeddings.weight[1:], 0, 1)
        self.pos_embeddings = torch.nn.Embedding(self.state_size, self.hidden_size)
        self.emb_layernorm = torch.nn.LayerNorm(self.hidden_size)
        self.emb_dropout = torch.nn.Dropout(self.emb_dropout_rate)

        # self.transformer_block = torch.nn.Sequential()
        # for _ in range(self.num_blocks):
        #     self.transformer_block.append(MultiheadAttention(self.hidden_size, self.num_heads, self.dropout_rate, self.device))
        self.last_layernorm = torch.nn.LayerNorm(self.hidden_size, eps=1e-12)

        self.step_mlp = torch.nn.Sequential(
            SinusoidalPositionEmbeddings(self.hidden_size),
            torch.nn.Linear(self.hidden_size, self.hidden_size*2),
            torch.nn.SiLU(),
            torch.nn.Linear(self.hidden_size*2, self.hidden_size),
        )

        self.att = nn.ModuleList(
            [TransformerBlock(self.hidden_size, self.num_heads, self.dropout_rate) for _ in range(self.num_blocks)])
        self.dropout = torch.nn.Dropout(self.dropout_rate)

        self.rq = ResidualQuantizer(self.n_stages, self.num_clusters, self.hidden_size, self.dropout_rate).to(self.device)
        self.init_rq()

        self.CE_loss = torch.nn.CrossEntropyLoss()
        self.MSE_loss = torch.nn.MSELoss()

        # diffusion
        self.timesteps = timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta_sche = beta_sche
        self.device = device

        if beta_sche == 'linear':
            self.betas = linear_beta_schedule(timesteps=self.timesteps, beta_start=self.beta_start, beta_end=self.beta_end)
        elif beta_sche == 'exp':
            self.betas = exp_beta_schedule(timesteps=self.timesteps)
        elif beta_sche =='cosine':
            self.betas = cosine_beta_schedule(timesteps=self.timesteps)
        elif beta_sche =='sqrt':
            self.betas = torch.tensor(betas_for_alpha_bar(self.timesteps, lambda t: 1-np.sqrt(t + 0.0001),)).float()
        elif beta_sche == 'trunc_lin':
            scale = 1000 / self.timesteps
            beta_start = scale * 0.0001 + 0.01
            beta_end = scale * 0.02 + 0.01
            if beta_end > 1:
                beta_end = scale * 0.001 + 0.01
            self.betas = torch.linspace(beta_start, beta_end, self.timesteps, dtype=torch.float32)
        self.betas = self.betas.to(device)

        # define alphas 
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1. / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1. / self.alphas_cumprod - 1)

        self.posterior_mean_coef1 = self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        self.posterior_mean_coef2 = (1. - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1. - self.alphas_cumprod)
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)



    def init_rq(self):
        item_emb_weight_np = self.item_embeddings.weight.cpu().detach().numpy()
        weight = item_emb_weight_np

        for i in range(self.n_stages):
            # 执行KMeans聚类
            kmeans = KMeans(n_clusters=self.num_clusters, random_state=0, n_init='auto').fit(weight)
            # 返回聚类中心（PyTorch Tensor）
            centers = torch.tensor(kmeans.cluster_centers_, 
                            dtype=self.item_embeddings.weight.dtype, 
                            device=self.device)
            self.rq.vq_layers[i].embeddings.weight = torch.nn.Parameter(centers)
            quantized, _, _ = self.rq.vq_layers[i](torch.tensor(weight).to(self.device), torch.ones(weight.shape[0]))
            weight -= quantized.detach().cpu().numpy()


    def get_item_emb(self, item):
        return self.item_embeddings(item)


    def forward(self, seq, x_t, step, train_flag = False):

        # seq: [batch, len]
        # x_t: [batch, dim]
        # step: [batch]

        item_emb = self.item_embeddings(seq)
        mask_seq = (seq>0).float()
        quantized, rq_loss, indices = self.rq(item_emb, mask_seq) 
        #pos_id = torch.arange(self.state_size, dtype=torch.long, device=self.device).unsqueeze(0)
        #item_emb += self.pos_embeddings(pos_id)
        item_emb = self.emb_dropout(item_emb)
        item_emb = self.emb_layernorm(item_emb)

        step_emb = self.step_mlp(step) # [bacth, dim]
        #lam = self.delta + torch.randn(seq.shape[0], self.state_size, device=self.device) * torch.sqrt(self.delta) # [batch, len]

        last_item_emb = item_emb[:, -1, :]
        lam = torch.normal(mean=torch.full_like(last_item_emb, self.delta), std=torch.full_like(last_item_emb, self.delta)).to(self.device)
        
        z = self.lam_history * item_emb # + lam * (x_t.unsqueeze(1) + step_emb.unsqueeze(1))
        z[:, -1, :] += lam * (x_t + step_emb)
        if train_flag:
            z += self.lam_intent * quantized
        
        for i in range(self.num_blocks):
            z = self.att[i](z, mask_seq)
        rep_diffu = self.last_layernorm(self.dropout(z))
        out = rep_diffu[:, -1, :]
        return out, rq_loss
    
    
    def get_all_indices(self):
        emb_weight = self.item_embeddings.weight # [item_num, dim]
        mask = torch.ones(emb_weight.shape[0])
        _, _, indices = self.rq(emb_weight, mask)
        return indices[0]
    

    def augmentation(self, seq, ratio=0.1):
        # seq: [batch, max_len]
        seq_aug = seq.clone()
        len_seq = (seq != 0).sum(dim=-1)
        
        # 计算每个序列需要替换的数量
        num_substitute = torch.floor(len_seq * ratio).int()
        
        # 找出需要替换的序列
        valid_mask = num_substitute > 0
        if not valid_mask.any():
            return seq_aug
        
        # 获取所有有效序列的索引和对应的替换数量
        valid_indices = torch.where(valid_mask)[0]
        valid_num_substitute = num_substitute[valid_mask]
        
        # 计算每个有效序列中需要替换的位置
        max_replacements = valid_num_substitute.max()
        pos_aug = torch.zeros((len(valid_indices), max_replacements), 
                            dtype=torch.long, device=self.device)
        
        for i, (idx, n_sub) in enumerate(zip(valid_indices, valid_num_substitute)):
            positions = torch.randperm(len_seq[idx]).to(self.device)[:n_sub] + self.state_size - len_seq[idx]
            pos_aug[i, :n_sub] = positions
        
        # 获取需要替换的物品
        batch_indices = valid_indices.unsqueeze(1).expand(-1, max_replacements)
        items_to_replace = seq[batch_indices, pos_aug]

        # 批量处理所有替换物品的嵌入和向量量化
        valid_mask_replace = torch.arange(max_replacements, device=self.device).unsqueeze(0) < valid_num_substitute.unsqueeze(1)
        flat_items = items_to_replace[valid_mask_replace]
        
        if len(flat_items) > 0:
            item_embs = self.item_embeddings(flat_items)
            _, _, item_indices = self.rq(item_embs, torch.ones(len(flat_items), device=self.device))

        item_indices = item_indices[0]

        # 获取所有索引的映射关系
        all_indices = self.get_all_indices()
        self.all_indices_map = {}
        for idx in torch.unique(all_indices):
            self.all_indices_map[idx.item()] = torch.where(all_indices == idx)[0]

        # 为每个需要替换的物品找到替代品
        substitutes = torch.zeros_like(flat_items)
        for j, (orig_item, vq_idx) in enumerate(zip(flat_items, item_indices)):
            if vq_idx.item() in self.all_indices_map:
                candidates = self.all_indices_map[vq_idx.item()]
                substitutes[j] = candidates[torch.randint(0, len(candidates), (1,))]
            else:
                substitutes[j] = orig_item  # 如果没有找到候选，保持原物品
        
        # 更新序列
        seq_aug[batch_indices[valid_mask_replace], pos_aug[valid_mask_replace]] = substitutes
    
        return seq_aug


    def calculate_loss(self, seq, target):
        t = torch.randint(0, self.timesteps, (target.shape[0],)).to(self.device)
        x_0 = self.get_item_emb(target)
        x_noise = self.add_noise(x_0, t)
        x_predicted, rq_loss = self.forward(seq, x_noise, t, train_flag=True)
        item_emb_weight = self.item_embeddings.weight
        y = torch.matmul(x_predicted, item_emb_weight.transpose(0, 1))
        loss_diffu_value = self.CE_loss(y, target)

        
        seq_aug = self.augmentation(seq, ratio=self.ratio_substitute)
        #seq_aug = seq.clone()
        t_cl = torch.randint(0, self.timesteps, (target.shape[0],)).to(self.device)
        x_noise_cl = self.add_noise(x_0, t)
        x_cl, _ = self.forward(seq_aug, x_noise_cl, t_cl, train_flag=True)

        cl_loss = self.MSE_loss(x_predicted, x_cl)
        #print("CL_loss: ", cl_loss)
        
        
        loss = loss_diffu_value + self.lam_rqloss * rq_loss + self.lam_cl_loss * cl_loss
        return loss, cl_loss
    

    def add_noise(self, x, t):
        noise = torch.randn_like(x)
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].unsqueeze(-1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].unsqueeze(-1)
        return sqrt_alphas_cumprod_t * x + sqrt_one_minus_alphas_cumprod_t * noise


    def reverse_process(self, seq, x, t):
        x_0, _ = self.forward(seq, x, t)
        mean = self.posterior_mean_coef1[t].unsqueeze(-1) * x_0 + self.posterior_mean_coef2[t].unsqueeze(-1) * x
        
        if t[0] == 0:
            return mean
        else:
            noise = torch.randn_like(x)
            return mean + torch.sqrt(self.posterior_variance[t].unsqueeze(-1)) * noise


    def sample(self, seq):
        x = torch.randn(seq.shape[0], self.hidden_size, device=self.device)
        t = torch.ones(seq.shape[0]).to(self.device).long()
        for i in reversed(range(self.timesteps)):
            x = self.reverse_process(seq, x, t*i)

        return x


    def predict(self, seq):
        x_0 = self.sample(seq)

        item_emb_weight = self.item_embeddings.weight
        y = torch.matmul(x_0, item_emb_weight.transpose(0, 1))
        return y
    


