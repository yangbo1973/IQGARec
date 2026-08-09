import torch
from .module import *
import torch.nn.functional as F
from utils import *

class DiffuRec(torch.nn.Module):
    def __init__(self, hidden_size, item_num, state_size, emb_dropout_rate, dropout_rate, delta, device
                 , timesteps, beta_start, beta_end, beta_sche, num_heads=1, num_blocks=1):
        super(DiffuRec, self).__init__()

        self.device = device
        self.hidden_size = hidden_size
        self.item_num = item_num
        self.state_size = state_size
        self.dropout_rate = dropout_rate
        self.emb_dropout_rate = emb_dropout_rate
        self.num_heads = num_heads
        self.num_blocks = num_blocks
        self.delta = torch.tensor(delta).to(device)

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



    def get_item_emb(self, item):
        return self.item_embeddings(item)


    def forward(self, seq, x_t, step, train_flag=False):

        # seq: [batch, len]
        # x_t: [batch, dim]
        # step: [batch]

        item_emb = self.item_embeddings(seq)
        #pos_id = torch.arange(self.state_size, dtype=torch.long, device=self.device).unsqueeze(0)
        #item_emb += self.pos_embeddings(pos_id)
        item_emb = self.emb_dropout(item_emb)
        item_emb = self.emb_layernorm(item_emb)

        step_emb = self.step_mlp(step) # [bacth, dim]
        #lam = self.delta + torch.randn(seq.shape[0], self.state_size, device=self.device) * torch.sqrt(self.delta) # [batch, len]
        lam = torch.normal(mean=torch.full_like(item_emb, self.delta), std=torch.full_like(item_emb, self.delta)).to(self.device)
        z = item_emb + lam * (x_t.unsqueeze(1) + step_emb.unsqueeze(1))

        '''
        timeline_mask = (seq == 0)
        z *= ~timeline_mask.unsqueeze(-1)
        for i in range(self.num_blocks):
            z = self.transformer_block[i](z)
            z *= ~timeline_mask.unsqueeze(-1)
        z = self.last_layernorm(z)

        x_0 = z[:, -1, :]
        '''
        mask_seq = (seq>0).float()
        for i in range(self.num_blocks):
            z = self.att[i](z, mask_seq)
        rep_diffu = self.last_layernorm(self.dropout(z))
        out = rep_diffu[:, -1, :]
        return out, None
    

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
    

if __name__ == "__main__":

    batch_size = 256
    hidden_size = 64
    max_len = 50
    device = 'cuda'

    zero = torch.zeros(batch_size, 25)
    non_zero = torch.arange(1, 26).expand_as(zero)
    seq = torch.cat((zero, non_zero), dim=1).to(device).long()
    x_t = torch.randn(batch_size, hidden_size).to(device)
    step = torch.ones(batch_size, dtype=torch.long).to(device)

    model = DiffuRec(hidden_size, 1000, max_len, 0.1, 1.0, device).to(device)

    y = model(seq, x_t, step)



