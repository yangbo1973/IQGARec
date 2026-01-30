import torch.utils.data as data_utils
import torch
from torch import nn
import datetime
import numpy as np
import math
import torch.nn.functional as F

class TrainDataset(data_utils.Dataset):
    def __init__(self, id2seq, max_len):
        self.id2seq = id2seq
        self.max_len = max_len

    def __len__(self):
        return len(self.id2seq)

    def __getitem__(self, index):
        seq = self._getseq(index)
        labels = [seq[-1]]
        tokens = seq[:-1]
        tokens = tokens[-self.max_len:]
        mask_len = self.max_len - len(tokens)
        tokens = [0] * mask_len + tokens
        return torch.LongTensor(tokens), torch.LongTensor(labels)

    def _getseq(self, idx):
        return self.id2seq[idx]


class Data_Train():
    def __init__(self, data_train, args):
        self.u2seq = data_train
        self.max_len = args.max_len
        self.batch_size = args.batch_size
        self.split_onebyone()

    def split_onebyone(self):
        self.id_seq = {}
        self.id_seq_user = {}
        idx = 0
        for user_temp, seq_temp in self.u2seq.items():
            for star in range(len(seq_temp)-1):
                self.id_seq[idx] = seq_temp[:star+2]
                self.id_seq_user[idx] = user_temp
                idx += 1

    def get_pytorch_dataloaders(self):
        dataset = TrainDataset(self.id_seq, self.max_len)
        return data_utils.DataLoader(dataset, batch_size=self.batch_size, shuffle=True, pin_memory=True)


class ValDataset(data_utils.Dataset):
    def __init__(self, u2seq, u2answer, max_len):
        self.u2seq = u2seq
        self.users = sorted(self.u2seq.keys())
        self.u2answer = u2answer
        self.max_len = max_len

    def __len__(self):
        return len(self.users)

    def __getitem__(self, index):
        user = self.users[index]
        seq = self.u2seq[user]
        answer = self.u2answer[user]
        seq = seq[-self.max_len:]
        padding_len = self.max_len - len(seq)
        seq = [0] * padding_len + seq
        return torch.LongTensor(seq),  torch.LongTensor(answer)


class Data_Val():
    def __init__(self, data_train, data_val, args):
        self.batch_size = args.batch_size
        self.u2seq = data_train
        self.u2answer = data_val
        self.max_len = args.max_len
        

    def get_pytorch_dataloaders(self):
        dataset = ValDataset(self.u2seq, self.u2answer, self.max_len)
        dataloader = data_utils.DataLoader(dataset, batch_size=self.batch_size, shuffle=False, pin_memory=True)
        return dataloader


class TestDataset(data_utils.Dataset):
    def __init__(self, u2seq, u2_seq_add, u2answer, max_len):
        self.u2seq = u2seq
        self.u2seq_add = u2_seq_add
        self.users = sorted(self.u2seq.keys())
        self.u2answer = u2answer
        self.max_len = max_len

    def __len__(self):
        return len(self.users)

    def __getitem__(self, index):
        user = self.users[index]
        seq = self.u2seq[user] + self.u2seq_add[user]
        # seq = self.u2seq[user]
        answer = self.u2answer[user]
        seq = seq[-self.max_len:]
        padding_len = self.max_len - len(seq)
        seq = [0] * padding_len + seq
        return torch.LongTensor(seq), torch.LongTensor(answer)


class Data_Test():
    def __init__(self, data_train, data_val, data_test, args):
        self.batch_size = args.batch_size
        self.u2seq = data_train
        self.u2seq_add = data_val
        self.u2answer = data_test
        self.max_len = args.max_len

    def get_pytorch_dataloaders(self):
        dataset = TestDataset(self.u2seq, self.u2seq_add, self.u2answer, self.max_len)
        dataloader = data_utils.DataLoader(dataset, batch_size=self.batch_size, shuffle=False, pin_memory=True)
        return dataloader


class CHLSDataset(data_utils.Dataset):
    def __init__(self, data, max_len):
        self.data = data
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        data_temp = self.data[index]
        seq = data_temp[:-1]
        answer = [data_temp[-1]]
        seq = seq[-self.max_len:]
        padding_len = self.max_len - len(seq)
        seq = [0] * padding_len + seq
        return torch.LongTensor(seq), torch.LongTensor(answer)


class Data_CHLS():
    def __init__(self, data, args):
        self.batch_size = args.batch_size
        self.max_len = args.max_len
        self.data = data

    def get_pytorch_dataloaders(self):
        dataset = CHLSDataset(self.data, self.max_len)
        dataloader = data_utils.DataLoader(dataset, batch_size=self.batch_size, shuffle=False, pin_memory=True)
        return dataloader


def evaluate(model, data_loader, k, device, pretrain=False):
    start_time = datetime.datetime.now()  # 程序开始时间

    hr_list = torch.zeros(len(k)).to(device)
    ndcg_list = torch.zeros(len(k)).to(device)
    user_num = 0

    for _, batch in enumerate(data_loader):
        batch = [x.to(device) for x in batch]
        item_seq = batch[0]
        target = batch[1]

        y = model.predict(item_seq)
        # y = model.predict(item_seq, pretrain)
        #scores_rec, rep_diffu, _, _, _, _ = model(batch[0], batch[1], train_flag=False)
        #y = model.diffu_rep_pre(rep_diffu)
            
        _, result = torch.topk(y, k[-1], dim=-1) # result: [batch, k[-1]]
        calculate_hit_ndcg(result, k, target, hr_list, ndcg_list, device)
        user_num += target.shape[0]

    for i in range(len(k)):
        hit=hr_list[i]/ user_num
        ndcg=ndcg_list[i]/user_num
        print("k | hit@k | ndcg@k")
        print("%d %.8f %.8f" % (k[i], hit, ndcg))
    over_time = datetime.datetime.now()  # 程序结束时间
    total_time = (over_time - start_time).total_seconds()
    print('total times: %s' % total_time)
    return hr_list[0] / user_num, ndcg_list[0] / user_num, \
           hr_list[1] / user_num, ndcg_list[1] / user_num, \
           hr_list[2] / user_num, ndcg_list[2] / user_num,

def calculate_hit_ndcg(sorted_list,topk,true_items,hit_purchase,ndcg_purchase, device):
    for i in range(len(topk)):
        rec_list = sorted_list[:, :topk[i]] # [batch, topk[-1]]
        # for j in range(len(true_items)):
        #     if true_items[j] in rec_list[j] :
        #         rank = torch.argwhere(rec_list[j] == true_items[j])[0][0] + 1
        #         hit_purchase[i] += 1.0
        #         ndcg_purchase[i] += 1.0 / torch.log2(rank + 1)

        mask = (true_items == rec_list) # [batch, topk[-1]]
        rank = mask.nonzero()[:, -1]
        if rank.shape[0] != 0:
            hit_purchase[i] += rank.shape[0]
            ndcg_purchase[i] += torch.sum(1 / torch.log2(rank + 2), dim=-1)
        
        
def linear_beta_schedule(timesteps, beta_start, beta_end):
    beta_start = beta_start
    beta_end = beta_end
    return torch.linspace(beta_start, beta_end, timesteps)

def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)

def exp_beta_schedule(timesteps, beta_min=0.1, beta_max=10):
    x = torch.linspace(1, 2 * timesteps + 1, timesteps)
    betas = 1 - torch.exp(- beta_min / timesteps - x * 0.5 * (beta_max - beta_min) / (timesteps * timesteps))
    return betas

def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].
    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.

    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    
    res = torch.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)


def get_named_beta_schedule(schedule_name, num_diffusion_timesteps):
    """
    Get a pre-defined beta schedule for the given name.
    The beta schedule library consists of beta schedules which remain similar in the limit of num_diffusion_timesteps. Beta schedules may be added, but should not be removed or changed once they are committed to maintain backwards compatibility.
    """
    if schedule_name == "linear":
        # Linear schedule from Ho et al, extended to work for any number of
        # diffusion steps.
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)
    elif schedule_name == "cosine":
        return betas_for_alpha_bar(num_diffusion_timesteps, lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,)
    elif schedule_name == 'sqrt':
        return betas_for_alpha_bar(num_diffusion_timesteps,lambda t: 1-np.sqrt(t + 0.0001),  )
    elif schedule_name == "trunc_cos":
        return betas_for_alpha_bar_left(num_diffusion_timesteps, lambda t: np.cos((t + 0.1) / 1.1 * np.pi / 2) ** 2,)
    elif schedule_name == 'trunc_lin':
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001 + 0.01
        beta_end = scale * 0.02 + 0.01
        if beta_end > 1:
            beta_end = scale * 0.001 + 0.01
        return np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)
    elif schedule_name == 'pw_lin':
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001 + 0.01
        beta_mid = scale * 0.0001  #scale * 0.02
        beta_end = scale * 0.02
        first_part = np.linspace(beta_start, beta_mid, 10, dtype=np.float64)
        second_part = np.linspace(beta_mid, beta_end, num_diffusion_timesteps - 10 , dtype=np.float64)
        return np.concatenate([first_part, second_part])
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function, which defines the cumulative product of (1-beta) over time from t = [0,1].
    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and produces the cumulative product of (1-beta) up to that part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):  ## 2000
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


def betas_for_alpha_bar_left(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function, but shifts towards left interval starting from 0
    which defines the cumulative product of (1-beta) over time from t = [0,1].

    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    betas.append(min(1-alpha_bar(0), max_beta))
    for i in range(num_diffusion_timesteps-1):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


def space_timesteps(num_timesteps, section_counts):
    """
    Create a list of timesteps to use from an original diffusion process,
    given the number of timesteps we want to take from equally-sized portions
    of the original process.

    For example, if there's 300 timesteps and the section counts are [10,15,20]
    then the first 100 timesteps are strided to be 10 timesteps, the second 100
    are strided to be 15 timesteps, and the final 100 are strided to be 20.

    If the stride is a string starting with "ddim", then the fixed striding
    from the DDIM paper is used, and only one section is allowed.

    :param num_timesteps: the number of diffusion steps in the original
                          process to divide up.
    :param section_counts: either a list of numbers, or a string containing
                           comma-separated numbers, indicating the step count
                           per section. As a special case, use "ddimN" where N
                           is a number of steps to use the striding from the
                           DDIM paper.
    :return: a set of diffusion steps from the original process to use.
    """
    if isinstance(section_counts, str):
        if section_counts.startswith("ddim"):
            desired_count = int(section_counts[len("ddim") :])
            for i in range(1, num_timesteps):
                if len(range(0, num_timesteps, i)) == desired_count:
                    return set(range(0, num_timesteps, i))
            raise ValueError(
                f"cannot create exactly {num_timesteps} steps with an integer stride"
            )
        section_counts = [int(x) for x in section_counts.split(",")]
    size_per = num_timesteps // len(section_counts)
    extra = num_timesteps % len(section_counts)
    start_idx = 0
    all_steps = []
    for i, section_count in enumerate(section_counts):
        size = size_per + (1 if i < extra else 0)
        if size < section_count:
            raise ValueError(
                f"cannot divide section of {size} steps into {section_count}"
            )
        if section_count <= 1:
            frac_stride = 1
        else:
            frac_stride = (size - 1) / (section_count - 1)
        cur_idx = 0.0
        taken_steps = []
        for _ in range(section_count):
            taken_steps.append(start_idx + round(cur_idx))
            cur_idx += frac_stride
        all_steps += taken_steps
        start_idx += size
    return set(all_steps)

    

def long_short(tra_data, k, max_len):
    seqs = tra_data.u2seq
    len_seq = [len(seqs[i]) for i in seqs]
    len_seq = np.array(len_seq)

    len_sort = np.argsort(len_seq)
    shortest = [seqs[i][:-1] for i in len_sort[:k]]
    shortest_target = torch.tensor([seqs[i][-1] for i in len_sort[:k]], dtype=torch.int32)
    middle = [seqs[i][:-1] for i in len_sort[len(len_sort)//2 - k//2 : len(len_sort)//2 + k//2]]
    middle_target = torch.tensor([seqs[i][-1] for i in len_sort[len(len_sort)//2 - k//2 : len(len_sort)//2 + k//2]], dtype=torch.int32)
    longest = [seqs[i][:-1] for i in len_sort[-k:]]
    longest_target = torch.tensor([seqs[i][-1] for i in len_sort[-k:]], dtype=torch.int32)

    shortest = seqs_to_tensor(shortest, max_len)
    middle = seqs_to_tensor(middle, max_len)
    longest = seqs_to_tensor(longest, max_len)

    return torch.concat([shortest, middle, longest], dim=0), torch.concat([shortest_target, middle_target, longest_target])



def seqs_to_tensor(seqs, max_len):
    processed_seqs = []
    for seq in seqs:
        seq_len = len(seq)
        
        if seq_len > max_len:
            processed_seq = seq[-max_len:]
        elif seq_len < max_len:
            padding = [0] * (max_len - seq_len)
            processed_seq = padding + seq
            
        processed_seqs.append(processed_seq)

    result = torch.tensor(processed_seqs, dtype=torch.int32)
    return result


def calculate_variability(seqs, target, model, k):
    seqs = seqs.to(model.device)
    target = target.to(model.device)
    result = []

    for _ in range(k):
        t = torch.randint(0, model.timesteps, (seqs.shape[0],)).to(model.device)
        x_0 = model.get_item_emb(target)
        x_noise = model.add_noise(x_0, t)
        x_predicted, _ = model.forward(seqs, x_noise, t, train_flag=True)
        result.append(x_predicted)

    mean = torch.zeros_like(result[0])
    for i in range(k):
        mean += result[i]
    mean /= k

    sd = torch.zeros_like(result[0])
    for i in range(k):
        sd += (result[i] - mean) ** 2
    sd /= (k-1)
    sd = torch.sqrt(sd)
    short, mid, long = torch.chunk(sd, chunks=3, dim=0)
    short = torch.sum(short) / short.shape[0]
    mid = torch.sum(mid) / mid.shape[0]
    long = torch.sum(long) / long.shape[0]
    total = torch.sum(sd) / sd.shape[0]

    return short.item(), mid.item(), long.item(), total.item()
