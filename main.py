import argparse
import torch
import os
import datetime
import pickle
from utils import *
from model.IQGARec import IQGARec

def parse_args():
    parser = argparse.ArgumentParser(description="Run supervised GRU.")

    parser.add_argument('--epoch', type=int, default=300,
                        help='Number of max epochs.')
    parser.add_argument('--data', nargs='?', default='beauty',
                        help='data directory')
    parser.add_argument('--batch_size', type=int, default=512,
                        help='Batch size.')
    parser.add_argument('--hidden_size', type=int, default=128,
                        help='Number of hidden factors, i.e., embedding size.')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate.')
    parser.add_argument('--num_heads', default=4, type=int)
    parser.add_argument('--num_blocks', default=4, type=int)
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--emb_dropout', default=0.3, type=float)
    parser.add_argument('--random_seed', default=0, type=float)
    parser.add_argument('--l2', default=0., type=float)

    parser.add_argument('--num_clusters', default=32, type=int)
    parser.add_argument('--lambda_uncertainty', default=0.02, type=float)
    parser.add_argument('--lambda_history', default=1, type=float)
    parser.add_argument('--lambda_intent', default=0.1, type=float)
    parser.add_argument('--lambda_rqloss', default=0.125, type=float)
    parser.add_argument('--lambda_cl_loss', default=15, type=float)
    parser.add_argument('--ratio_substitute', default=0.1, type=float)
    parser.add_argument('--n_stages', default=2, type=int)

    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--max_len', type=int, default=50,
                    help='seq size')
    parser.add_argument('--diffusion_steps', type=int, default=32,
                        help='timesteps for diffusion')
    parser.add_argument('--beta_end', type=float, default=0.02,
                        help='beta end of diffusion')
    parser.add_argument('--beta_start', type=float, default=0.0001,
                        help='beta start of diffusion')
    parser.add_argument('--beta_sche', nargs='?', default='trunc_lin',
                        help='')
    parser.add_argument('--noise_schedule', default='trunc_lin', help='Beta generation')
    parser.add_argument('--eval_interval', type=int, default=10, help='the number of epoch to eval')

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # load data
    data_directory = './data/' + args.data
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_directory = os.path.join('./save', current_time + args.data)
    os.mkdir(save_directory)
    with open(os.path.join(data_directory, 'dataset.pkl'), 'rb') as f:
        data_raw = pickle.load(f)
    item_num = len(data_raw['smap'])
    tra_data = Data_Train(data_raw['train'], args)
    val_data = Data_Val(data_raw['train'], data_raw['val'], args)
    test_data = Data_Test(data_raw['train'], data_raw['val'], data_raw['test'], args)
    tra_data_loader = tra_data.get_pytorch_dataloaders()
    val_data_loader = val_data.get_pytorch_dataloaders()
    test_data_loader = test_data.get_pytorch_dataloaders()
    short_middle_long = long_short(tra_data, 10, args.max_len)

    topk=[1,5,10,20]
    model = IQGARec(args.hidden_size, item_num, args.max_len, args.emb_dropout, args.dropout, args.lambda_uncertainty, 
                    args.lambda_history, args.lambda_intent, args.lambda_rqloss, args.lambda_cl_loss, args.ratio_substitute, args.n_stages, 
                      args.device, args.diffusion_steps, args.beta_start, args.beta_end, args.noise_schedule, 
                      args.num_clusters, args.num_heads, args.num_blocks).to(args.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)
    CE_loss = torch.nn.CrossEntropyLoss()
    now_best = 0

    # train
    for i in range(args.epoch):
        model.train()
        start_time_i = datetime.datetime.now()
        total_loss = 0
        total_cl_loss = 0
        for index, train_batch in enumerate(tra_data_loader):
            train_batch = [x.to(args.device) for x in train_batch]
            seq = train_batch[0]
            target = train_batch[1].squeeze()
            
            optimizer.zero_grad()

            loss, cl_loss = model.calculate_loss(seq, target)
            
            loss.backward()
            optimizer.step()

            total_loss += loss
            total_cl_loss += cl_loss

        
        print("the loss in %dth epoch is: %f " % (i, total_loss), end='')
        print("CL_Loss: ", total_cl_loss.item())
        lr_scheduler.step()

        over_time_i = datetime.datetime.now()  # 程序结束时间
        total_time_i = (over_time_i - start_time_i).total_seconds()
        print('total times: %s' % total_time_i)

        sd = calculate_variability(short_middle_long[0], short_middle_long[1], model, 20)
        print("SD: ", sd)

        if (i+1) % args.eval_interval == 0:

            model.eval()
            with torch.no_grad():
                # validate
                print('-------------------------- VAL PHRASE --------------------------')
                evaluate(model, val_data_loader, topk, args.device)

                # test
                print('-------------------------- TEST PHRASE --------------------------')
                result = evaluate(model, test_data_loader, topk, args.device)

                if result[4] > now_best:
                    now_best = result[4]
                    torch.save(model, os.path.join(save_directory, 'model.pth'))





# todo:
# 增加与非扩散模型的对比学习
# token-level diffusion
# 噪声加在中间层