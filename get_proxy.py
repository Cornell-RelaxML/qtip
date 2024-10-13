import torch

for i in range(80):
    for l in ['q', 'k', 'v', 'o', 'up', 'gate', 'down']:
        err = torch.load(f'/scratch/alberttseng/ckpt/3.1_70b_2bit/{i}_{l}.pt')['proxy_err']
        if err > 0.01:
            print(i, l, err)
        
