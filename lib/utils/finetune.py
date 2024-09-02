import torch
from torch import nn


def save_linear(module, path):
    saved_layer = torch.load(path, map_location=torch.device('cpu'))
    saved_layer['SU'] = module.SU.data.to(torch.float16)
    saved_layer['SV'] = (
        module.SV.data.float() /
        saved_layer['Wscale'].float().to(module.SV.data.device)).cpu()
    if module.tlut is not None:
        saved_layer['tlut'] = module.tlut.data.to(torch.float16)
    torch.save(saved_layer, path)


def calculate_mse_loss(layer, dataloader, device):
    layer.eval()
    total_loss = 0
    ct = 0
    position_ids = None
    with torch.no_grad():
        for source, target in dataloader:
            if position_ids is None:
                position_ids = torch.arange(source.shape[1],
                                            device=device).unsqueeze(0)
            target = target.to(device, non_blocking=True)
            total_loss += nn.MSELoss()(layer(source.to(device),
                                             position_ids=position_ids)[0],
                                       target)
            ct += 1
    layer.train()
    return (total_loss / ct).cpu().item()


def calculate_ce_loss_model(model, dataloader, start_dev, in_q, out_q):
    model.eval()
    total_loss = 0
    ct = 0
    with torch.no_grad():
        for source, target in dataloader:
            in_q.put(source)
            output = model(source.to(start_dev))['logits'][:, :-1].contiguous()
            output = output.view(-1, output.shape[-1])
            target = out_q.get().to(output.device)
            target = target.view(-1, target.shape[-1])
            total_loss += nn.CrossEntropyLoss()(output, target)
            ct += 1
    model.train()
    return (total_loss / ct).cpu().item()
