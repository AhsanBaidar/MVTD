import argparse
import torch
from thop import profile
from thop.utils import clever_format
import time
import importlib
from torch import nn
import numpy as np
import os

current_cwd = os.getcwd()


def parse_args():
    """
    args for training.
    """
    parser = argparse.ArgumentParser(description='Parse args for training')
    # for train
    parser.add_argument('--script', type=str, default='mcitrack',
                        help='training script name')
    parser.add_argument('--config', type=str, default='mcitrack_b224', help='yaml configure file name')
    args = parser.parse_args()

    return args


def get_complexity_MHA(m:nn.MultiheadAttention, x, y):
    """(L, B, D): sequence length, batch size, dimension"""
    d_mid = m.embed_dim
    query, key, value = x[0], x[1], x[2]
    Lq, batch, d_inp = query.size()
    Lk = key.size(0)
    """compute flops"""
    total_ops = 0
    # projection of Q, K, V
    total_ops += d_inp * d_mid * Lq * batch  # query
    total_ops += d_inp * d_mid * Lk * batch * 2  # key and value
    # compute attention
    total_ops += Lq * Lk * d_mid * 2
    m.total_ops += torch.DoubleTensor([int(total_ops)])


def evaluate(model, template_list,search_list,template_anno_list,hanning,neck_h_state,enc_opt,neck_out, bs):
    """Compute FLOPs, Params, and Speed"""
    custom_ops = {nn.MultiheadAttention: get_complexity_MHA}
    # encoder
    macs1, params1 = profile(model, inputs=(template_list, search_list, template_anno_list,None,None,None,"encoder"), custom_ops=custom_ops ,verbose=True)
    macs3, params3 = profile(model, inputs=(template_list, search_list, template_anno_list,enc_opt,neck_h_state,None,"neck"), custom_ops=custom_ops ,verbose=True)
    macs, params = clever_format([macs1+macs3, params1+params3], "%.3f")
    print('encoder macs is ', macs)
    print('encoder params is ', params)
    # decoder
    macs2, params2 = profile(model, inputs=(template_list, search_list, template_anno_list,enc_opt,neck_h_state,neck_out,"decoder"), custom_ops=custom_ops, verbose=True)
    macs, params = clever_format([macs2, params2], "%.3f")
    print('decoder macs is ', macs)
    print('decoder params is ', params)
    # the whole model
    macs, params = clever_format([macs1 + macs2 + macs3, params1 + params2 +params3], "%.3f")
    print('overall macs is ', macs)
    print('overall params is ', params)

    '''Speed Test'''
    T_w = 50
    T_t = 500
    print("testing speed ...")
    with torch.no_grad():
        # overall
        for i in range(T_w):
            _ = model(template_list, search_list, template_anno_list,None,None,None,"encoder")
            _ = model(template_list, search_list, template_anno_list, enc_opt,neck_h_state, None, "neck")
            _ = model(template_list, search_list, template_anno_list, enc_opt, neck_h_state, neck_out, "decoder")
        start = time.time()
        for i in range(T_t):
            _ = model(template_list, search_list, template_anno_list,None,None,None,"encoder")
            _ = model(template_list, search_list, template_anno_list, enc_opt,neck_h_state, None, "neck")
            _ = model(template_list, search_list, template_anno_list, enc_opt, neck_h_state, neck_out, "decoder")
        end = time.time()
        avg_lat = (end - start) / (T_t * bs)
        print("The average overall latency is %.2f ms" % (avg_lat * 1000))



def get_data(bs, sz):
    img_patch = torch.randn(bs, 3, sz, sz)
    return img_patch

if __name__ == "__main__":
    device = "cuda:0"
    torch.cuda.set_device(device)
    # Compute the Flops and Params of our STARK-S model
    args = parse_args()
    '''update cfg'''

    yaml_fname = current_cwd  + '/experiments/%s/%s.yaml' % (args.script, args.config)
    config_module = importlib.import_module('lib.config.%s.config' % args.script)
    cfg = config_module.cfg
    config_module.update_config_from_file(yaml_fname)
    '''set some values'''
    bs = 1
    z_sz = cfg.TEST.TEMPLATE_SIZE
    x_sz = cfg.TEST.SEARCH_SIZE
    hanning = None
    '''import mcitrack network module'''
    model_module = importlib.import_module('lib.models.mcitrack')
    model_constructor = model_module.build_mcitrack
    model = model_constructor(cfg)
    # get the template and search
    template = get_data(bs, z_sz)
    search = get_data(bs, x_sz)
    neck_h_state = [None] * cfg.MODEL.NECK.N_LAYERS
    # transfer to device
    model = model.to(device)
    template = template.to(device)
    search = search.to(device)
    model.eval()
    # evaluate the model properties
    template_list = [template] * 5
    template_anno_list = [torch.tensor([[0.5,0.5,0.5,0.5]]).to(device)] *5
    search_list = [search]
    enc_opt = model(template_list,search_list,template_anno_list,mode="encoder")
    encoder_out, neck_out, neck_h_state = model(enc_opt=enc_opt,neck_h_state=neck_h_state,mode="neck")
    evaluate(model, template_list,search_list,template_anno_list, hanning, neck_h_state=neck_h_state,enc_opt=enc_opt,neck_out=neck_out,bs=bs)

