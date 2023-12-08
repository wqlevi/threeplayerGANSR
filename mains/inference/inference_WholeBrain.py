import torch
import argparse
from torch.nn.functional import interpolate
import numpy as np
import nibabel as nb
import sys,os
from glob import glob
sys.path.append('../')
from inference import load_model
from utils import add_noise

def load_ln_G(opt):
    # Why is "device = cpu": memory requirement > GPU mem
    if opt['use_ln_model']:
        G = load_model(opt, state_key='state_dict', use_ln=True,device='cpu')
    else:
        G = load_model(opt, state_key='Gnet_state_dict', device='cpu')
    return G

def produce_whole(opt, nii_name, model):
    nii = nb.load(nii_name)
    ts = torch.Tensor(add_noise(nii.get_fdata(), opt['noise_level']))[None,None]
    y = model(ts)
    new_nii = nb.Nifti1Image(y.squeeze().detach().numpy(), np.eye(4)) # save as SR img with identical matrix as default affine
    save_name = opt['ckp_path'].rsplit("/",1)[0]+"/saved_nii"
    os.makedirs(save_name, exist_ok=True)
    file_name = nii_name.split("/")[-1].split(".")[0]
    nb.save(new_nii, "%s/%s_sr_%s.nii.gz"%(save_name, file_name, str(opt['noise_level']).replace(".","")))

def main(opt:dict):
    G = load_ln_G(opt)

    if opt['single_img']: # input a single image with complete directory
        produce_whole(opt, opt['data_path'], G)
    else:                 # input is a directory of the data's folder 
        for f in glob(opt['data_path']+"*.nii.gz"):
            produce_whole(opt, f, G)

if __name__ == '__main__':
    parser = argparse.ArgumentParser() 
    parser.add_argument('--data_path', default = "/big_data/qi1/HCP_3T_v1/")
    parser.add_argument('--ckp_path', default = "/big_data/qi1/saved_models/DWT_Unet_D/DWT_Unet_D_Crop_15.ckpt")
    parser.add_argument('--model_type', default = "model_DISGAN")
    parser.add_argument('--G_type', default = "Generator")
    parser.add_argument('--model_name', default = "generator")
    parser.add_argument('--use_ln_model',action='store_true')
    parser.add_argument('--single_img',action='store_true')
    parser.add_argument('--noise_level',type=float, default=0.)
    opt = parser.parse_args()
    opt_dict = vars(opt)
    main(opt_dict)
