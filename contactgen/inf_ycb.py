import os
import random
import argparse
import numpy as np
import trimesh
import torch
from tqdm import tqdm
from omegaconf import OmegaConf
from contactgen.utils.cfg_parser import Config
from contactgen.model import ContactGenModel
from contactgen.datasets.eval_dataset_ycb import TestSetYCB

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ContactGen Eval')
    parser.add_argument('--checkpoint', default='checkpoint/checkpoint.pt', type=str, help='exp dir')
    parser.add_argument('--n_samples', default=64, type=int, help='number of samples per object')
    parser.add_argument('--save_root', default='../human_contact/ycb', type=str, help='result save root')
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    
    args = parser.parse_args()
    os.makedirs(args.save_root, exist_ok=True)
    
    torch.cuda.manual_seed_all(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    cfg_path = "contactgen/configs/default.yaml"
    cfg = Config(default_cfg_path=cfg_path)
    device = torch.device('cuda')
    model = ContactGenModel(cfg).to(device)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    model.eval()
    
    dataset = TestSetYCB()
    
    # List to store metadata for all samples
    metadata = []
    
    for idx, input in tqdm(enumerate(dataset)):
        obj_name = input['obj_name']
        obj_verts = torch.from_numpy(input['obj_verts']).unsqueeze(dim=0).float().to(device).repeat(args.n_samples, 1, 1)
        obj_vn = torch.from_numpy(input['obj_vn']).unsqueeze(dim=0).float().to(device).repeat(args.n_samples, 1, 1)
        
        with torch.no_grad():
            sample_results = model.sample(obj_verts, obj_vn)
        contacts_object, partition_object, uv_object = sample_results
        
        # Process each sample
        for i in range(args.n_samples):
            metadata_entry = {
                'grasp_id': idx * args.n_samples + i,  # Unique grasp_id
                'obj_name': obj_name,
                'obj_verts': obj_verts[i].cpu(),
                'obj_vn': obj_vn[i].cpu(),
                'robot_name': 'mano',
                'obj_cmap': contacts_object[i].cpu(),
                'obj_partition': partition_object[i].cpu(),
                'obj_uv': uv_object[i].cpu(),
            }
            metadata.append(metadata_entry)
    
    # Save the contact map dataset
    save_path = os.path.join(args.save_root, 'cmap_dataset.pt')
    torch.save(metadata, save_path)
    
    print(f"Saved {len(metadata)} samples to {save_path}")
    print("all done")