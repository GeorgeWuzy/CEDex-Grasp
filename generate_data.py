
import argparse
import os.path
import time
import sys
import shutil
import torch
import json
from datetime import datetime
from utils_model.AdamGrasp import AdamGrasp

DATASET_CONFIG = {
    'contactdb': {
        'path': 'human_contact/contactdb/cmap_dataset.pt',
        'name': 'ContactDB'
    },
    'ycb': {
        'path': 'human_contact/ycb/cmap_dataset.pt',
        'name': 'YCB'
    },
}

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--robot_name', default='barrett', type=str,
                        choices=['ezgripper', 'robotiq_3finger',
                                 'barrett', 'allegro', 'shadowhand', 'leaphand'])
    parser.add_argument('--dataset', default='contactdb', type=str,
                        choices=['contactdb', 'ycb'],
                        help='Dataset to use for optimization')
    parser.add_argument('--num_particles', default=64, type=int)
    parser.add_argument('--save_top_k', default=16, type=int,
                        help='Number of top grasps to save based on energy')
    parser.add_argument('--learning_rate', default=1e-2, type=float)
    parser.add_argument('--max_iter', default=200, type=int)

    args_ = parser.parse_args()
    tag = str(time.time())
    return args_, tag


def load_and_filter_datasets(dataset_name):
    
    config = DATASET_CONFIG[dataset_name]
    
    dataset_path = config['path']
    display_name = config['name']
    
    print(f"Selected dataset: {display_name}")
    print(f"Loading from: {dataset_path}")
    print("Processing ALL objects from the dataset")
    
    dataset = torch.load(dataset_path)
    print(f"Loaded {display_name} dataset with {len(dataset)} samples")
    
    for data_sample in dataset:
        data_sample['dataset_type'] = display_name

    print(f"Total samples: {len(dataset)}")
    
    return dataset

def extract_top_k_grasps(record, robot_name, object_name, dataset_name, top_k=4):
    q_tra, energy = record
    final_q = q_tra.detach()  # [num_particles, joint_dim]
    final_energy = energy.detach()  # [num_particles]
    sorted_indices = torch.argsort(final_energy)
    top_k_indices = sorted_indices[:top_k]

    top_k_grasps = []
    for i, idx in enumerate(top_k_indices):
        grasp_dict = {
            'q': final_q[idx].cpu(),
            'object_name': f"{dataset_name}+{object_name}",
            'robot_name': robot_name
        }
        top_k_grasps.append(grasp_dict)
    
    return top_k_grasps

if __name__ == '__main__':
    torch.set_printoptions(precision=4, sci_mode=False, edgeitems=8)
    args, time_tag = get_parser()
    print(args)
    print(f'Starting dataset generation...')

    robot_name = args.robot_name
    dataset_name = args.dataset
    save_top_k = args.save_top_k
    
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_logs_dir = f'logs/dataset_generation_{current_time}'
    robot_logs_dir = os.path.join(base_logs_dir, f'{robot_name}_{dataset_name}')
    grasp_dataset_dir = os.path.join(robot_logs_dir, 'generated_grasps')
    
    os.makedirs(base_logs_dir, exist_ok=True)
    os.makedirs(robot_logs_dir, exist_ok=True)
    os.makedirs(grasp_dataset_dir, exist_ok=True)

    # 保存运行配置
    config_info = {
        'time': current_time,
        'command': ' '.join(sys.argv),
        'dataset': dataset_name,
        'robot': robot_name,
        'save_top_k': save_top_k,
        'max_iter': args.max_iter,
        'num_particles': args.num_particles,
        'learning_rate': args.learning_rate,
    }
    
    with open(os.path.join(base_logs_dir, 'generation_config.json'), 'w') as f:
        json.dump(config_info, f, indent=2)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = load_and_filter_datasets(dataset_name)

    # init model
    model = AdamGrasp(robot_name=robot_name,
                      num_particles=args.num_particles, init_rand_scale=0.5, max_iter=args.max_iter,
                      steps_per_iter=1, learning_rate=args.learning_rate, device=device)
    
    processed_count = 0
    skipped_count = 0
    total_grasps_generated = 0
    
    total_processing_time = time.time()  # Starting time for whole processing

    for idx, data_sample in enumerate(dataset):
        object_name = data_sample['obj_name']
        dataset_type = data_sample['dataset_type']
        # Start timing for processing
        start_time = time.time()
        
        object_point_cloud = data_sample['obj_verts'].to(device)
        object_normals = data_sample['obj_vn'].to(device)
        contact_map_value = data_sample['obj_cmap'].to(device)
        contact_partition = data_sample['obj_partition'].to(device)
        contact_uv = data_sample['obj_uv'].to(device)
        human_hand_verts = None
        human_hand_parts = None
        
        print(f"\nProcessing {idx+1}/{len(dataset)}: {dataset_type}:{object_name} (processed: {processed_count})")
        
        # Combine point cloud, normals, and contact map
        contact_map_goal = torch.cat([object_point_cloud, object_normals, contact_map_value], dim=1).to(device)

        # Run the optimization
        running_name = f'{dataset_name}_{object_name}_{idx}'
        time1 = time.time()
        record = model.run_adam(contact_map_goal=contact_map_goal, 
                                contact_part=contact_partition,
                                running_name=running_name)
        print(time.time()-time1)
        # Extract top-k grasps
        top_k_grasps = extract_top_k_grasps(record, robot_name, object_name, dataset_name, save_top_k)
        
        # Save grasp dataset
        saving_start_time = time.time()
        save_filename = f'{dataset_name}_{object_name}_{idx}_top{save_top_k}.pt'
        save_path = os.path.join(grasp_dataset_dir, save_filename)
        torch.save(top_k_grasps, save_path)
        print(f"Saved {len(top_k_grasps)} grasps to {save_path}")
        
        processed_count += 1
        total_grasps_generated += len(top_k_grasps)
        
        # Print times
        print(f"Successfully generated {len(top_k_grasps)} grasps for {object_name}")
    # After the loop, print total processing time
    total_processing_time = time.time() - total_processing_time
    print(f'Total Processing Time for all samples: {total_processing_time:.4f} seconds')