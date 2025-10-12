import os
import sys
import json
import argparse
import warnings
from termcolor import cprint
import trimesh as tm  # Ensure trimesh library is installed
from validation.isaac_validator import IsaacValidator  # IsaacGym must be imported before PyTorch
import torch
from utils.hand_model import create_hand_model
from utils.rotation import q_rot6d_to_q_euler, q_euler_to_q_rot6d
import trimesh.sample

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)
OBJ_DIR = 'data/object'

def load_object_mesh(dataset_type, object_name):
    # Load the object mesh based on dataset type
    if dataset_type == 'contactdb':
        object_mesh_path = os.path.join(OBJ_DIR, f'contactdb/{object_name}/{object_name}.stl')
    elif dataset_type == 'ycb':
        models_directory = os.path.join(OBJ_DIR, 'ycb')
        object_mesh_path = None
        for folder in os.listdir(models_directory):
            if object_name in folder:
                object_mesh_path = os.path.join(models_directory, folder, 'google_16k', 'nontextured.stl')
                break

    object_mesh = tm.load(object_mesh_path)
    if isinstance(object_mesh, tm.Scene):
        object_mesh = list(object_mesh.geometry.values())[0]
    
    print(f"Successfully loaded object: {object_name} from {object_mesh_path}")
    return object_mesh

def compute_penetrations(opt_q, object_mesh, robot_name, device):  
    # Compute penetration values for each joint configuration
    npts_object = 512
    object_point_cloud, faces_indices = trimesh.sample.sample_surface(mesh=object_mesh, count=npts_object)  
    object_normal_cloud = torch.tensor([object_mesh.face_normals[x] for x in faces_indices]).float().to(device)  
    object_point_cloud = torch.Tensor(object_point_cloud).float().to(device)  

    hand_model = create_hand_model(robot_name, device=device)  
    hand_model.update_status(opt_q.clone())  # Update hand's joint configuration  
    hand_surface_points_ = hand_model.get_transformed_links_pc(q=opt_q.clone())[:, :3]  
    hand_surface_points_expanded = hand_surface_points_.unsqueeze(1)  # (npts_hand, 1, 3)
    object_point_cloud_expanded = object_point_cloud.unsqueeze(0)  # (1, npts_object, 3)
    hand_object_dist = (hand_surface_points_expanded - object_point_cloud_expanded).norm(dim=2)  # (npts_hand, npts_object)
    hand_object_dist, hand_object_indices = hand_object_dist.min(dim=1)
    hand_object_points = object_point_cloud[hand_object_indices]  
    hand_object_normal = object_normal_cloud[hand_object_indices]  
    hand_object_signs = ((hand_object_points - hand_surface_points_) * hand_object_normal).sum(dim=1)  
    hand_object_signs = (hand_object_signs > 0).float()  
    penetration_values = hand_object_signs * hand_object_dist  
    return penetration_values  

def isaac_main(
    object_name: str,
    batch_size: int,
    q_batch: torch.Tensor = None,
    grasps_data: list = None,
    use_gui: bool = False,
    evaluate_diversity: bool = False,
):

    urdf_assets_meta = json.load(open('data/urdf/urdf_assets_meta_extended.json'))
    # Extract the robot name from the input file
    robot_name = os.path.basename(args.input_file).split('.')[0]
    
    # Retrieve robot URDF path based on the extracted robot name
    robot_urdf_path = urdf_assets_meta['urdf_path'][robot_name]
    
    dataset_name = object_name.split('+')[0]
    object_name = object_name.split('+')[1]
    
    # Load the object URDF file
    if dataset_name.lower() == 'contactdb':
        object_urdf_path = os.path.join(OBJ_DIR, f'contactdb/{object_name}/{object_name}.urdf')

    elif dataset_name.lower() == 'ycb':
        models_directory = os.path.join(OBJ_DIR, 'ycb')
        template_urdf_path = os.path.join(OBJ_DIR, 'ycb/default_object.urdf')

        all_folders = os.listdir(models_directory)
        print(f"Available folders in {models_directory}:")
        for folder in all_folders:
            if object_name.lower() in folder.lower():
                relative_mesh_path = f"{models_directory}/{folder}/google_16k/nontextured.stl"
                with open(template_urdf_path, 'r') as f:
                    urdf_content = f.read()

                urdf_content = urdf_content.replace("MESH_PATH_PLACEHOLDER", relative_mesh_path)
                urdf_content = urdf_content.replace("OBJECT_NAME_PLACEHOLDER", folder)

                temp_urdf_dir = f'{models_directory}/{folder}'
                temp_urdf_filename = f'{folder}.urdf'
                temp_urdf_full_path = f'{temp_urdf_dir}/{temp_urdf_filename}'
                with open(temp_urdf_full_path, 'w') as f:
                    f.write(urdf_content)

                object_urdf_path = temp_urdf_full_path  # Updated to point to generated URDF
                break
    else:
        raise ValueError("Unsupported dataset. Please use 'contactdb' or 'ycb'.")

    hand = create_hand_model(robot_name)
    joint_orders = hand.get_joint_orders()

    simulator = IsaacValidator(
        robot_name=robot_name, 
        joint_orders=joint_orders, 
        batch_size=batch_size,
        gpu=0,  # Hardcoded
        use_gui=use_gui
    )
    simulator.set_asset(
        robot_path=os.path.dirname(robot_urdf_path),
        robot_file=os.path.basename(robot_urdf_path),  
        object_path=os.path.dirname(object_urdf_path),  
        object_file=os.path.basename(object_urdf_path)  
    )
    simulator.create_envs()

    if q_batch is not None and q_batch.shape[-1] != len(joint_orders):
        q_batch = q_rot6d_to_q_euler(q_batch)  # 6->3

    simulator.set_actor_pose_dof(q_batch.to(torch.device('cpu')))
    success, q_grasped = simulator.run_sim()
    simulator.destroy()

    successful_grasps = []  # For storing successful grasps
    if success.any():
        success_q_isaac = q_grasped[success]  # Extract successful joint angles

        if evaluate_diversity:
            diversity_std = torch.std(success_q_isaac, dim=0).mean().item()  # Calculate standard deviation and average

        # Calculate penetration values
        for q, was_success in zip(success_q_isaac, success.nonzero(as_tuple=True)[0]):
            # Build the grasp data including q_isaac
            grasp_data = grasps_data[was_success.item()]  # Retrieve original grasp data
            grasp_data['q_final'] = q.cpu().numpy()  # Add q_isaac data
            successful_grasps.append(grasp_data)  # Store the successful grasp data
            
        # Log relevant information
        success_num = len(successful_grasps)
        result_str = f"[{robot_name}/{object_name}] Success: {success_num}/{batch_size}"
        if evaluate_diversity and 'diversity_std' in locals():
            result_str += f" | Diversity Std: {diversity_std:.4f}"
        cprint(result_str, 'green')

    return successful_grasps

def process_pt_file(pt_file_path: str, object_name: str, use_gui: bool, evaluate_diversity: bool):
    # Load the .pt file
    grasps_data = torch.load(pt_file_path, map_location='cpu')

    # Filter grasps based on object_name
    filtered_grasps = [grasp for grasp in grasps_data if grasp['object_name'] == object_name]
    
    if not filtered_grasps:
        print(f"No grasps matching the object name '{object_name}' found in {pt_file_path}.")
        return

    q_batch = torch.stack([grasp['q'] for grasp in filtered_grasps], dim=0)
    batch_size = q_batch.shape[0]

    successful_grasps = isaac_main(
        object_name=object_name,
        batch_size=batch_size,
        q_batch=q_batch,
        grasps_data=filtered_grasps,
        use_gui=use_gui,
        evaluate_diversity=evaluate_diversity
    )

    # Output results after processing
    success_count = len(successful_grasps)
    success_rate = success_count / batch_size if batch_size > 0 else 0
    print(f"Processed {pt_file_path}: Found {success_count} successful grasps (Success Rate: {success_rate:.2f})")

if __name__ == '__main__':
    warnings.simplefilter(action='ignore', category=FutureWarning)
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, required=True, help='Path to the .pt file')
    parser.add_argument('--object_name', type=str, required=True, help='Object name in the format <dataset>+<object>. e.g., contactdb+banana')
    parser.add_argument('--use_gui', action='store_true')
    parser.add_argument('--eval_diversity', action='store_true', help='Whether to evaluate diversity')
    args = parser.parse_args()

    print(f'Evaluating {args.object_name} from {args.input_file}...')
    process_pt_file(args.input_file, args.object_name, args.use_gui, args.eval_diversity)