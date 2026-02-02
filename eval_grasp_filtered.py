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
import time

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)
OBJ_DIR = 'data/object'

def isaac_main(
    robot_name: str,
    object_name: str,
    batch_size: int,
    q_batch: torch.Tensor = None,
    grasps_data: list = None,
    gpu: int = 0,
    use_gui: bool = False,
):
    if use_gui:
        gpu = 0

    urdf_assets_meta = json.load(open(os.path.join(ROOT_DIR, 'GraspOptimization/data/urdf/urdf_assets_meta_extended.json')))
    robot_urdf_path = urdf_assets_meta['urdf_path'][robot_name]
    dataset_name = object_name.split('+')[0]
    object_name = object_name.split('+')[1]
    
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

                object_urdf_path = f'{models_directory}/{folder}/{temp_urdf_filename}'
                break

    hand = create_hand_model(robot_name)
    joint_orders = hand.get_joint_orders()

    simulator = IsaacValidator(
        robot_name=robot_name, 
        joint_orders=joint_orders, 
        batch_size=batch_size,
        gpu=gpu, 
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
        q_batch = q_rot6d_to_q_euler(q_batch) # 6->3
    simulator.set_actor_pose_dof(q_batch.to(torch.device('cpu')))
    success, q_grasped = simulator.run_sim()
    simulator.destroy()

    successful_grasps = []  # For storing successful grasps
    diversity_std = None  # Variable for diversity std if calculated

    if success.any():
        success_q_isaac = q_grasped[success]  # Extract successful joint angles
        diversity_std = torch.std(success_q_isaac, dim=0).mean().item()  # Calculate standard deviation and average
            
        for q, was_success in zip(success_q_isaac, success.nonzero(as_tuple=True)[0]):
            # Build the grasp data including q_isaac
            grasp_data = grasps_data[was_success.item()]  # Retrieve original grasp data
            grasp_data['q_final'] = q.cpu().numpy()  # Add q_isaac data
            successful_grasps.append(grasp_data)  # Store the successful grasp data
    try:
        return successful_grasps, diversity_std
    except Exception as e:
        return successful_grasps, None

def process_directory(logs_path: str, robot_name: str, gpu: int, use_gui: bool, save_successes: bool):
    success_count = 0
    total_grasps = 0
    successful_grasps_all = []  # List to collect all successful grasps
    diversity_std_all = []  # List to store diversity standard deviations

    generated_grasps_path = os.path.join(logs_path, 'generated_grasps')  # Ensure this is where generated grasps are at
    filtered_grasps_path = os.path.join(logs_path, 'filtered_grasps')  # Path for saving filtered grasps

    success_rates = {}
    diversity_stats = {}

    for file_name in os.listdir(generated_grasps_path):
        if file_name.endswith('.pt'):
            q_file_path = os.path.join(generated_grasps_path, file_name)
            grasps_data = torch.load(q_file_path, map_location='cpu')
            dataset_object_name = grasps_data[0]['object_name']

            q_batch = torch.stack([grasp['q'] for grasp in grasps_data], dim=0)
            batch_size = q_batch.shape[0]

            successful_grasps, diversity_std_temp = isaac_main(
                robot_name=robot_name,
                object_name=dataset_object_name,
                batch_size=batch_size,
                q_batch=q_batch,
                grasps_data=grasps_data,
                gpu=gpu,
                use_gui=use_gui,
            )
            cprint(file_name, 'blue')

            # Calculate success rate and store it
            success_rate = len(successful_grasps) / batch_size if batch_size > 0 else 0
            success_rates[file_name] = success_rate  # Store the success rate

            # Store diversity if calculated
            diversity_stats[file_name] = diversity_std_temp  # Store the diversity stat
            diversity_std_all.append(diversity_std_temp)  # Also add to overall stats

            # Update overall success counts
            success_count += len(successful_grasps)
            total_grasps += batch_size
            successful_grasps_all.extend(successful_grasps)  # Append to list of all successful grasps

            # Save successful grasps for this file immediately if the flag is set
            if save_successes and len(successful_grasps) > 0:
                save_successful_grasps(successful_grasps, filtered_grasps_path, file_name)

            cprint(f"Processed {file_name}: Success Rate = {success_rate:.2f}, Diversity = {diversity_std_temp}", 'yellow')

    return success_count, total_grasps

def save_successful_grasps(successful_grasps, output_path, original_file_name):
    """Save successful grasps into a new .pt file."""
    os.makedirs(output_path, exist_ok=True)  # Create output directory if it doesn't exist
    # Create a base name from the original file name
    base_name = original_file_name.split('.')[0].replace('top16', 'success')  # Adjust naming pattern

    # Prepare data for saving
    for grasp in successful_grasps:
        # Assuming 'q_final' is stored and that we need to convert it back to 6D format
        # print(grasp['q_final'])
        grasp['q_final'] = q_euler_to_q_rot6d(torch.tensor(grasp['q_final']).unsqueeze(0)).squeeze(0)

    # Create a unique file name based on the count of successful grasps
    grasp_id = f"{base_name}_{len(successful_grasps)}.pt"
    file_path = os.path.join(output_path, grasp_id)

    torch.save(successful_grasps, file_path)  # Save successful grasp data
    print(f"Saved successful grasp data to: {file_path}")

if __name__ == '__main__':
    warnings.simplefilter(action='ignore', category=FutureWarning)
    parser = argparse.ArgumentParser()
    parser.add_argument('--robot_name', type=str, required=True)
    parser.add_argument('--logs_path', type=str, required=True)  # Path containing the folder of .pt files
    parser.add_argument('--use_gui', action='store_true')
    parser.add_argument('--save_success', default=True, help='Whether to save successful grasp data')
    parser.add_argument('--gpu', default=0, type=int)
    args = parser.parse_args()

    print(f'GPU: {args.gpu}')

    success_count, total_grasps = process_directory(args.logs_path, args.robot_name, args.gpu, args.use_gui, args.save_success)
    if total_grasps > 0:
        cprint(f"[Overall Result] Success Rate: {success_count}/{total_grasps}", 'green')
    else:
        cprint("[Overall Result] No grasps processed.", 'red')