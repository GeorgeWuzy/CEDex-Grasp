import os
import argparse
import torch
import trimesh as tm
import open3d as o3d
import numpy as np
from utils_model.HandModel import get_handmodel

OBJ_DIR = 'data/object'

colors = {
    'light_red': [0.85882353, 0.74117647, 0.65098039],
    'light_blue': [144.0 / 255, 210.0 / 255, 236.0 / 255]
}

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', required=True, type=str,
                        help='Path to the .pt file, e.g. logs/allegro.pt')
    return parser.parse_args()

def load_object_mesh(dataset_type, object_name):
    """Load the object mesh."""
    if dataset_type == 'contactdb':
        object_mesh_path = os.path.join(OBJ_DIR, f'contactdb/{object_name}/{object_name}.stl')

    elif dataset_type == 'ycb':
        models_directory = os.path.join(OBJ_DIR, 'ycb')
        object_mesh_path = None
        
        for folder in os.listdir(models_directory):
            if object_name in folder:
                test_path = f'{models_directory}/{folder}/google_16k/nontextured.stl'
                object_mesh_path = test_path
                break

    object_mesh = tm.load(object_mesh_path)
    
    # If it's a Scene object, extract the first mesh
    if isinstance(object_mesh, tm.Scene):
        object_mesh = list(object_mesh.geometry.values())[0]
    
    print(f"Successfully loaded object: {object_name} from {object_mesh_path}")
    return object_mesh

def visualize_grasp(hand_meshes, object_mesh):
    """Visualize the grasp with the hand and the object."""
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    # Visualize hand meshes
    for hand_mesh in hand_meshes:
        vis_hand = o3d.geometry.TriangleMesh()
        vis_hand.vertices = o3d.utility.Vector3dVector(hand_mesh.vertices)
        vis_hand.triangles = o3d.utility.Vector3iVector(hand_mesh.faces)
        vis_hand.paint_uniform_color(colors['light_red'])
        vis_hand.compute_vertex_normals()
        vis.add_geometry(vis_hand)

    # Visualize object mesh
    vis_obj = o3d.geometry.TriangleMesh()
    vis_obj.vertices = o3d.utility.Vector3dVector(object_mesh.vertices)
    vis_obj.triangles = o3d.utility.Vector3iVector(object_mesh.faces)
    vis_obj.paint_uniform_color(colors['light_blue'])
    vis_obj.compute_vertex_normals()
    vis.add_geometry(vis_obj)

    vis.run()
    vis.destroy_window()
    
if __name__ == '__main__':
    args = get_parser()
    
    # Load the .pt file
    if not os.path.exists(args.input_file):
        print(f"Error: Input file does not exist: {args.input_file}")
        exit(1)
    
    # Extract the robot name from the filename
    robot_name = os.path.basename(args.input_file).split('.')[0]
    
    print(f"Robot name: {robot_name}")

    # Load grasp data
    grasp_data = torch.load(args.input_file)
    
    # Organize grasps by object names
    object_grasps = {}
    vis_list = ["ycb+006_mustard_bottle", "ycb+055_baseball", "ycb+016_pear", "ycb+014_lemon", "ycb+021_bleach_cleanser", "ycb+053_mini_soccer_ball", "ycb+015_peach", "ycb+002_master_chef_can", "ycb+004_sugar_box", "ycb+077_rubiks_cube",
                "contactdb+torusmedium", "contactdb+apple", "contactdb+duck", "contactdb+banana", "contactdb+waterbottle", "contactdb+piggybank", "contactdb+elephant", "contactdb+lightbulb", "contactdb+toruslarge"]

    for grasp in grasp_data[::-1]:
        object_grasp = grasp['object_name']
        
        # Check if the current object_grasp is in vis_list
        if object_grasp in vis_list:
            if object_grasp not in object_grasps:
                object_grasps[object_grasp] = []
            object_grasps[object_grasp].append(grasp)

    # Visualize one random grasp for each object
    for object_grasp, grasps in object_grasps.items():
        # Select a random grasp
        random_grasp = np.random.choice(grasps)
        dataset_type, object_name = random_grasp['object_name'].split('+')
        q = random_grasp['q']
        
        print(f"Processing object: {object_grasp}")

        # Load the object mesh
        object_mesh = load_object_mesh(dataset_type, object_name)

        # Get the hand meshes
        hand_meshes = []
        q_expanded = q.unsqueeze(0)
        hand_model = get_handmodel(robot_name, 1, 'cuda', 1.)
        hand_meshes_from_q = hand_model.get_meshes_from_q(q_expanded.cuda(), 0)
        hand_meshes += hand_meshes_from_q
        
        # Visualize the grasp
        visualize_grasp(hand_meshes, object_mesh)

    print("All objects processed and visualized!")