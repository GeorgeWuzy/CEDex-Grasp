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
        object_mesh_path = f'{models_directory}/{object_name}/google_16k/nontextured.stl'

    elif dataset_type == 'objaverse':
        objaverse_base_dir = os.path.join(OBJ_DIR, 'objaverse/object_dataset')
        category = ''
        
        # Handle prefixes to determine category and clean object_name
        if object_name.startswith('large_'):
            category = 'large'
            object_name = object_name[6:]  # Remove 'large_'
        elif object_name.startswith('medium_'):
            category = 'medium'
            object_name = object_name[7:]  # Remove 'medium_'
        elif object_name.startswith('small_'):
            category = 'small'
            object_name = object_name[6:]  # Remove 'small_'
            
        # Construct the full path: data/object/objaverse/{category}/{name}/{name}.obj
        object_mesh_path = os.path.join(objaverse_base_dir, category, object_name, f'{object_name}.obj')

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
    if 'allegro' in args.input_file:
        robot_name = 'allegro'
    elif 'barrett' in args.input_file:
        robot_name = 'barrett'
    elif 'shadowhand' in args.input_file:
        robot_name = 'shadowhand'
    
    print(f"Robot name: {robot_name}")

    # Load grasp data
    grasp_data = torch.load(args.input_file)
    
    # Organize grasps by object names
    object_grasps = {}
    vis_list = ["ycb+006_mustard_bottle", "ycb+055_baseball", "ycb+016_pear", "ycb+014_lemon", "ycb+021_bleach_cleanser", "ycb+053_mini_soccer_ball", "ycb+015_peach", "ycb+002_master_chef_can", "ycb+004_sugar_box", "ycb+077_rubiks_cube",
                "contactdb+torusmedium", "contactdb+apple", "contactdb+duck", "contactdb+banana", "contactdb+waterbottle", "contactdb+piggybank", "contactdb+elephant", "contactdb+lightbulb", "contactdb+toruslarge", 
                "large_1699128117be40d9a0b2b7437510ecc0", "large_16497e942e5b45f3ac55d93b3594d62a", "large_165e7f540e1c44dc867f2182bccb7981", "large_16558d37f0104c06baba50a896d52b28", "large_160e847759404e4c8c491ca3a0391a69"]

    for grasp in grasp_data:
        object_grasp = grasp['object_name']
        
        # Check if the current object_grasp is in vis_list
        if object_grasp in vis_list:
            if object_grasp not in object_grasps:
                object_grasps[object_grasp] = []
            object_grasps[object_grasp].append(grasp)

    # Visualize one random grasp for each object
    for object_grasp, grasps in object_grasps.items():
    # for grasps in grasp_data:
        # random_grasp = grasps
        # Select a random grasp
        random_grasp = np.random.choice(grasps)
        if '+' in random_grasp['object_name']:
            dataset_type, object_name = random_grasp['object_name'].split('+')
        else: 
            dataset_type, object_name = 'objaverse', random_grasp['object_name']
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