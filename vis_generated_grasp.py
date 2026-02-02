
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
    parser.add_argument('--input_dir', required=True, type=str,
                        help='Directory containing generated_grasps subdirectory), e.g. logs/dataset_generation_20250717_235629/shadowhand_ycb')
    parser.add_argument('--robot_name', default='barrett', type=str,
                        choices=['robotiq_3finger', 'barrett', 'allegro', 'shadowhand', 'leaphand'])
    return parser.parse_args()

def load_object_mesh(dataset_type, object_name):
    if dataset_type == 'contactdb':
        object_mesh_path = os.path.join(OBJ_DIR, f'contactdb/{object_name}/{object_name}.stl')
                
    elif dataset_type == 'ycb':
        models_directory = os.path.join(OBJ_DIR, 'ycb')
        object_mesh_path = f'{models_directory}/{object_name}/google_16k/nontextured.stl'
                
    object_mesh = tm.load(object_mesh_path)
    
    if isinstance(object_mesh, tm.Scene):
        object_mesh = list(object_mesh.geometry.values())[0]
    
    print(f"Successfully loaded object: {object_name} from {object_mesh_path}")
    return object_mesh
        
def save_combined_mesh(hand_meshes, object_mesh, output_path):
    combined_hand_vertices = np.vstack([mesh.vertices for mesh in hand_meshes])
    combined_hand_faces = []
    vertex_offset = 0
    for mesh in hand_meshes:
        combined_hand_faces.append(mesh.faces + vertex_offset)
        vertex_offset += len(mesh.vertices)
    combined_hand_faces = np.vstack(combined_hand_faces)

    combined_vertices = np.vstack([combined_hand_vertices, object_mesh.vertices])
    combined_faces = np.vstack([combined_hand_faces, object_mesh.faces + len(combined_hand_vertices)])

    combined_mesh = tm.Trimesh(vertices=combined_vertices, faces=combined_faces)

    hand_color = np.array(colors['light_red']) * 255
    object_color = np.array(colors['light_blue']) * 255
    
    combined_mesh.visual.vertex_colors = np.vstack([
        np.tile(hand_color.astype(np.uint8), (len(combined_hand_vertices), 1)),
        np.tile(object_color.astype(np.uint8), (len(object_mesh.vertices), 1)),
    ])

    ply_path = output_path.replace('.obj', '.ply')
    combined_mesh.export(ply_path, file_type='ply')
    print(f"Saved combined mesh to {ply_path}")

def visualize_and_save(hand_meshes, object_mesh, save_path, show_window=True):
    if show_window:
        vis = o3d.visualization.Visualizer()
        vis.create_window()

        for hand_mesh in hand_meshes:
            vis_hand = o3d.geometry.TriangleMesh()
            vis_hand.vertices = o3d.utility.Vector3dVector(hand_mesh.vertices)
            vis_hand.triangles = o3d.utility.Vector3iVector(hand_mesh.faces)
            vis_hand.paint_uniform_color(colors['light_red'])
            vis_hand.compute_vertex_normals()
            vis.add_geometry(vis_hand)

        vis_obj = o3d.geometry.TriangleMesh()
        vis_obj.vertices = o3d.utility.Vector3dVector(object_mesh.vertices)
        vis_obj.triangles = o3d.utility.Vector3iVector(object_mesh.faces)
        vis_obj.paint_uniform_color(colors['light_blue'])
        vis_obj.compute_vertex_normals()
        vis.add_geometry(vis_obj)

        ctr = vis.get_view_control()
        
        object_center = object_mesh.bounds.mean(axis=0)
        
        ctr.set_front([1, 0, 0])
        ctr.set_lookat(object_center)
        ctr.set_up([0, 0, 1])
        ctr.set_zoom(0.8)

        vis.run()
        vis.destroy_window()
    
    # save_combined_mesh(hand_meshes, object_mesh, save_path)

if __name__ == '__main__':
    args = get_parser()
    
    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory does not exist: {args.input_dir}")
        exit(1)
    
    data_dir = os.path.join(args.input_dir, 'generated_grasps')
    output_dir = os.path.join(args.input_dir, '0_generated_grasps_vis_ply')
    basename = data_dir.split('/')[-2]

    if not os.path.exists(data_dir):
        print(f"Error: Generated grasps directory does not exist: {data_dir}")
        exit(1)
    
    # os.makedirs(output_dir, exist_ok=True)
    
    print(f"Input directory: {args.input_dir}")
    print(f"Reading .pt files from: {data_dir}")
    print(f"Saving .obj files to: {output_dir}")
    
    pt_files = [f for f in os.listdir(data_dir) if f.endswith('.pt')]
    pt_files.sort()
    
    print(f"Found {len(pt_files)} .pt files to process")
    
    hand_model = get_handmodel(args.robot_name, 1, 'cuda', 1.)
    
    processed_count = 0
    
    for idx, pt_file in enumerate(pt_files):
        pt_path = os.path.join(data_dir, pt_file)

        print(f"\nProcessing {processed_count + 1}/{len(pt_files)}: {pt_file}")
        
        grasp_data = torch.load(pt_path)
        
        dataset_type, object_name = grasp_data[0]['object_name'].split('+')

        object_mesh = load_object_mesh(dataset_type, object_name)
        if object_mesh is None:
            print(f"Skipping {pt_file} due to mesh loading error")
            continue
        
        for i, grasp in enumerate(grasp_data):
            q = grasp['q']
            print(f"  Processing grasp {i+1}/{len(grasp_data)}")
            
            hand_meshes = []
            q_expanded = q.unsqueeze(0)
            hand_meshes_from_q = hand_model.get_meshes_from_q(q_expanded.cuda(), 0)
            hand_meshes += hand_meshes_from_q
            
            output_filename = f"{basename}_{idx}_{i}.obj"
            output_path = os.path.join(output_dir, output_filename)
            
            visualize_and_save(hand_meshes, object_mesh, output_path)
        
        processed_count += 1
            
    print(f"\nProcessing completed! {processed_count}/{len(pt_files)} files processed successfully.")
    print(f"Results saved to: {output_dir}")
