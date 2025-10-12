import os
import sys
import json
import argparse
import warnings
from termcolor import cprint

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from validation.isaac_validator import IsaacValidator  # IsaacGym must be imported before PyTorch
from utils.hand_model import create_hand_model
from utils.rotation import q_rot6d_to_q_euler

import torch


def isaac_main(
    mode: str,
    robot_name: str,
    object_name: str,
    batch_size: int,
    q_batch: torch.Tensor = None,
    gpu: int = 0,
    use_gui: bool = False
):
    """
    For filtering dataset and validating grasps.

    :param mode: str, 'filter' or 'validation'
    :param robot_name: str
    :param object_name: str
    :param batch_size: int, number of grasps in Isaac Gym simultaneously
    :param q_batch: torch.Tensor (validation only)
    :param gpu: int, specify the GPU device used by Isaac Gym
    :param use_gui: bool, whether to visualize Isaac Gym simulation process
    :return: success: (batch_size,), bool, whether each grasp is successful in Isaac Gym;
             q_isaac: (success_num, DOF), torch.float32, successful joint values after the grasp phase
    """
    if mode == 'filter' and batch_size == 0:  # special judge for num_per_object = 0 in dataset
        return 0, None
    if use_gui:  # for unknown reason otherwise will segmentation fault :(
        gpu = 0

    urdf_assets_meta = json.load(open(os.path.join(ROOT_DIR, 'GraspOptimization/data/urdf/urdf_assets_meta_extended.json')))
    robot_urdf_path = urdf_assets_meta['urdf_path'][robot_name]
    object_name_split = object_name.split('+') if object_name is not None else None
    # object_urdf_path = f'{object_name_split[0]}/{object_name_split[1]}/{object_name_split[1]}.urdf'
    object_urdf_path = f'{object_name_split[0]}/{object_name_split[1]}/coacd_decomposed_object_one_link.urdf'

    hand = create_hand_model(robot_name)
    joint_orders = hand.get_joint_orders()

    simulator = IsaacValidator(
        robot_name=robot_name, 
        joint_orders=joint_orders, 
        batch_size=batch_size,
        gpu=gpu, 
        is_filter=(mode == 'filter'),
        # use_gui=use_gui
        use_gui=True
    )
    print("[Isaac] IsaacValidator is created.")
    simulator.set_asset(
        robot_path=os.path.dirname(robot_urdf_path),
        robot_file=os.path.basename(robot_urdf_path),
        object_path=os.path.dirname(object_urdf_path),
        object_file=os.path.basename(object_urdf_path)
    )
    simulator.create_envs()
    print("[Isaac] IsaacValidator preparation is done.")

    if q_batch.shape[-1] != len(joint_orders):
        q_batch = q_rot6d_to_q_euler(q_batch)

    simulator.set_actor_pose_dof(q_batch.to(torch.device('cpu')))
    success, q_isaac = simulator.run_sim()
    simulator.destroy()

    return success, q_isaac