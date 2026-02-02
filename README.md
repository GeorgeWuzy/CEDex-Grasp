# CEDex Code and Dataset
Official Repository of **CEDex: Cross-Embodiment Dexterous Grasp Generation at Scale from Human-like Contact Representations (ICRA 2026)**

<span class="author-block">
  <a href="https://georgewuzy.github.io/" style="color: #4A90E2 !important;">Zhiyuan Wu</a><sup>1</sup>,
</span>
<span class="author-block">
  <a href="https://rolpotamias.github.io/" style="color: #4A90E2 !important;">Rolandos Alexandros Potamias</a><sup>2</sup>,
</span>
<span class="author-block">
  <a href="https://scholar.google.com/citations?user=osQ5dAkAAAAJ&hl=zh-CN&oi=ao" style="color: #4A90E2 !important;">Xuyang Zhang</a><sup>1</sup>,
</span>
<span class="author-block">
  <a href="https://zhongqunzhang.github.io/" style="color: #4A90E2 !important;">Zhongqun Zhang</a><sup>3</sup>,
</span>
<span class="author-block">
  <a href="https://jiankangdeng.github.io/" style="color: #4A90E2 !important;">Jiankang Deng</a><sup>2</sup>,
</span>
<span class="author-block">
  <a href="https://shanluo.github.io/" style="color: #4A90E2 !important;">Shan Luo</a><sup>1</sup>
</span>

<sup>1</sup> King's College London, <sup>2</sup> Imperial College London, <sup>3</sup> Nankai University

<p align="center">
    <a href='https://arxiv.org/abs/2509.24661'>
      <img src='https://img.shields.io/badge/Paper-arXiv-red?style=plastic&logo=arXiv&logoColor=red' alt='Paper arXiv'>
    </a>
    <a href='https://arxiv.org/pdf/2509.24661'>
      <img src='https://img.shields.io/badge/Paper-PDF-FF9547?style=plastic&logo=adobeacrobatreader&logoColor=FF9547' alt='Paper PDF'>
    </a>
    <a href='https://georgewuzy.github.io/cedex-website/'>
      <img src='https://img.shields.io/badge/Project-Page-66C0FF?style=plastic&logo=Google%20chrome&logoColor=66C0FF' alt='Project Page'>
    </a>
</p>

## Overview
![Pipeline](assets/pipeline.gif)  
In this paper, we propose **CEDex**, a novel cross-embodiment dexterous grasp synthesis method that bridges human grasping kinematics and robot kinematics by aligning robot kinematic models with generated human-like contact representations. Using CEDex, we construct the largest cross-embodiment grasp dataset to date, comprising **500K objects** across four gripper types with **20M total grasps**.

## Dataset Release

We have released a part of our **synthesis object set** grasp data, including objects from [Objaverse](https://objaverse.allenai.org/). We will soon release more data. Please stay tuned for updates!

![Simulation Objects](assets/sim_objects.gif)


We have released the **real-world object set** grasp data, including objects from [ContactDB](https://contactdb.cc.gatech.edu/) and [YCB](https://www.ycbbenchmarks.com/).

![Real-World Objects](assets/rw_objects.gif)

The code for data generation will soon be published. Please stay tuned for updates!

## Dependencies

Create a new conda environment. My CUDA version (nvcc --version) is 12.4
```bash  
conda create -n cedex python=3.8  
conda activate cedex
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu124
git clone https://github.com/facebookresearch/pytorch3d.git
cd pytorch3d
# export MAX_JOBS=4 # set MAX_JOBS if your workstation doesn't have enough memory
pip install -e .
unset MAX_JOBS
cd ../
``` 

Install required packages.
```bash  
pip install -r requirements.txt
``` 

Install Isaac Gym Environment (for validation).

Download [Isaac Gym](https://developer.nvidia.com/isaac-gym/download) from the official website, then:
```bash  
tar -xvf IsaacGym_Preview_4_Package.tar.gz
cd isaacgym/python
pip install -e .
``` 

Install pointnet_lib for ContactGen.
```bash
cd contactgen
cd pointnet_lib
python setup.py install
cd ../../
```

## Data Preparation

For **real-world object set** grasp data, download the robot and object data from [Google Drive link](https://drive.google.com/file/d/1xmBV66SO-TjkREYTujh08QkucCWHYLxx/view?usp=sharing) and extract the contents to the `data/` directory. Our grasp data is in `cedex/{robot_name}.pt`. 

For **synthesis object set** grasp data, download the object and grasp data from [Google Drive link](https://drive.google.com/drive/folders/158vnKHRjZ0DihWwUAgu52iE3MtHeoR8W?usp=sharing) and extract the contents to the `data/object/objaverse` directory. Download the grasp data from [Google Drive link](https://drive.google.com/drive/folders/158vnKHRjZ0DihWwUAgu52iE3MtHeoR8W?usp=sharing) and put the contents to the `cedex_objaverse/` directory. `{robot_name}_isaac.pt` refer to grasp data after grasp execution in Isaac with no penetration. 

## Usage

### Data Usage

In our data, each grasp data contains the following keys: `['q']` and `['object_name']`. You can use the dataset by calling the robot hand through the `utils_model/HandModel.py`. This implementation is consistent with [GenDexGrasp](https://github.com/tengyu-liu/GenDexGrasp), but please note that there are differences in object URDF files. Make sure to use our provided objects.

To visualize the grasping data, you can use the following command:

```bash  
python vis_cedex.py --input_file cedex/allegro.pt  
python vis_cedex.py --input_file cedex_objaverse/shadowhand.pt  
```

### Grasp Validation

For grasp validation, our implementation is based on [DRO-Grasp](https://github.com/zhenyuwei2003/DRO-Grasp). We have modified some parameters of the controller. You can evaluate the grasping performance using the following command:

```bash
python eval_grasp.py --input_file cedex/allegro.pt --object_name ycb+055_baseball # --use_gui --eval_diversity
```

### Grasp Generation

You can use CEDex to generate your own grasp data. 

Stage 1: Generate Human-Like Contact Maps
```bash
cd contactgen
python inf_contactdb.py
python inf_ycb.py
cd ../
```

Stage 2: Generate Grasps
```bash
python generate_data.py --robot_name barrett --dataset contactdb
```
**Note**: Some hyperparameters can be adjusted to get more stable grasp results or more diverse grasp results. 
1) Initial positions of robot hands in `utils_model/CMapAdam.py`
2) Hyperparameters of physical constraints in `utils_model/CMapAdam.py`
3) Sort the top-k saved grasps in `generate_grasp.py` by different properties in contact energy

Stage 3: Visualize Generated Grasps
```bash
python vis_generated_grasp.py --robot_name barrett --input_dir logs/dataset_generation_20260202_162359/barrett_contactdb/ # replace the input_dir to your log dir. 
```

Stage 4: Validate and Filter Generated Grasps
```bash
python eval_grasp_filtered.py --robot_name barrett --logs_path logs/dataset_generation_20260202_162359/barrett_contactdb/ # --use_gui # replace the logs_path to your log dir
```

## Citation

If you find this work helpful, please consider citing us using the following BibTeX entry:

```bibtex  
@article{wu2025cedex,  
  title={CEDex: Cross-Embodiment Dexterous Grasp Generation at Scale from Human-like Contact Representations},  
  author={Wu, Zhiyuan and Potamias, Rolandos Alexandros and Zhang, Xuyang and Zhang, Zhongqun and Deng, Jiankang and Luo, Shan},  
  journal={arXiv preprint arXiv:2509.24661},  
  year={2025}  
}  
```

## Contact

If you have any questions, feel free to contact me through email at [zhiyuan.1.wu@kcl.ac.uk](mailto:zhiyuan.1.wu@kcl.ac.uk).