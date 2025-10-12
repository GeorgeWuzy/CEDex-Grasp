# CEDex Code and Dataset
Official Repository of **CEDex: Cross-Embodiment Dexterous Grasp Generation at Scale from Human-like Contact Representations**

Zhiyuan Wu<sup>1</sup>, Rolandos Alexandros Potamias<sup>2</sup>, Xuyang Zhang<sup>1</sup>, Zhongqun Zhang<sup>3</sup>, Jiankang Deng<sup>2</sup>, Shan Luo<sup>1</sup>  
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

We have currently released the **real-world object set** grasp data, including objects from [ContactDB](https://contactdb.cc.gatech.edu/) and [YCB](https://www.ycbbenchmarks.com/).

![Real-World Objects](assets/rw_objects.gif)

We will soon release a larger scale of **synthesis objects** grasp data, including objects from [Objaverse](https://objaverse.allenai.org/). Please stay tuned for updates!

![Simulation Objects](assets/sim_objects.gif)

The code for data generation will be published after the acceptance of the paper.