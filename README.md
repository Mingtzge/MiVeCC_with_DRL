# MiVeCC_with_DRL
This is a Multi-intersection Vehicular Cooperative Control (MiVeCC) scheme to enable cooperation among vehicles in a 3*3 unsignalized intersections. we proposed a algorithm combined heuristic-rule and two-stage deep reinforcement learning. The heuristic-rule achieves vehicles across the intersections without collisions. Based on the heuristic-rule, DDPG is used to optimize the collaborative control of vehicles and improve the traffic efficiency. Simulation results show that the proposed algorithm can improve travel efficiency at multiple intersections by up to 4.59 times without collision compared with existing methods.

A Multi-intersection Vehicular Cooperative Control based on End-Edge-Cloud Computing|[paper](https://arxiv.org/pdf/2012.00500.pdf)

![visible](https://github.com/Mingtzge/MiVeCC_with_DRL/blob/main/show_imgs/multi.gif)

## Prerequisites
- Linux or macOS
- Python 3
- matlab 2017b
- CPU or NVIDIA GPU + CUDA CuDNN

## Getting Started
### Installation
- Clone this repo:
```bash
git clone https://github.com/Mingtzge/MiVeCC_with_DRL.git
cd MiVeCC_with_DRL
```

### Test the pre-trained model (End_edge_cloud)
```
python MiVeCC_main.py --s_exp_name Edge_baseline --mat_path arvTimeNewVeh_new_1200_multi3_3.mat --priori_knowledge --type test --m_exp_name Cloud_baseline --visible --video_name test
```

### Cloud train/test (on the "main" branch)
- generate the arriveTime file
```bash
matlab gen_multi_arvTime.m
```
- Train a model:
```
python MiVeCC_main.py --mat_path arvTimeNewVeh_new_900_multi3_3_l.mat --type train --priori_knowledge --exp_name cloud_demo
```
To see more intermediate results, run 
```
tensorboard --logdir ./model_data/cloud_demo/log
```
- Test the model:
```
python MiVeCC_main.py --mat_path arvTimeNewVeh_new_900_multi3_3.mat --type test --priori_knowledge --exp_name cloud_demo --visible --video_name cloud_demo
```
Note:the visual prarameters "--visible" and "--video_name" is optional. If use the "--visible", there will be a simulation interface to show the running interface of the vehicle in real time. the "--video_name cloud_demo" is used to generate a video ,named "cloud_demo.avi", saved in "./results_img/".

### Edge train/test (on the "single_intersection" branch)
- generate the arriveTime file
```
git checkout single_intersection
matlab gen_single_arvTime.m
```
- Train a model:
```
python MiVeCC_main.py --mat_path arvTimeNewVeh_new_900_l.mat --type train --priori_knowledge --exp_name edge_demo
```
To see more intermediate results, run 
```
tensorboard --logdir ./model_data/edge_demo/log
```
- Test the model:
```
python MiVeCC_main.py --mat_path arvTimeNewVeh_new_900.mat --type test --priori_knowledge --exp_name edge_demo --visible --video_name edge_demo
```
Note:the visual prarameters "--visible" and "--video_name" is optional. If use the "--visible", there will be a simulation interface to show the running interface of the vehicle in real time. the "--video_name edge_demo" is used to generate a video ,named "edge_demo.avi", saved in "./results_img/".
