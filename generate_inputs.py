#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import cv2
import numpy as np
from PIL import Image
import pandas
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
device = torch.device("cuda")

def Test():
    print('test')
#---------------------------------------------------------------------------------------------------------#
# pose = [x(m), y(m), theta(rad)]
def RelativePose(pose1, pose2):
    c = np.cos(-pose1[2])
    s = np.sin(-pose1[2])
    x = pose2[0] - pose1[0]
    y = pose2[1] - pose1[1]
    Theta = pose2[2] - pose1[2]
    X = c*x - s*y
    Y = s*x + c*y
    return np.array([X, Y, Theta])

#---------------------------------------------------------------------------------------------------------#
def AbsolutePose(pose1, relPose):
    c = np.cos(pose1[2])
    s = np.sin(pose1[2])
    x = c*relPose[0] - s*relPose[1]
    y = s*relPose[0] + c*relPose[1]
    Theta = pose1[2] + relPose[2]
    X = pose1[0] + x
    Y = pose1[1] + y
    return np.array([X, Y, Theta])

#---------------------------------------------------------------------------------------------------------#
def GenerateInputVector(traj_num_list):
    max_succ_dist = 0.5
    max_succ_ang = math.radians(10)
    
    image_list = []
    image1_list = []
    image2_list = []
    rel_pose_list = []
    
    for k in traj_num_list:
        dir = './data/traj_' + str(k) + '/'
        print(dir)
        pose_list = pandas.read_csv(dir + 'pose_list.csv', header=None).values
        N = len(pose_list)
        for i in range(0, N-1):
            image_list.append(cv2.imread(dir + 'img_' + str(i) + '.jpg', 0)/255)
            for j in range(i+1, N, 1):
                rel_pose = RelativePose(pose_list[i], pose_list[j])
                dist = np.sqrt(rel_pose[0]*rel_pose[0] + rel_pose[1]*rel_pose[1])
                ang = rel_pose[2]
                if dist < max_succ_dist and ang < max_succ_ang:
                    image1_list.append(cv2.imread(dir + 'img_' + str(i) + '.jpg', 0)/255)
                    image2_list.append(cv2.imread(dir + 'img_' + str(j) + '.jpg', 0)/255)
                    rel_pose_list.append(rel_pose)
        image_list.append(cv2.imread(dir + 'img_' + str(N-1) + '.jpg', 0)/255)
    return pose_list, image_list, image1_list, image2_list, rel_pose_list

        
#---------------------------------------------------------------------------------------------------------#
class TrajectoryDataset(Dataset):

    def __init__(self, root_dir, traj_num_list , max_num_pose, max_dis, max_ang):
        idx_map = np.zeros(len(traj_num_list), dtype = int)
        rel_pose_idx_map = []
        rel_pose_map = []
        l = 0
        for k in traj_num_list:
            dir = root_dir + 'traj_' + str(k) + '/'
            print(k)

            pose_list = pandas.read_csv(dir + 'pose_list.csv', header=None).values
            N = len(pose_list)
            rel_pose_idx_list = np.zeros(N-1, dtype = int)
            rel_pose_list = []

            for i in range(0, N-1):
                num_poses = 0
                for j in range(i+1, N, 1):
                    rel_pose = RelativePose(pose_list[i], pose_list[j])
                    dis = np.sqrt(rel_pose[0]*rel_pose[0] + rel_pose[1]*rel_pose[1])
                    ang = rel_pose[2]

                    if dis <= max_dis and ang <= max_ang and (j - i) <= max_num_pose:
                        rel_pose_list.append(rel_pose)
                        num_poses = num_poses + 1
                    else:
                        break      
                rel_pose_idx_list[i] = num_poses
            rel_pose_idx_list = np.cumsum(rel_pose_idx_list)
            idx_map[l] = len(rel_pose_list)

            if rel_pose_idx_list[-1] != idx_map[l]:
                print('Error in TrajectoryDataset')
            rel_pose_idx_map.append(rel_pose_idx_list)
            rel_pose_map.append(rel_pose_list)
            l = l + 1
        idx_map = np.cumsum(idx_map)
        
        self.idx_map = idx_map
        self.rel_pose_idx_map = rel_pose_idx_map
        self.rel_pose_map = rel_pose_map
        self.root_dir = root_dir
        self.traj_num_list = traj_num_list

    def __len__(self):
        return self.idx_map[-1]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        if idx >= self.idx_map[-1] or idx < 0:
            print('Larger than bound')

        outer_idx = np.argmax(self.idx_map > idx)
        if outer_idx > 0:
            ij = idx - self.idx_map[outer_idx - 1]
            inner_idx = np.argmax(self.rel_pose_idx_map[outer_idx] > ij)
        else:
            ij = idx
            inner_idx = np.argmax(self.rel_pose_idx_map[outer_idx] > ij)

        i = inner_idx
        if i > 0:
            j = ij - self.rel_pose_idx_map[outer_idx][i - 1] + i + 1
        else:
            j = ij + i + 1
        
        dir = self.root_dir + 'traj_' + str(self.traj_num_list[outer_idx]) + '/'
        
        image1 = cv2.imread(dir + 'img_' + str(i) + '.jpg', 0)/255
        image2 = cv2.imread(dir + 'img_' + str(j) + '.jpg', 0)/255
        rel_pose = self.rel_pose_map[outer_idx][ij]
        
        sample = {'image1': image1, 'image2': image2, 'rel_pose': rel_pose}

        return sample


#---------------------------------------------------------------------------------------------------------#
class Net(nn.Module):
    #Architecture is LeNet modification
    def __init__(self):
        super(Net, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=5, stride=2, padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(), 
            nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=0),
            nn.BatchNorm2d(256),
            nn.ReLU(),)
        self.linear_layers = nn.Sequential(
            nn.Linear(4096, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            )

    def forward(self, x):
        x = self.conv_layers(x)
        x = torch.flatten(x, 1)
        x = self.linear_layers(x)
        return x[0][0]

