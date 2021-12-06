#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import numpy as np
from PIL import Image
import pandas
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
device = torch.device("cuda")

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

def AbsolutePose(pose1, relPose):
    c = np.cos(pose1[2])
    s = np.sin(pose1[2])
    x = c*relPose[0] - s*relPose[1]
    y = s*relPose[0] + c*relPose[1]
    Theta = pose1[2] + relPose[2]
    X = pose1[0] + x
    Y = pose1[1] + y
    return np.array([X, Y, Theta])

def GenerateInputVector(dir):
    max_succ_dist = 0.5
    max_succ_ang = math.radians(10)
    pose_list = pandas.read_csv(dir + 'pose_list.csv', header=None).values
    N = len(pose_list)
    
    image_list = []
    image1_list = []
    image2_list = []
    rel_pose_list = []
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

class Net(nn.Module):
    #Architecture is LeNet modification
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(2, 6, 5).cuda()
        self.conv2 = nn.Conv2d(6, 16, 5).cuda()
        self.fc1 = nn.Linear(44944, 120).cuda()
        self.fc2 = nn.Linear(120, 84).cuda()
        self.fc3 = nn.Linear(84, 1).cuda()

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        #print(x.shape)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

