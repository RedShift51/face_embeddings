# facedetector
import cv2
import numpy as np
import mtcnn

# insightface
import torch
from imageio import imread
from torchvision import transforms
import insightface

import torch.nn.functional as F

import argparse


def get_rot_mat(theta):
    theta = torch.tensor(theta)
    return torch.tensor([[torch.cos(theta), -torch.sin(theta), 0],
                         [torch.sin(theta), torch.cos(theta), 0]])

def rot_img(x, theta, dtype):
    rot_mat = get_rot_mat(theta)[None, ...].type(dtype).repeat(x.shape[0],1,1)
    grid = F.affine_grid(rot_mat, x.size()).type(dtype)
    x = F.grid_sample(x, grid)
    return x


def main(img):
    # detect faces
    pnet, rnet, onet = mtcnn.get_net_caffe('output/converted')
    detector = mtcnn.FaceDetector(pnet, rnet, onet, device='cuda:0')
    #img = '../FaceDetector/tests/asset/images/office5.jpg'
    img = imread(img)
    boxes, landmarks = detector.detect(img)
    img = torch.tensor(img.astype(np.float32), device=torch.device("cuda:0")).permute(2, 0, 1)

    # embed faces
    embedder = insightface.iresnet100(pretrained=True)
    embedder.eval()

    mean = [127.5] * 3 #[0.5] * 3
    std = [128.] * 3 #[0.5 * 256 / 255] * 3
    preprocess = transforms.Compose([
        transforms.Normalize(mean, std)
    ])

    landmarks = landmarks.float()
    boxcpu = boxes.cpu().numpy()
    for f0 in range(boxcpu.shape[0]):
        angle = torch.atan2(landmarks[f0][1, 1] - landmarks[f0][0, 1], \
                            landmarks[f0][1, 0] - landmarks[f0][0, 0])# + np.pi/2
        local_patch = img[:, boxes[f0][1]: boxes[f0][3], boxes[f0][0]: boxes[f0][2]].unsqueeze(0)
        local_patch = rot_img(local_patch, angle, dtype=torch.cuda.FloatTensor)
        local_patch = local_patch.squeeze(0)

        tensor_face = F.interpolate(local_patch, size=112)
        tensor_face = tensor_face.permute(0, 2, 1)
        tensor_face = F.interpolate(tensor_face, size=112)
        tensor_face = tensor_face.permute(0, 2, 1)

        tensor_face = preprocess(tensor_face.cpu())
        with torch.no_grad():
            features = embedder(tensor_face.unsqueeze(0))[0].numpy()
            print(features)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img", type=str)
    args = parser.parse_args()
    return vars(args)

if __name__=="__main__":
    arguments = parse_args()
    main(arguments["img"])
