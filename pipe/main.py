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

    mean = [0.5] * 3
    std = [0.5 * 256 / 255] * 3
    preprocess = transforms.Compose([
        transforms.Normalize(mean, std)
    ])

    boxcpu = boxes.cpu().numpy()
    for f0 in range(boxcpu.shape[0]):
        tensor_face = F.interpolate(
                img[:, boxes[f0][1]: boxes[f0][3], boxes[f0][0]: boxes[f0][2]], 
                size=112)
        tensor_face = tensor_face.permute(0, 2, 1)
        tensor_face = F.interpolate(tensor_face, size=112)
        tensor_face = tensor_face.permute(0, 1, 2)

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
