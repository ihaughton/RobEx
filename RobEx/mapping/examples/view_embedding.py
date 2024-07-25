#!/usr/bin/env python
import torch
import cv2
import numpy as np

from RobEx.mapping import embedding

if __name__ == "__main__":
    W = 500
    H = 500
    device = "cuda"

    c, r = torch.meshgrid(torch.arange(W)-W//2, torch.arange(H)-H//2)
    c, r = c.t().float().to(device), r.t().float().to(device)
    im = torch.stack([c, r], dim=2)

    points_embedding = embedding.positional_encoding(
        im, num_encoding_functions=1, scale=10.)

    embed1 = points_embedding[:, :, 2:4]
    ones = torch.zeros([embed1.shape[0], embed1.shape[1], 1], device=device) -1.
    embed1 = torch.cat([embed1, ones], dim=2)
    embed1 = embed1.cpu().numpy()
    embed1 = ((embed1+1)*128).astype(np.uint8)
    cv2.imshow("embed", embed1)
    cv2.imwrite("embed_ng.png", embed1)
    cv2.waitKey(1)

    B_layer = torch.nn.Linear(2, 3, bias=True).to(device)
    B_layer.weight.data.normal_(std=1)
    B_layer.bias.data.uniform_(-np.pi, np.pi)
    print(B_layer.weight.data)
    points_embedding = embedding.positional_encoding(
        im, scale=10., B_layer=B_layer).detach().cpu().numpy()
    points_embedding = ((points_embedding+1)*128).astype(np.uint8)
    cv2.imshow("embed_gauss", points_embedding)
    cv2.imwrite("embed_g1.png", points_embedding)
    cv2.waitKey(1)

    B_layer = torch.nn.Linear(2, 3, bias=True).to(device)
    B_layer.weight.data.normal_(std=1)
    B_layer.bias.data.uniform_(-np.pi, np.pi)
    print(B_layer.weight.data)
    points_embedding = embedding.positional_encoding(
        im, scale=20., B_layer=B_layer).detach().cpu().numpy()
    points_embedding = ((points_embedding+1)*128).astype(np.uint8)
    cv2.imshow("embed_gauss2", points_embedding)
    cv2.imwrite("embed_g2.png", points_embedding)
    cv2.waitKey(1)


    B_layer = torch.nn.Linear(2, 3, bias=True).to(device)
    B_layer.weight.data.normal_(std=1)
    B_layer.bias.data.uniform_(-np.pi, np.pi)
    print(B_layer.weight.data)
    points_embedding = embedding.positional_encoding(
        im, scale=30., B_layer=B_layer).detach().cpu().numpy()
    points_embedding = ((points_embedding+1)*128).astype(np.uint8)
    cv2.imshow("embed_gauss3", points_embedding)
    cv2.imwrite("embed_g3.png", points_embedding)
    cv2.waitKey(1)

    B_layer = torch.nn.Linear(2, 1, bias=False).to(device)
    B_layer.weight.data.normal_()
    # dir = torch.tensor([0.3, 0.7])
    # B_layer.weight.data = dir
    B_layer.weight.data = B_layer.weight / B_layer.weight.norm(dim=1)[:, None]
    print(B_layer.weight.data)
    # B_layer.bias.data.uniform_(-np.pi, np.pi)
    points_embedding = embedding.positional_encoding(
        im, scale=10., B_layer=B_layer).detach().cpu().numpy()

    points_embedding = ((points_embedding+1)*128).astype(np.uint8)
    fill = np.zeros(points_embedding.shape).astype(np.uint8)
    points_embedding = np.concatenate((fill, fill, points_embedding), axis=2)

    center = np.array([H//2,W//2])
    direction = (B_layer.weight.data[0].cpu().numpy()*200).astype(int)
    direction = center + direction
    cv2.line(points_embedding, center, direction, (0, 255, 0), thickness=3)

    cv2.imshow("embed_gauss4", points_embedding)
    cv2.imwrite("embed_g4.png", points_embedding)
    cv2.waitKey(0)
