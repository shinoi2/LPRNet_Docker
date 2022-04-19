#! /usr/bin/env python
# coding=utf8

import time
from concurrent import futures
import grpc
from model.LPRNet import build_lprnet
import torch
import numpy as np
import lprnet_pb2, lprnet_pb2_grpc

_ONE_DAY_IN_SECONDS = 60 * 60 * 24

CHARS = ['京', '沪', '津', '渝', '冀', '晋', '蒙', '辽', '吉', '黑',
         '苏', '浙', '皖', '闽', '赣', '鲁', '豫', '鄂', '湘', '粤',
         '桂', '琼', '川', '贵', '云', '藏', '陕', '甘', '青', '宁',
         '新',
         '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
         'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K',
         'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
         'W', 'X', 'Y', 'Z', 'I', 'O', '-'
         ]
model = "./weights/Final_LPRNet_model.pth"
image_size = (94, 24)

class Service(lprnet_pb2_grpc.LprnetServiceServicer):

    def __init__(self):
        self.lprnet = build_lprnet(
            lpr_max_len=8,
            class_num=len(CHARS),
            dropout_rate=0)
        device = torch.device("cuda:0")
        self.lprnet.to(device)
        self.lprnet.load_state_dict(torch.load(model))

    def predict(self, request, context):
        images = []
        for image in request.images:
            img = np.frombuffer(image, dtype=np.float32).copy()
            img = img.reshape(3,24,94)
            images.append(torch.from_numpy(img))
        imgs = torch.stack(images, 0)
        prebs = self.lprnet(imgs.cuda())
        prebs = prebs.cpu().detach().numpy()
        preb_labels = list()
        for i in range(prebs.shape[0]):
            preb = prebs[i, :, :]
            preb_label = list()
            for j in range(preb.shape[1]):
                preb_label.append(np.argmax(preb[:, j], axis=0))
            no_repeat_blank_label = list()
            pre_c = preb_label[0]
            if pre_c != len(CHARS) - 1:
                no_repeat_blank_label.append(pre_c)
            for c in preb_label: # dropout repeate label and blank label
                if (pre_c == c) or (c == len(CHARS) - 1):
                    if c == len(CHARS) - 1:
                        pre_c = c
                    continue
                no_repeat_blank_label.append(c)
                pre_c = c
            preb_labels.append(no_repeat_blank_label)
        labels = []
        for preb_label in preb_labels:
            label = ""
            for i in preb_label:
                label += CHARS[i]
            labels.append(label)
        print(labels)
        return lprnet_pb2.LprnetResponse(labels=labels)


def run():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    lprnet_pb2_grpc.add_LprnetServiceServicer_to_server(Service(), server)
    server.add_insecure_port('[::]:50052')
    server.start()
    print("start service...")
    try:
        while True:
            time.sleep(_ONE_DAY_IN_SECONDS)
    except KeyboardInterrupt:
        server.stop(0)


if __name__ == '__main__':
    run()