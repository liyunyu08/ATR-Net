import torch
import torch.nn as nn
import torch.nn.functional as F
from models.backbone_res12 import ResNet
from models.conv4 import ConvNet
from models.TACF import TACF
import numpy as np



class ATR_Net(nn.Module):

    def __init__(self, args, resnet=False, mode=None):
        super().__init__()
        self.mode = mode
        self.args = args
        self.resnet = resnet
        self.scale = nn.Parameter(torch.FloatTensor([1.0]), requires_grad=True)
        self.num_token = args.num_token


        if resnet:
            self.encoder_dim = 640
            self.encoder = ResNet()
            print("This is ResNet")


        else:
            self.encoder_dim  = 64
            self.encoder = ConvNet(4)
            print("This is ConvNet")


        print("descriptor:",args.num_descriptor,"num_token:" ,self.num_token)
        self.tacf = TACF(args, self.encoder_dim, 2, 1, args.num_descriptor, self.num_token)
        self.fc = nn.Linear(self.encoder_dim, self.args.num_class)

    def forward(self, input):
        if self.mode == 'fc':
            return self.fc_forward(input)

        elif self.mode == 'encoder':
            x =self.encoder(input)
            return x

        elif self.mode == 'ATR_Net':
            spt,qry = input
            ch_task_spt, ch_task_qry = self.tacf(spt, qry)


            return self.metric(ch_task_spt,ch_task_qry)


        else:
            raise ValueError('Unknown mode')



    def fc_forward(self, x):
        x = x.mean(dim=[-1,-2])
        return self.fc(x)


    def metric(self, token_support, token_query):


        qry_pooled = token_query.mean(dim=[-1])


        token_spt = self.normalize_feature(token_support)
        token_qry = self.normalize_feature(token_query)


        way = token_spt.shape[0]
        num_qry = token_qry.shape[0]


        token_spt = token_spt.unsqueeze(0).repeat(num_qry, 1, 1, 1)
        token_qry = token_qry.unsqueeze(1).repeat(1, way, 1, 1)


        spt_attended_pooled = token_spt.mean(dim=[-1])
        qry_attended_pooled = token_qry.mean(dim=[-1])

        similarity_matrix = F.cosine_similarity(spt_attended_pooled, qry_attended_pooled, dim=-1)

        logits = similarity_matrix * self.scale

        if self.training:
            return logits, self.fc(qry_pooled)
        else:
            return logits



    def normalize_feature(self, x):
        return x - x.mean(1).unsqueeze(1)

