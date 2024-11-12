import torch
from torch import nn
import torch.nn.functional as F
from libs.models.Attention.CBAM import CBAMBlock
from torchvision.models import vgg19_bn, vgg16_bn, inception_v3, alexnet
from torchvision import transforms
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

class Basic_network(nn.Module):
    def __init__(self, base_network='vgg_16', pretrained=False,out_feature_n=64):
        super(Basic_network, self).__init__()
        # Define CNN to extract features.
        self.features = None
        if base_network == 'vgg_16':
            self.features = vgg16_bn(pretrained=pretrained)

            self.backbone = nn.Sequential(*list(self.features.children())[:-2])
            # To delete fc8
            # self.features.classifier = nn.Sequential(*list(self.features.classifier.children())[:-2])
            # print(self.features)

            # self.features.classifier = nn.Sequential(*list(self.features.classifier.children())[:-5])

            # print(self.features)

            # self.features.classifier = nn.Sequential(*list(self.features.classifier.children())[:-5])

            # self.fc = nn.Sequential(nn.Linear(4096*2, out_feature_n))
        elif base_network == 'vgg_19':
            self.features = vgg19_bn(pretrained=pretrained)
            # To delete fc8
            self.features.classifier = nn.Sequential(*list(self.features.classifier.children())[:-2])
            # self.fc = nn.Sequential(nn.Linear(4096*2, out_feature_n))
        elif base_network == 'inception_v3':
            #TODO It is unreliable.
            self.features = inception_v3(pretrained=pretrained)
            # To delete the last layer.
            # self.features.fc = nn.Sequential(*list(self.features.fc.children())[:-1])
            # self.fc = nn.Sequential(nn.Linear(2048*2, out_feature_n))
        elif base_network == 'debug':
            self.features = alexnet(pretrained=pretrained)
            # To delete the last layer
            self.features.classifier = nn.Sequential(*list(self.features.classifier.children())[:-2])
            # self.fc = nn.Sequential(nn.Linear(4096*2, out_feature_n))

        assert self.features, "Illegal CNN network name!"

    def forward(self, x_1):
        features = self.backbone(x_1)
        
        # features = self.fc(features)

        return features


class SpaceAttentionNet(nn.Module):
    def __init__(self, base_network='vgg_16', pretrained=False, out_feature_n=64, use_gpu=False, dropout=0.8,
                #  rnn_hidden_size=64, rnn_num_layers=1, num_classes=2, 
                 ):
        super(SpaceAttentionNet, self).__init__()
        self.cnn_network = Basic_network(base_network=base_network, pretrained=pretrained,out_feature_n=out_feature_n)
        self.cnn_network.eval()
       
        if base_network=="vgg_16":
            channel = 512
            kernel_size = 7

        self.visual_spatial_att = CBAMBlock(channel=channel,reduction=16,kernel_size=kernel_size)

        self.haptic_spatial_att = CBAMBlock(channel=channel,reduction=16,kernel_size=kernel_size)

        self.mlp = nn.Sequential(
            nn.Linear(2*channel*kernel_size*kernel_size, 4096),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(4096, 2048),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, 2)
            )
        
        self.visual_gate = nn.Sequential(
            nn.Linear(channel*kernel_size*kernel_size, 4096),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(4096, 2048),
            nn.Dropout(dropout),
            nn.Linear(2048, 4096),
            nn.Dropout(dropout),
            nn.Linear(4096, channel*kernel_size*kernel_size),
            nn.Sigmoid(),
            )
        
        self.haptic_gate = nn.Sequential(
            nn.Linear(channel*kernel_size*kernel_size, 4096),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(4096, 2048),
            nn.Dropout(dropout),
            nn.Linear(2048, 4096),
            nn.Dropout(dropout),
            nn.Linear(4096, channel*kernel_size*kernel_size),
            nn.Sigmoid(),
            )
        

        self.use_gpu = use_gpu

    def forward(self, x_1, x_2):
        """
        :param x_1: a list of 8 rgb imgs(tensor)
        :param x_2: a list of 8 tactile imgs(tensor)
        :return: network output
        """

        # visual
        visaul_img_feature = self.cnn_network(128+x_1[-1].to(device)-x_1[0].to(device))
        visual_spatial_featue = torch.flatten(self.visual_spatial_att(visaul_img_feature), start_dim=1)
        visual_gate = self.visual_gate(visual_spatial_featue)
        visula_feature = torch.mul(visual_spatial_featue, visual_gate)

        # haptic
        haptic_img_feature = self.cnn_network(128+x_2[-1].to(device)-x_2[0].to(device))
        haptic_spatial_featue = torch.flatten(self.haptic_spatial_att(haptic_img_feature), start_dim=1)
        haptic_gate = self.haptic_gate(haptic_spatial_featue)
        haptic_feature = torch.mul(haptic_spatial_featue, haptic_gate)

        
        # output = self.rnn_network(cnn_features)
        features = torch.cat([visula_feature, haptic_feature], dim=1)
        output = self.mlp(features)
        return output


if __name__ == "__main__":
    network = SpaceAttentionNet()
    visula_imgs = [torch.rand(3,3,224,224) for i in range(3)]
    haptic_imgs = [torch.rand(3,3,224,224) for i in range(3)]
    output = network(visula_imgs, haptic_imgs)
    print(network)
