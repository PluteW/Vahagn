import torch
from torch import nn
import torch.nn.functional as F
from libs.models.Attention.SEAttention import SEAttention
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

            self.backbone = nn.Sequential(*list(self.features.children())[:-1])
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


class TimeAttentionNet(nn.Module):
    def __init__(self, base_network='vgg_16', pretrained=False, out_feature_n=64, frame=8, cover_frame_num=2, use_gpu=False, dropout=0.8,
                #  rnn_hidden_size=64, num_classes=2, 
                 ):
        super(TimeAttentionNet, self).__init__()
        self.cnn_network = Basic_network(base_network=base_network, pretrained=pretrained,out_feature_n=out_feature_n)
        self.cnn_network.eval()
       
        if base_network=="vgg_16":
            self.channel = 512
            self.kernel_size = 7
        self.frame = frame
        self.cover_frame_num = cover_frame_num
        
        self.visual_time_att = SEAttention(frame=frame-cover_frame_num)

        self.haptic_time_att = SEAttention(frame=frame-cover_frame_num)

        self.mlp = nn.Sequential(
            nn.Linear(2*self.channel*self.kernel_size*self.kernel_size, 4096),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(4096, 2048),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 2)
            )
        
        self.visual_gate = nn.Sequential(
            nn.Linear(self.channel*self.kernel_size*self.kernel_size, 4096),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(4096, 2048),
            nn.Dropout(dropout),
            nn.Linear(2048, 4096),
            nn.Dropout(dropout),
            nn.Linear(4096, self.channel*self.kernel_size*self.kernel_size),
            nn.Sigmoid(),
            )
        
        self.haptic_gate = nn.Sequential(
            nn.Linear(self.channel*self.kernel_size*self.kernel_size, 4096),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(4096, 2048),
            nn.Dropout(dropout),
            nn.Linear(2048, 4096),
            nn.Dropout(dropout),
            nn.Linear(4096, self.channel*self.kernel_size*self.kernel_size),
            nn.Sigmoid(),
            )
        

        self.use_gpu = use_gpu

    def forward(self, x_1, x_2):
        """
        :param x_1: a list of 8 rgb imgs(tensor)
        :param x_2: a list of 8 tactile imgs(tensor)
        :return: network output
        """
        
        batch_n = x_1[0].shape[0]
        frame_n = self.frame -self.cover_frame_num
         
        visual_inputs = (torch.stack(x_1[-(self.frame-self.cover_frame_num):],0)-torch.stack(x_1[:(self.frame-self.cover_frame_num)],0)).reshape(-1,3,224,224).to(device)   # num batch channels w h 
        haptic_imputs = (torch.stack(x_2[-(self.frame-self.cover_frame_num):],0)-torch.stack(x_2[:(self.frame-self.cover_frame_num)],0)).reshape(-1,3,224,224).to(device)
        
        # visual_inputs = torch.stack(x_1,0).reshape(-1,3,224,224).to(device)   # num batch channels w h 
        # haptic_imputs = torch.stack(x_2,0).reshape(-1,3,224,224).to(device)
        
        
        # visual
        visaul_img_feature = self.cnn_network(visual_inputs).reshape(batch_n,frame_n,512,-1)
        visual_time_featue = self.visual_time_att(visaul_img_feature)
        visual_time_featue = torch.flatten(torch.mean(visual_time_featue,dim=1).squeeze(1),start_dim=1)
        visual_gate = self.visual_gate(visual_time_featue)
        visula_feature = torch.mul(visual_time_featue, visual_gate)

        # haptic
        haptic_img_feature = self.cnn_network(haptic_imputs).reshape(batch_n,frame_n,512,-1)
        haptic_time_featue = self.haptic_time_att(haptic_img_feature)
        haptic_time_featue = torch.flatten(torch.mean(haptic_time_featue,dim=1).squeeze(1),start_dim=1)
        haptic_gate = self.haptic_gate(haptic_time_featue)
        haptic_feature = torch.mul(haptic_time_featue, haptic_gate)

        
        # output = self.rnn_network(cnn_features)
        features = torch.cat([visula_feature, haptic_feature], dim=1)
        output = self.mlp(features)
        return output


if __name__ == "__main__":
    network = TimeAttentionNet()
    visula_imgs = [torch.rand(3,3,224,224) for i in range(8)]
    haptic_imgs = [torch.rand(3,3,224,224) for i in range(8)]
    output = network(visula_imgs, haptic_imgs)
    print(network)
