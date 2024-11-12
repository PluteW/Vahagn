import sys
sys.path.append(".")
sys.path.append("..")
sys.path.append("...")
import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
from libs.models.Attention.SEAttention import SEAttention
from libs.models.Attention.MutliHeadAttention import SelfAttention
from libs.models.Attention.CBAM import CBAMBlock, ChannelAttention
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

def to_2tuple(x):
    if isinstance(x, int):
        return (x, x)
    elif isinstance(x, tuple):
        return x

class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=56, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x
    

class TrainablePositionEncoding(nn.Module):
    def __init__(self, max_sequence_length, d_embed):
        super().__init__()
        self.max_sequence_length = max_sequence_length
        self.d_embed = d_embed
        self.pe = nn.Embedding(self.max_sequence_length, self.d_embed)
        nn.init.constant(self.pe.weight, 0.)

    def forward(self,x):
        
        positions = torch.arange(self.max_sequence_length).to(device).unsqueeze(0)
        position_embeddings = self.pe(positions)
        return x + position_embeddings 


class Vahagn(nn.Module):
    def __init__(self, 
                #  base_network='vgg_16', pretrained=False, out_feature_n=64, 
                 frame=8, cover_frame_num=2, use_gpu=False, dropout=0.8,space_patch_size=16,sapce_patch_embed_dim=768,time_embed_dim=1024):
        super(Vahagn, self).__init__()
        # self.cnn_network = Basic_network(base_network=base_network, pretrained=pretrained,out_feature_n=out_feature_n)
        # self.cnn_network.eval()
       
        # if base_network=="vgg_16":
        #     self.channel = 512
        #     self.kernel_size = 7

        self.frame = frame
        self.cover_frame_num = cover_frame_num

        # MTAG space 定义
        self.sapce_patch_embed_dim = sapce_patch_embed_dim
        self.space_patch_size = space_patch_size
        self.space_patch_num = (224//space_patch_size)**2
        self.sapce_patch_layer = PatchEmbed(224,space_patch_size,3,sapce_patch_embed_dim)
        self.space_position_embededing = TrainablePositionEncoding(self.space_patch_num,sapce_patch_embed_dim)

        # 空间内单模态注意力
        self.visual_space_att = SEAttention(self.space_patch_num)
        # CBAMBlock(channel=self.patch_num,reduction=16,kernel_size=space_patch_size)
        self.haptic_space_att = SEAttention(self.space_patch_num)
        self.space_norm = nn.LayerNorm(normalized_shape=self.space_patch_num)

        # 空间内的模态输入门
        self.space_visual_gate = nn.Sequential(
            nn.Linear(self.sapce_patch_embed_dim*self.space_patch_num, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            # nn.Linear(4096, 2048),
            # nn.Dropout(dropout),
            nn.Linear(512, self.sapce_patch_embed_dim*self.space_patch_num),
            nn.Sigmoid(),
            )

        self.space_haptic_gate = nn.Sequential(
            nn.Linear(self.sapce_patch_embed_dim*self.space_patch_num, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            # nn.Linear(4096, 2048),
            # nn.Dropout(dropout),
            nn.Linear(512, self.sapce_patch_embed_dim*self.space_patch_num,),
            nn.Sigmoid(),
            )

        # 空间内的模态融合
        self.space_mlp = nn.Sequential(
            nn.Linear(2*self.sapce_patch_embed_dim*self.space_patch_num, 512), 
            # nn.ReLU(),
            # nn.Dropout(dropout),
            # nn.Linear(512, 1024),
            # nn.ReLU(),
            # nn.Dropout(dropout),
            # nn.Linear(2048, 512),
            nn.ReLU()
            )
        # 空间内的模态融合输出门
        self.space_gate = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 512),
            nn.Sigmoid(),
            )
        
        # MTAG time 定义
        self.time_embed_dim = time_embed_dim
        self.time_patch_size = 224
        self.time_patch_num = 1
        self.time_patch_layer = PatchEmbed(224,self.time_patch_size,3,time_embed_dim)
        self.time_position_embededing = TrainablePositionEncoding(frame-cover_frame_num,time_embed_dim)
        
        # 时间内单模态注意力
        d_o = 512
        self.visual_time_att = SelfAttention(n_head=1, d_k=128, d_v=128, d_x=time_embed_dim, d_o=d_o)
        self.haptic_time_att = SelfAttention(n_head=1, d_k=128, d_v=128, d_x=time_embed_dim, d_o=d_o)

        self.viasual_time_att_flaten = nn.Linear(d_o*(self.frame-self.cover_frame_num),1024)

        self.haptic_time_att_flaten = nn.Linear(d_o*(self.frame-self.cover_frame_num),1024)

        # 时间内的模态输入门
        self.time_visual_gate = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            # nn.Linear(4096, 2048),
            # nn.Dropout(dropout),
            nn.Linear(512, 1024),
            nn.Sigmoid(),
            )

        self.time_haptic_gate = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            # nn.Linear(4096, 2048),
            # nn.Dropout(dropout),
            nn.Linear(512, 1024),
            nn.Sigmoid(),
            )

        # 时间内的模态融合
        self.time_mlp = nn.Sequential(
            nn.Linear(2*1024, 1024),   # 16384
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, 512),
            # nn.ReLU(),
            # nn.Dropout(dropout),
            # nn.Linear(2048, 512),
            nn.ReLU()
            )
        # 时间内的模态融合输出门
        self.time_gate = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 512),
            nn.Sigmoid(),
            )
        
        # MTAG fusion 定义
        self.mlp = nn.Sequential(
            nn.Linear(1024, 512),
            # nn.ReLU(),
            # nn.Dropout(dropout),
            # nn.Linear(4096, 2048),
            # nn.ReLU(),
            # nn.Dropout(dropout),
            # nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(512, 2)
            )
        

        self.use_gpu = use_gpu

    # def label_emb(self, patch_num):
    #     position_enc = np.array([
    #         [pos / np.power(10000, 2 * (j // 2) / self.patch_num) for j in range(self.channel)]
    #         if pos!= 0 else np.zeros(self.channel) for pos in range(patch_num)])


    def forward(self, x_1, x_2):
        """
        :param x_1: a list of 8 rgb imgs(tensor)
        :param x_2: a list of 8 tactile imgs(tensor)
        :return: network output
        """
        
        batch_n = x_1[0].shape[0]

        # MTAG Space 推算
        visual_space_inputs = (x_1[-1]-x_1[-2]).reshape(-1,3,224,224).to(device)
        haptic_space_imputs = (x_2[-1]-x_2[-2]).reshape(-1,3,224,224).to(device)
        # space_poition_emb = self.space_position_embededing()

        visual_patchs = self.sapce_patch_layer(visual_space_inputs)
        visual_patchs_add_position = self.space_position_embededing(visual_patchs)
        
        haptic_patchs = self.sapce_patch_layer(haptic_space_imputs)
        haptic_patchs_add_position = self.space_position_embededing(haptic_patchs)   # B self.patch_num self.sapce_patch_embed_dim

        # visual_space_feature = visual_patchs*self.visual_space_att(torch.unsqueeze(visual_patchs,2)).squeeze(2) + visual_patchs   # B self.patch_num self.sapce_patch_embed_dim
        # haptic_space_feature = haptic_patchs*self.visual_space_att(torch.unsqueeze(haptic_patchs,2)).squeeze(2) + haptic_patchs
        
        visual_space_feature = self.visual_space_att(torch.unsqueeze(visual_patchs_add_position,2)).squeeze(2)   # B self.patch_num self.sapce_patch_embed_dim
        haptic_space_feature = self.haptic_space_att(torch.unsqueeze(haptic_patchs_add_position,2)).squeeze(2)

        # MTODO 对不对？
        visual_space_feature_ = visual_space_feature.transpose(1,2)   # B self.sapce_patch_embed_dim self.space_patch_num 
        haptic_space_feature_ = haptic_space_feature.transpose(1,2)  

        visual_space_feature_flatten = torch.flatten(self.space_norm(visual_space_feature_), start_dim=1)    # B （self.sapce_patch_embed_dim x self.patch_num)
        haptic_space_feature_flatten = torch.flatten(self.space_norm(haptic_space_feature_), start_dim=1)

        visual_space_gate = self.space_visual_gate(visual_space_feature_flatten)
        haptic_space_gate = self.space_haptic_gate(haptic_space_feature_flatten)

        visual_space_feature = visual_space_feature_flatten*visual_space_gate
        haptic_space_feature = haptic_space_feature_flatten*haptic_space_gate
        
        space_feature = self.space_mlp(torch.cat((visual_space_feature, haptic_space_feature),dim=1))   # B 2048
        space_gate = self.space_gate(space_feature)

        # space_feature_out = torch.mul(space_feature, space_gate)      # B 2048
        space_feature_out = space_feature * space_gate

        # MTAG Time 推算
        # visual_inputs = torch.stack(x_1,0).reshape(-1,3,224,224).to(device)   # num batch channels w h 
        # haptic_imputs = torch.stack(x_2,0).reshape(-1,3,224,224).to(device)
        
        frame_n = self.frame -self.cover_frame_num

        visual_time_inputs = (torch.stack(x_1[-frame_n:],0)-torch.stack(x_1[:frame_n],0)).reshape(-1,3,224,224).to(device)   # batch*num channels w h 
        haptic_time_imputs = (torch.stack(x_2[-frame_n:],0)-torch.stack(x_2[:frame_n],0)).reshape(-1,3,224,224).to(device)

        # time_position_embededing = self.time_position_embededing()  # frame_n time_embed_dim

        # from torchsummary import summary
        # summary(model=self.time_patch_layer.to("cuda:0"),input_size=haptic_time_imputs.size(),batch_size=1,device="cuda")
        
        visual_time_patchs = self.time_patch_layer(visual_time_inputs)
        visual_time_patchs_ = visual_time_patchs.reshape(batch_n,frame_n,self.time_embed_dim)
        visual_time_patchs = self.time_position_embededing(visual_time_patchs_)  # B frame_n time_embed_dim
        
        haptic_time_patchs = self.time_patch_layer(haptic_time_imputs)
        haptic_time_patchs_ = haptic_time_patchs.reshape(batch_n,frame_n,self.time_embed_dim)
        haptic_time_patchs =  self.time_position_embededing(haptic_time_patchs_)
        
 
        # summary(model=self.time_patch_layer.to("cuda:0"),input_size=haptic_time_imputs.size(),batch_size=1,device="cuda")

        visual_time_feature_flatten = self.viasual_time_att_flaten(torch.flatten(self.visual_time_att(visual_time_patchs)[1],start_dim=1))  # B frame_n*time_embed_dim
        haptic_time_feature_flatten = self.haptic_time_att_flaten(torch.flatten(self.haptic_time_att(haptic_time_patchs)[1],start_dim=1))

        
        # summary(model=self.haptic_time_att.to("cuda:0"),input_size=haptic_time_patchs.unsqueeze(2).size(),batch_size=1,device="cuda")
        
        visual_time_gate = self.time_visual_gate(visual_time_feature_flatten)   # # B frame_n*time_embed_dim
        haptic_time_gate = self.time_haptic_gate(haptic_time_feature_flatten)

        # print(visual_time_feature_flatten.shape, visual_time_gate.shape)
        visual_time_feature_out = visual_time_feature_flatten*visual_time_gate  # B frame_n*time_embed_dim
        haptic_time_feature_out = haptic_time_feature_flatten*haptic_time_gate

        time_feature = self.time_mlp(torch.cat((visual_time_feature_out, haptic_time_feature_out),dim=1))   # B 2048

        time_gate = self.time_gate(time_feature)
        time_feature_out = time_feature*time_gate
        
        # MTAG fusion 推算
        # output = self.rnn_network(cnn_features)
        features = torch.cat((space_feature_out, time_feature_out), dim=1)
        output = self.mlp(features)
                   
        
        return output


if __name__ == "__main__":

    
    
    tp = TrainablePositionEncoding(196,768)
    x = tp()

    pe = PatchEmbed(224,16,3)
    image = torch.rand(3,3,224,224)
    x = pe(image)   # 3 196 198

    network = Vahagn()
    visula_imgs = [torch.rand(3,3,224,224) for i in range(8)]
    haptic_imgs = [torch.rand(3,3,224,224) for i in range(8)]
    output = network(visula_imgs, haptic_imgs)
    print(network)
