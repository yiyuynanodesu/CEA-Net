from torch import nn
import torch.nn.init as init
import torch
import torch.nn.functional as F
from .modules import resnet
from .modules.TextTransformer import TextBlock

from .modules.DyRoutFusion_CLS import DyRoutTrans,SentiCLS
from .modules.SparX.models import sparx_mamba_t
from .modules.WaveMixV3 import Waveblock

from timm.models.layers import DropPath, to_2tuple, trunc_normal_

# CCM
# EDAF
# AACEM

def calculate_ratio(uni_modal, multi_modal, k=2.):
    ratio = {}
    for m in ['T', 'V']:
        uni_modal[m] = torch.exp(-1 * k * torch.pow(torch.abs(uni_modal[m] - multi_modal), 2))

    # 进行归一化
    for m in ['T', 'V']:
        ratio[m] = uni_modal[m] / (uni_modal['T'] + uni_modal['V'])
        ratio[m] = ratio[m].unsqueeze(-1)
    return ratio

class EFAttention(nn.Module):
    def __init__(self, in_channels, kernel_size = 3):
        super().__init__()

        # x_c
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2)
        self.sigmoid = nn.Sigmoid()

        # x_s
        self.Conv1x1 = nn.Conv2d(in_channels, 1, kernel_size=1, bias=False)
        self.norm = nn.Sigmoid()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self,x):
        # x_c
        y1 = self.gap(x)
        y1 = y1.squeeze(-1).permute(0, 2, 1)
        y1 = self.conv(y1)
        y1 = self.sigmoid(y1)
        y1 = y1.permute(0, 2, 1).unsqueeze(-1)
        x_c =  x * y1.expand_as(x)

        # x_s
        q = self.Conv1x1(x)
        q = self.norm(q)
        x_s = x * q
        return x_c + x_s

class DyRoutTransOptions:
    def __init__(self):
        self.hidden_size = 256
        self.ffn_size = 512 
        self.seq_lens = [1, 1]

class Model(nn.Module):
    def __init__(self, num_classes, vocab_size, embedding_size):
        super(Model, self).__init__()
        self.dim = embedding_size
        # resnet
        self.pretrained_model = resnet.resnet50(pretrained=True)
        self.pretrained_model.avgpool = nn.AdaptiveAvgPool2d(1)
        self.pretrained_model.fc = nn.Linear(512 * 4, num_classes)
        # text Embedding layer
        self.embedding_layer = nn.Embedding(vocab_size, self.dim)
        # image transforer
        self.img_conv_1 = nn.Conv2d(2048, self.dim, kernel_size=1)
        self.img_conv_2 = nn.Conv2d(2*self.dim, self.dim, kernel_size=1)
        
        self.img_transformer = sparx_mamba_t(pretrained=True)
        self.img_classifier = self.img_transformer.classifier
        self.img_transformer.classifier = nn.Identity()
        self.img_classifier[-1] = nn.Identity()

        ModuleList = []
        ModuleList.append(nn.Linear(self.dim, 1))
        ModuleList.append(nn.GELU())
        self.img_view = nn.Sequential(*ModuleList)
        self.text_view = nn.Sequential(*ModuleList)

        # wave_layers
        mult= 4
        ff_channel= 96
        final_dim = 96
        dropout= 0.5
        depth = 1
        self.wave_layers = nn.ModuleList([])
        for _ in range(depth): 
            self.wave_layers.append(Waveblock(mult = mult, ff_channel = ff_channel, final_dim = final_dim, dropout = dropout))
        self.img_transformer.wave_layers = self.wave_layers
        
        # text transformer
        self.texttransformer_encoder = TextBlock(dim=self.dim)
        self.avgpool2 = nn.AdaptiveAvgPool2d(1)
        self.avgpool1 = nn.AdaptiveAvgPool1d(1)

        self.img_EFA = EFAttention(self.dim)
        self.text_EFA = EFAttention(self.dim)
        opt = DyRoutTransOptions()
        self.fusion = DyRoutTrans(opt)
    
    def forward(self, input_i, input_t, labels):
        batch = input_t.size(0)
        resnet_out, rpn_feature, res_feature = self.pretrained_model(input_i)
        rpn_feature = self.img_conv_1(rpn_feature)
        #  img layer
        img_att = self.img_transformer(input_i)
        img_att = torch.cat((rpn_feature, img_att), dim=1)
        img_att = self.img_conv_2(img_att)
        
        # text Embedding layer
        text_embedding = self.embedding_layer(input_t).permute(0, 2, 1)  # [8,128,100]
        text_feature = text_embedding.reshape(batch, self.dim, 10, 10)
        text_atten = self.texttransformer_encoder(text_feature)  # text_atten:[8,128,10,10]

        img_att = self.img_EFA(img_att)
        img_avg = self.img_classifier(img_att)
        text_atten = self.text_EFA(text_atten)
        text_avg = self.avgpool2(text_atten).view(batch, -1)  # [8,128]
        
        if labels == None:
            ratio = None
        else:
            img_modal = self.img_view(img_avg)
            text_modal = self.text_view(text_avg)
            uni_modal = {
                'T': text_modal,  # 文本特征
                'V': img_modal,    # 视觉特征
            }
            labels = labels.view(-1,1)
            ratio = calculate_ratio(uni_modal,labels,k=0.1)
        uni_fea = {
            'T': text_avg.unsqueeze(1),  # 文本特征
            'V': img_avg.unsqueeze(1),    # 视觉特征
        }
        fusion_features = self.fusion(uni_fea, ratio)
        return img_avg, res_feature, text_avg, fusion_features
    
class MAMLModel(nn.Module):
    def __init__(self, backbone, output_dim):
        super(MAMLModel, self).__init__()
        self.backbone = backbone
        self.img_classifier = nn.Linear(512,output_dim)
        self.text_classifier = nn.Linear(512,output_dim)
        self.res_classifier = nn.Linear(2048, output_dim) 
        self.mul_classifier = SentiCLS()

    def forward(self, input_i,input_t,labels):
        img_avg, res_feature, text_avg, fusion_features = self.backbone(input_i,input_t, labels)
        return self.img_classifier(img_avg), self.res_classifier(res_feature), self.text_classifier(text_avg), self.mul_classifier(fusion_features)