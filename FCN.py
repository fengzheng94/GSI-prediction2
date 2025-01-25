import torch
import torch.nn as nn
import torchvision.models as models

# 创建FCN模型
class FCN(nn.Module):
    def __init__(self, num_classes):
        super(FCN, self).__init__()
        pretrained_model = models.segmentation.fcn_resnet50(pretrained=True)
        self.base_model = pretrained_model.backbone
        self.model = pretrained_model.classifier
        self.model[4] = nn.Conv2d(512, num_classes, kernel_size=(1, 1))

    def forward(self, x):
        x = self.base_model(x)['out']
        x = self.model(x)
        return x

# 加载模型
model = FCN(num_classes=21)  # 设置类别数，可以根据实际情况修改

# 载入图像，预处理
input_image = torch.randn(1, 3, 224, 224)  # 假设输入图像大小为224x224
output = model(input_image)