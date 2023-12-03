import torch
from model_GoogLeNet import GoogLeNet
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import json

data_transform = transforms.Compose(
    [transforms.Resize((224, 224)),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# 加载图像
img = Image.open("./predict01.jpg")
plt.imshow(img)
# [N, C, H, W]
img = data_transform(img)
# 扩展维度
img = torch.unsqueeze(img, dim=0)

# 加载分类字典
try:
    json_file = open('./class_indices.json', 'r')
    class_indict = json.load(json_file)
except Exception as e:
    print(e)
    exit(-1)

# 创建模型
# 预测时，创建模型，不需要辅助分类器，故aux_logits=False
model = GoogLeNet(num_classes=5, aux_logits=False)
# 加载模型权重
model_weight_path = "./GoogLeNet.pth"
# 训练的时候，保存参数，辅助分类器的权重，也保存进此文件了
# 设置strict=False，即不执行严格的模块匹配
missing_keys, unexpected_keys = model.load_state_dict(torch.load(model_weight_path), strict=False)
model.eval()
with torch.no_grad():
    # 预测分类
    output = torch.squeeze(model(img))
    predict = torch.softmax(output, dim=0)
    predict_cla = torch.argmax(predict).numpy()
# 打印分类结果
print(class_indict[str(predict_cla)])
plt.show()

