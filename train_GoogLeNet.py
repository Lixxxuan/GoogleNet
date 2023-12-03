import torch  # 引入torch框架
import torch.nn as nn  # 引入torch.nn torch.nn是pytorch中自带的一个函数库，里面包含了神经网络中使用的一些常用函数
from torchvision import transforms, \
    datasets  # 引入torchvision，torchvision是pytorch的一个图形库，它服务于PyTorch深度学习框架的，主要用来构建计算机视觉模型。
import torchvision
import json  # JSON，全称是 JavaScript Object Notation，即 JavaScript对象标记法。
import matplotlib.pyplot as plt
import os
import torch.optim as optim
from model_GoogLeNet import GoogLeNet


def main():
    # 指定训练设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    # 预处理
    # transforms.Compose()负责将这两个对图像的操作串联起来。
    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     # 将给定图像随机裁剪为不同的大小和宽高比，然后缩放所裁剪得到的图像为制定的大小；（即先随机采集，然后对裁剪得到的图像缩放为同一大小）
                                     transforms.RandomHorizontalFlip(),  # 依据概率p对PIL图片进行水平翻转(p默认是0.5)
                                     transforms.ToTensor(),  # 先由HWC转置为CHW格式；再转为float类型；最后，每个像素除以255。
                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
                                    # 功能：逐channel的对图像进行标准化（均值变为0，标准差变为1），可以加快模型的收敛
                                    ),
        "val": transforms.Compose([transforms.Resize((224, 224)),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
                                  )}
    # 设置数据集根路径
    data_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
    # 设置数据集的路径
    image_path = os.path.join(data_root, "data_set", "flower_data")
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)  # 断言，如果文件路径不存在则报错
    train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "train"), transform=data_transform["train"])
    '''
dataset=torchvision.datasets.ImageFolder(
                       root, transform=None, 
                       target_transform=None, 
                       loader=<function default_loader>, 
                       is_valid_file=None)
root：图片存储的根目录，即各类别文件夹所在目录的上一级目录。
transform：对图片进行预处理的操作（函数），原始图片作为输入，返回一个转换后的图片。
target_transform：对图片类别进行预处理的操作，输入为 target，输出对其的转换。 如果不传该参数，即对 target 不做任何转换，返回的顺序索引 0,1, 2…
loader：表示数据集加载方式，通常默认加载方式即可。
is_valid_file：获取图像文件的路径并检查该文件是否为有效文件的函数(用于检查损坏文件)
'''

    train_num = len(train_dataset)  # 数据集的长度

    # {'daisy':0, 'dandelion':1, 'roses':2, 'sunflower':3, 'tulips':4}
    flower_list = train_dataset.class_to_idx
    cla_dict = dict((val, key) for key, val in flower_list.items())
    # 将分类字典写入文件
    json_str = json.dumps(cla_dict, indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    batch_size = 2
    # 设置worker数量
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    print('Using {} dataloader workers every process'.format(nw))

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=nw)

    validate_dataset = datasets.ImageFolder(root=os.path.join(image_path, "val"), transform=data_transform["val"])
    val_num = len(validate_dataset)
    validate_loader = torch.utils.data.DataLoader(validate_dataset, batch_size=batch_size, shuffle=False,
                                                  num_workers=nw)

    print("using {} images for training, {} images fot validation.".format(train_num, val_num))

    # 测试模块
    # test_data_iter = iter(validate_loader)
    # test_image, test_label = test_data_iter.next()

    # net = torchvision.models.googlenet(num_classes=5)
    # model_dict = net.state_dict()
    # pretrain_model = torch.load("googlenet.pth")
    # del_list = ["aux1.fc2.weight", "aux1.fc2.bias",
    #             "aux2.fc2.weight", "aux2.fc2.bias",
    #             "fc.weight", "fc.bias"]
    # pretrain_dict = {k: v for k, v in pretrain_model.items() if k not in del_list}
    # model_dict.update(pretrain_dict)
    # net.load_state_dict(model_dict)

    # 定义模型
    net = GoogLeNet(num_classes=5, aux_logits=True, init_weights=True)
    net.to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.0003)

    best_acc = 0.0
    save_path = './GoogLeNet.pth'
    for epoch in range(30):
        # 进行训练
        # 此时，self.training=True
        net.train()
        running_loss = 0.0
        for step, data in enumerate(train_loader, start=0):
            images, labels = data
            optimizer.zero_grad()
            # 在GoogLeNet模型中，有两个辅助分类器，则损失函数有三个输出
            logits, aux_logits2, aux_logits1 = net(images.to(device))
            # 计算 主分类器 与真实标签的损失
            loss0 = loss_function(logits, labels.to(device))
            # 计算 辅助分类器1 与真实标签的损失
            loss1 = loss_function(aux_logits1, labels.to(device))
            # 计算 辅助分类器2 与真实标签的损失
            loss2 = loss_function(aux_logits2, labels.to(device))
            # 根据论文要求，辅助分类器按0.3的权重，加入到总损失中
            loss = loss0 + loss1 * 0.3 + loss2 * 0.3
            # 损失反向传播
            loss.backward()
            # 优化器更新模型参数
            optimizer.step()

            # 打印阶段
            running_loss += loss.item()
            # 打印训练进程
            rate = (step + 1) / len(train_loader)
            a = "*" * int(rate * 50)
            b = "." * int((1 - rate) * 50)
            print("\rtrain loss: {:^3.0f}%[{}->{}]{:.3f}".format(int(rate * 100), a, b, loss), end="")
        print()

        # 验证模块
        # 此时，self.training=False
        net.eval()
        acc = 0.0  # 计算累计准确率
        with torch.no_grad():
            for val_data in validate_loader:
                val_images, val_labels = val_data
                # 验证模块只有一个返回结果
                outputs = net(val_images.to(device))
                predict_y = torch.max(outputs, dim=1)[1]
                acc += (predict_y == val_labels.to(device)).sum().item()
            val_accurate = acc / val_num
            if val_accurate > best_acc:
                best_acc = val_accurate
                torch.save(net.state_dict(), save_path)
            print('[epoch %d] train_loss: %.3f  test_accuracy: %.3f' %
                  (epoch + 1, running_loss / step, val_accurate))

    print('Finished Training')


if __name__ == '__main__':
    main()
