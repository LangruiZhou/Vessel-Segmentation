import os.path
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from dataset import *
from model_unetpp import *
from model_resunet import *

# 定义设备 gpu/cpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 设置权重地址
# weight_path = 'parameters/unetpp.pth'
weight_path = 'parameters/resunet.pth'

# 数据集地址
# data_path = r'D:\Document\眼底图像血管分割\Datasets\splited_image_128'
data_path = r'D:\Document\眼底图像血管分割\Datasets\DRIVE\training'

# 训练效果图片保存路径
save_path = 'train_image'

if __name__ == '__main__':
    print(device)
    data_loader = DataLoader(getDataset(data_path), batch_size=1, shuffle=True)
    # 载入模型
    # model = UNet().to(device)
    # model = Unetpp(num_classes=1, input_channels=3, deep_supervision=False).to(device)
    model = ResUnet(3).to(device)

    if os.path.exists(weight_path):
        model.load_state_dict(torch.load(weight_path))
        print('Weight-load is successful!')
    else:
        print('Weight-load fails!')

    # 创建优化器
    optimizer = optim.Adam(model.parameters())
    # 构建损失函数
    loss_func = nn.BCELoss()

    epoch = 1   # 轮次
    while True:
        # 遍历数据集
        for i, (image, manual) in enumerate(data_loader):
            # 把获取的数据集放到设备上
            image = image.to(device)
            manual = manual.to(device)
            # 得到输出图像
            out_image = model(image)
            # print("out_image size:", out_image.shape)
            # 计算训练损失
            train_loss = loss_func(out_image, manual)
            # 梯度下降
            optimizer.zero_grad()   # 清空梯度
            train_loss.backward()   # 反向传播
            optimizer.step()        # 更新梯度

            # 每隔5次，打印一次权重
            if i % 5 == 0:
                print(f'{epoch}-{i}-loss = {train_loss.item()}')
            # 每隔20次，保存一次权重
            if i % 20 == 0:
                torch.save(model.state_dict(), weight_path)  # 获取模型参数，保存到weight_path

            # 训练效果检测
            _manual = manual[0]
            _out_image = out_image[0]
            # 拼接manual和out_image，对比训练效果
            img_compare = torch.stack([_manual, _out_image], dim=0)
            # save_image(img_compare, f'{save_path}/epoch{epoch}/{i}.png')  # 将拼接后的图片保存到save_path
            save_image(img_compare, f'{save_path}/test/{i}.png')
        epoch += 1  # 一轮结束，epoch+1








