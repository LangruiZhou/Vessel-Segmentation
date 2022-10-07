import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


transform = transforms.Compose([
    transforms.ToTensor()   # 把用PIL读取的图像转为Tensor
])


class getDataset(Dataset):
    def __init__(self, path):
        self.path = path    # /.../DRIVE/training
        self.manual_name = os.listdir(os.path.join(path, '1st_manual'))    # 获取1st_manual下所有文件名
        self.manual_name.sort()
        self.image_name = os.listdir(os.path.join(path, 'images'))      # 获取images下所有文件名
        self.image_name.sort()

    def __len__(self):
        return len(self.image_name)   # =='1st_manual'的文件数量

    def __getitem__(self, item):
        # 获取文件和路径
        manual_name = self.manual_name[item]
        manual_path = os.path.join(self.path, '1st_manual', manual_name)    # munual的文件路径
        image_name = self.image_name[item]
        image_path = os.path.join(self.path, 'images', image_name)
        # 读入image和manual
        manual = Image.open(manual_path)
        image = Image.open(image_path)
        # 返回归一化结果
        return transform(image), transform(manual)


# 数据增强方法（将原图切片）
def dataEnhancement():
    """
    这里保留一个"数据增强"的方法
    """
    return None


if __name__ == '__main__':
    path = '/Users/larry/Documents/GitHub/LongRoad/医学图像分割/眼底血管分割/DataSets/DRIVE/training'
    dataset = getDataset(path)

    print(dataset[0][0].shape)
    print(dataset[0][1].shape)



"""
def getDataset():
    # 设置DRIVE的训练集路径，本次实验只需要用到原图img和手动分割标签manual
    dataset_path = "/Users/larry/Documents/GitHub/LongRoad/医学图像分割/眼底血管分割/DataSets/DRIVE/training"
    img_dir = dataset_path + "/images/"
    manual_dir = dataset_path + "/1st_manual/"

    # 设置img和manual的原始尺寸，与U-Net论文保持一致
    N = 572

    # 初始化用于存放dataset的变量
    img = np.empty(())
    manual = np.empty(())

    # 获取数据集列表，读入图片
    img_list = os.listdir(img_dir)
    img_list.sort()  # os.listdir的结果是乱序的，需要单独排序，否则img和manual对不上，无法进行训练
    img_list = [img_dir+i for i in img_list]
    img = [np.asarray(Image.open(file)) for file in img_list]

    manual_list = os.listdir(manual_dir)
    manual_list.sort()
    manual_list = [manual_dir+i for i in manual_list]
    manual = [np.asarray(Image.open(file)) for file in manual_list]

    # 重新设置图片尺寸，与U-Net论文保持一致
    img = [cv2.resize(i, (N, N)) for i in img]
    manual = [cv2.resize(i, (N, N)) for i in manual]

    return [img, manual]
"""

# 获取数据集: img(572,572,3) manual(572,572)
# [img, manual] = getDataset(img_dir, manual_dir)



