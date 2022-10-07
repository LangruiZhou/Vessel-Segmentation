import os
import cv2
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

trans = transforms.Compose([transforms.ToTensor()])


class getDataset(Dataset):
    def __init__(self, path):
        self.path = path    # /.../DRIVE/training
        self.manual_name = os.listdir(os.path.join(path, 'manual'))    # 获取manual下所有文件名
        self.manual_name.sort()
        self.image_name = os.listdir(os.path.join(path, 'image'))      # 获取image下所有文件名
        self.image_name.sort()

    def __len__(self):
        return len(self.image_name)   # =='1st_manual'的文件数量

    def __getitem__(self, item):
        # 获取文件和路径
        manual_name = self.manual_name[item]
        manual_path = os.path.join(self.path, 'manual', manual_name)    # manual的文件路径
        image_name = self.image_name[item]
        image_path = os.path.join(self.path, 'image', image_name)
        # 读入image和manual
        manual = Image.open(manual_path).convert('L')
        image = Image.open(image_path)
        length = max(image.size)
        manual = manual.resize(size=(length, length))
        image = image.resize(size=(length, length))
        # PIL转为tensor格式
        image = trans(image)
        manual = trans(manual)

        return image, manual


if __name__ == '__main__':
    path = r'C:\Users\depth\Desktop\眼底图像血管分割\Code v2.0\splited_image'
    dataset = getDataset(path)

    print(dataset.manual_name[0])
    print(dataset[6][1].shape)


