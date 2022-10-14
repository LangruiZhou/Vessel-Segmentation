import os
import cv2
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.utils import save_image


trans = transforms.Compose([transforms.ToTensor()])
save_path = 'splited_image'


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
        manual = Image.open(manual_path).convert('L')
        image = Image.open(image_path)
        manual = manual.resize(size=(576, 576))
        image = image.resize(size=(576, 576))
        # PIL转为tensor格式
        # image = trans(image)
        # manual = trans(manual)

        return image, manual


def image_split(image, image_type, image_name):
    for left in range(0, 512, 64):
        for up in range(0, 512, 64):
            image_train = image.crop([left, up, left+128, up+128])
            image_train = trans(image_train)
            save_image(image_train, f'{save_path}/{image_type}/{image_name}_{(left*8+up)//64}.png')


if __name__ == '__main__':
    path = r'C:\Users\depth\Desktop\眼底图像血管分割\Datasets\DRIVE\training'
    dataset = getDataset(path)

    for i in range(0, 19):
        # dataset[i][0]:image, dataset[i][1]:manual
        image_split(dataset[i][0], 'image', 'image'+str(i))
        image_split(dataset[i][1], 'manual', 'manual'+str(i))
