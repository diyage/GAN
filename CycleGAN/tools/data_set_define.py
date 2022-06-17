from torch.utils.data import Dataset
import os
import PIL.Image as Image


class DoubleInputDataSet(Dataset):
    def __init__(self,
                 root: str,
                 transform=None,
                 training: bool = True,
                 training_rate: float = 0.7):
        super().__init__()
        self.root = root
        self.transform = transform
        self.training = training
        self.training_rate = training_rate
        self.images_path = self.__get_images_path()

    def __get_images_path(self):
        def cut_and_split(a: list, b: list):
            number = int(min(len(a), len(b)))
            a_ = a[0: number]
            b_ = b[0: number]
            train_number = int(number * self.training_rate)

            if self.training:
                return a_[0: train_number], b_[0: train_number]
            else:
                return a_[train_number:], b_[train_number:]

        style_a_and_b = os.listdir(self.root)

        style_a_path = os.path.join(self.root, style_a_and_b[0])
        style_b_path = os.path.join(self.root, style_a_and_b[1])

        images_a_names = os.listdir(style_a_path)
        images_b_names = os.listdir(style_b_path)

        images_a_names, images_b_names = cut_and_split(images_a_names, images_b_names)

        res = [
            [os.path.join(style_a_path, image_name) for image_name in images_a_names],
            [os.path.join(style_b_path, image_name) for image_name in images_b_names],
        ]

        return res

    def __len__(self):
        return len(self.images_path[0])

    def __getitem__(self, index):
        image_a_path, image_b_path = self.images_path[0][index], self.images_path[1][index]

        a = Image.open(image_a_path)
        b = Image.open(image_b_path)

        a_ = self.transform(a)
        b_ = self.transform(b)

        return a_, b_

