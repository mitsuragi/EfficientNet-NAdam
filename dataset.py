from torch.utils.data import Dataset, DataLoader
import torch
from torchvision.transforms import v2 as tfs
from PIL import Image
import os
import json

def gen_format(input_path):
    categories = os.listdir(os.path.join(input_path, 'train'))

    class_dict = {category: index for index, category in enumerate(categories)}

    with open(os.path.join(input_path, 'format.json'), 'w') as f:
        json.dump(class_dict, f, indent=4)

class DogsDataset(Dataset):
    def __init__(self, images_dir, transform, split='train'):
        if split not in ['train', 'evaluate', 'test']:
            raise ValueError('Only train, evaluate and test is allowed')

        self.path = os.path.join(images_dir, split)
        self.transform = transform

        with open(os.path.join(images_dir, 'format.json'), 'r') as fp:
            self.format = json.load(fp)

        self.length = 0
        self.files = []
        self.targets = torch.eye(len(self.format.values()))

        for _dir, _target in self.format.items():
            path = os.path.join(self.path, _dir)
            list_files = os.listdir(path)
            self.length += len(list_files)
            self.files.extend(map(lambda _x: (os.path.join(path, _x), _target), list_files))

        print(f"Загружено {self.length} изображений из {len(self.format)} классов для {split}")

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        path_file, target = self.files[item]
        idx = torch.argmax(self.targets[target]).item()
        img = Image.open(path_file).convert('RGB')

        if self.transform:
            img = self.transform(img)

        return img, idx

    def get_num_classes(self):
        return len(self.format.items())


def get_dataloaders(root_dir, batch_size, img_size, num_workers):
    if not os.path.exists(root_dir):
        raise FileNotFoundError(f'Директория {root_dir} отсутствует')

    gen_format(root_dir)

    train_transforms = tfs.Compose([
        tfs.Resize((img_size, img_size)),
        tfs.RandomHorizontalFlip(p=0.5),
        tfs.RandomRotation(degrees=10),
        tfs.ColorJitter(brightness=0.2, contrast=0.2),
        tfs.ToImage(),
        tfs.ToDtype(torch.float32),
        tfs.Normalize([0.485, 0.456, 0.406],
                      [0.229, 0.224, 0.225])
    ])

    eval_transforms = tfs.Compose([
        tfs.Resize((img_size, img_size)),
        tfs.ToImage(),
        tfs.ToDtype(torch.float32),
        tfs.Normalize([0.485, 0.456, 0.406],
                      [0.229, 0.224, 0.225])
    ])

    train_dataset = DogsDataset(root_dir, train_transforms, 'train')
    eval_dataset = DogsDataset(root_dir, eval_transforms, 'evaluate')
    test_dataset = DogsDataset(root_dir, eval_transforms, 'test')

    train_dataloader = DataLoader(train_dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=num_workers)
    eval_dataloader = DataLoader(eval_dataset,
                                 batch_size=batch_size,
                                 shuffle=False,
                                 num_workers=num_workers)

    test_dataloader = DataLoader(test_dataset,
                                 batch_size=batch_size,
                                 shuffle=False,
                                 num_workers=num_workers)
   
    return train_dataloader, eval_dataloader, test_dataloader, train_dataset.get_num_classes()
