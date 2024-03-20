import cv2
from torchvision.transforms import transforms
from torch.utils.data import Dataset


class ImageDataset(Dataset):
    def __init__(self, img_dir):
        self.img_paths = [path for path in img_dir.glob("*.jpg")]
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, item):
        img_path = self.img_paths[item]
        img = cv2.imread(str(img_path))
        if self.transform:
            img = self.transform(img)
        return img, img_path.stem
