import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
from torchvision.io import read_image
import matplotlib.pyplot as plt # conda install matplotlib


training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

labels_map = {
   0: "T-Shirt",
   1: "Trouser",
   2: "Pullover",
   3: "Dress",
   4: "Coat",
   5: "Sendal",
   6: "Shirt",
   7: "Sneaker",
   8: "Bag",
   9: "Ankle Boot",
}

figure = plt.figure(figsize=(8,8))
cols, rows = 3, 3

for i in range(1, cols * rows + 1):
   sample_idx = torch.randint(len(training_data), size=(1,)).item()
   img, label = training_data[sample_idx]
   figure.add_subplot(rows, cols, i)
   plt.title(labels_map[label])
   plt.axis("off")
   plt.imshow(img.squeeze(), cmap="gray")

# plt.show()


""" A custom Dataset class must implement three functions: __init__, __len__, and __getitem__. """

class CustomImageDataset(Dataset):
   def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
      self.img_labels = pd.read_csv(annotations_file, names=['file_name', 'label'])
      self.img_dir = img_dir
      self.transform =transform
      self.target_transform = target_transform

      """
      annotations_file: 주어진 이미지 데이터셋의 이미지 파일 이름과 레이블이 포함된 어노테이션 파일의 경로
      img_dir: 이미지 파일이 저장된 디렉토리의 경로
      transform: 이미지 데이터에 대한 변형(transform) 함수를 적용할 수 있도록 하는 PyTorch의 transforms 모듈의 객체 (기본값은 None)
      target_transform: 이미지 레이블에 대한 변형(transform) 함수를 적용할 수 있도록 하는 PyTorch의 transforms 모듈의 객체 (기본값은 None)
      """

   def __len__(self): # 데이터셋의 총 데이터 수 반환
      return len(self.img_labels) 

   def __getitem__(self, idx):
      img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
      image = read_image(img_path)
      label = self.img_labels.iloc[idx, 1]
      if self.transform:
         img = self.transform(image)
      if self.target_transform:
         label = self.target_transform(label)
      return image, label