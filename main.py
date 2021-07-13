import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import torch
import os
from torchvision import datasets
from torch.utils.data import Dataset
import pandas as pd
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import pandas as pd

class DogImageDataset(Dataset):
    def __init__(self, img_dir, transform=None, target_transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        self.imgs_labels = self._read_imgs_labels()

    def _read_imgs_labels(self):
        img_label = []
        for _ in os.listdir(self.img_dir):
            label = int(_.split(".")[0])-1
            for img_path in os.listdir(f"{self.img_dir}/{_}"):
                 img_label.append((label,f"{self.img_dir}/{_}/{img_path}"))
        return img_label

    def __len__(self):
        return len(self.imgs_labels)

    def __getitem__(self, idx):
        label, img_path = self.imgs_labels[idx]
        image = Image.open(img_path)
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label



# define the CNN architecture

class Net(nn.Module):
    def __init__(self, output_dim=131):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1), #in_channels, out_channels, kernel_size, stride, padding
            nn.MaxPool2d(2), #kernel_size
            nn.ReLU(inplace = True), # 4
            nn.Conv2d(32, 64, 3, padding = 1),
            nn.MaxPool2d(2),
            nn.ReLU(inplace = True),# 8
            nn.Conv2d(64, 128, 3, padding = 1),
            nn.MaxPool2d(2),
            nn.ReLU(inplace = True),# 16
            nn.Conv2d(128, 256, 3, padding = 1),
            nn.MaxPool2d(2),
            nn.ReLU(inplace = True)# 32
        )
        size = 2
        h_num_1 = 1024
        h_num_2 = 512
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * size * size, h_num_1),
            nn.ReLU(inplace = True),
            nn.Dropout(0.5),
            nn.Linear(h_num_1, h_num_2),
            nn.ReLU(inplace = True),
            nn.Linear(h_num_2, 133),
        )

    def forward(self, x):
        ## Define forward behavior
        x = self.features(x)
        h = x.view(x.shape[0], -1)
        x = self.classifier(h)
        return x



# instantiate the CNN
model_scratch = Net()
print("created model")


image_size = 64
image_resize = 80
def compute_normalization(img_dir):
    dataset = DogImageDataset(img_dir = img_dir,
                              transform = transforms.Compose([transforms.Resize(image_resize),
                                                              transforms.CenterCrop(image_size),
                                                              transforms.ToTensor()]))
    loader = DataLoader(
        dataset,
        batch_size=10,
        num_workers=1,
        shuffle=False
    )
    mean = 0.
    std = 0.
    nb_samples = 0.
    max_target = []
    for batch_samples in loader:
        data, target = batch_samples
        max_target.extend(target)
        batch_samples = data.size(0)
        data = data.view(batch_samples, data.size(1), -1)
        mean += data.mean(2).sum(0)
        std += data.std(2).sum(0)
        nb_samples += batch_samples

    mean /= nb_samples
    std /= nb_samples
    df = pd.DataFrame(max_target, columns=["colummn"])
    df.to_csv('list.csv', index=False)
    print(max(max_target))
    return mean.squeeze().tolist(), std.squeeze().tolist()

mean,std = compute_normalization("dogImages/train")
print(mean)
transform_train = transforms.Compose([transforms.RandomHorizontalFlip(),
                                transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
                                transforms.RandomAffine(degrees=[-10, 10], translate=(0.1, 0.1), scale=(0.5, 1.5)),
                                transforms.Resize(image_resize),
                                transforms.RandomCrop(image_size,padding=0, pad_if_needed=True),
                                transforms.ToTensor(),
                                transforms.Normalize(mean = mean, std = std)])
transform = transforms.Compose([transforms.Resize(image_resize),
                                transforms.CenterCrop(image_size),
                                transforms.ToTensor(),
                                transforms.Normalize(mean = mean, std = std)])

training_data = DogImageDataset(
    img_dir="dogImages/train",
    transform=transform_train
)
test_data = DogImageDataset(
    img_dir="dogImages/test",
    transform=transform
)
validation_data = DogImageDataset(
    img_dir="dogImages/valid",
    transform=transform
)

train_dataloader  = DataLoader(training_data, batch_size=4, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=4, shuffle=True)
valid_dataloader = DataLoader(validation_data, batch_size=1, shuffle=False)
loaders_scratch = {"train": train_dataloader, "test": test_dataloader, "valid": valid_dataloader}
print("data loaded.")

import torch.optim as optim

### TODO: select loss function
criterion_scratch = nn.CrossEntropyLoss()

### TODO: select optimizer
optimizer_scratch = optim.SGD(model_scratch.parameters(), lr=0.01, momentum=0.9)

import numpy as np
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def train(n_epochs, loaders, model, optimizer, criterion, use_cuda, save_path):
    """returns trained model"""
    # initialize tracker for minimum validation loss
    valid_loss_min = np.Inf

    for epoch in range(1, n_epochs+1):
        # initialize variables to monitor training and validation loss
        train_loss = 0.0
        valid_loss = 0.0
        num_train = 0
        num_valid = 0
        ###################
        # train the model #
        ###################
        model.train()
        for batch_idx, (data, target) in enumerate(loaders['train']):
            print("Train batch:",batch_idx,target)
            # move to GPU
            if use_cuda:
                data, target = data.cuda(), target.cuda()
            ## find the loss and update the model parameters accordingly
            ## record the average training loss, using something like
            ## train_loss = train_loss + ((1 / (batch_idx + 1)) * (loss.data - train_loss))
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            # print statistics
            num_train += len(data)
            train_loss += loss.item()


        ######################
        # validate the model #
        ######################
        model.eval()
        for batch_idx, (data, target) in enumerate(loaders['valid']):
            # move to GPU
            if use_cuda:
                data, target = data.cuda(), target.cuda()
            ## update the average validation loss
            output = model(data)
            loss = criterion(output, target)
            num_valid += len(data)
            valid_loss += loss.item()


        # print training/validation statistics
        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
            epoch,
            train_loss / num_train,
            valid_loss / num_valid
            ))

        ## TODO: save the model if validation loss has decreased
        if valid_loss_min > valid_loss:
            valid_loss_min = valid_loss
            torch.save(model.state_dict(), save_path)
    # return trained model
    return model

use_cuda = torch.cuda.is_available()

# train the model
model_scratch = train(100, loaders_scratch, model_scratch, optimizer_scratch,
                      criterion_scratch, use_cuda, 'model_scratch.pt')

# load the model that got the best validation accuracy
model_scratch.load_state_dict(torch.load('model_scratch.pt'))