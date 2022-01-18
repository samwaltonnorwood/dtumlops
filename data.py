import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms
import numpy as np

class CorruptedMNIST(filepath, train=True, transform=None):
    def __init__(self,transform=None):
        rootdir = filepath
        if train:
            filelist = ["{}/{}_{}.npz".format(rootdir, "train", i) for i in range(5)]
        else:
            filelist = ["{}/{}.npz".format(rootdir, "train")]
        all_data = [np.load(file) for file in filelist]
        merged_data = {}
        for data in all_data:
            [merged_data.update({k: v}) for k, v in data.items()]
        self.data_array = merged_data
        self.transform = transform
    
    def __len__(self):
        return len(self.data_array['images'])
    
    def __getitem__(self, id):
        if torch.is_tensor(id):
            id = id.tolist()
        image = self.data_array['images'][id]
        label = self.data_array['labels'][id]
        if self.transform:
            image = self.transform(image)
        sample = {'image': image, 'label': label}
        return sample

def mnist():
    transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))])
    trainset = CorruptedMNIST('/Users/samnorwood/ResearchProjects/DTU/MLOps/data/corruptmnist', train=True, transform=transform)
    train = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
    testset = CorruptedMNIST('/Users/samnorwood/ResearchProjects/DTU/MLOps/data/corruptmnist', train=False, transform=transform)
    test = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)
    return train, test

"""
rootdir = '/Users/samnorwood/ResearchProjects/DTU/MLOps/data/corruptmnist'
filelist = ["{}/{}_{}.npz".format(rootdir, "train", i) for i in range(5)]
all_data = [np.load(file) for file in filelist]
merged_data = {}
for data in all_data:
    [merged_data.update({k: v}) for k, v in data.items()]
for k, v in merged_data.items():

for i in range(1,101):
    ax = plt.subplot(10,10,i)
    index = np.random.randint(5000)
    ax.imshow(array['images'][index])
    ax.set_title(array['labels'][index])
    ax.set_yticklabels([])
    ax.set_xticklabels([])
plt.show()
"""