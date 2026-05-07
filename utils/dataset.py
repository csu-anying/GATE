import torch
import pandas as pd
from torch.utils.data import Dataset, ConcatDataset
import numpy as np
from sklearn.model_selection import train_test_split
from torchvision.transforms import transforms
from PIL import Image
import wfdb
from tqdm import tqdm
import os


class MIMIC_ECG_Dataset(Dataset):
    def __init__(self, ecg_meta_path, transform=None, **args):
        self.ecg_meta_path = ecg_meta_path
        self.mode = args['train_test']
        if self.mode == 'train':
            self.ecg_data = os.path.join(ecg_meta_path, 'train.npy')
            self.ecg_data = np.load(self.ecg_data, 'r')
        else:
            self.ecg_data = os.path.join(ecg_meta_path, 'val.npy')
            self.ecg_data = np.load(self.ecg_data, 'r')

        self.text_csv = args['text_csv']

        self.transform = transform

    def __len__(self):
        return self.text_csv.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # we have to divide 1000 to get the real value
        ecg = self.ecg_data[idx] / 1000
        # ecg = (ecg - np.min(ecg))/(np.max(ecg) - np.min(ecg) + 1e-8)

        # get raw text
        report = self.text_csv.iloc[idx]['total_report']

        sample = {'ecg': ecg, 'raw_text': report}

        if self.transform:
            if self.mode == 'train':
                sample['ecg'] = self.transform(sample['ecg'])
                sample['ecg'] = torch.squeeze(sample['ecg'], dim=0)
            else:
                sample['ecg'] = self.transform(sample['ecg'])
                sample['ecg'] = torch.squeeze(sample['ecg'], dim=0)
        return sample


class ECG_Dsataset:
    def __init__(self, data_path, dataset_name='mimic'):
        self.data_path = data_path
        self.dataset_name = dataset_name

        print(f'Load {dataset_name} dataset!')
        self.train_csv = pd.read_csv(os.path.join(self.data_path, 'train.csv'), low_memory=False)
        self.val_csv = pd.read_csv(os.path.join(self.data_path, 'val.csv'), low_memory=False)

        print(f'train size: {self.train_csv.shape[0]}')
        print(f'val size: {self.val_csv.shape[0]}')
        print(f'total size: {self.train_csv.shape[0] + self.val_csv.shape[0]}')

    def get_dataset(self, train_test, T=None):

        if train_test == 'train':
            print('Apply Train-stage Transform!')

            Transforms = transforms.Compose([
                transforms.ToTensor(),
            ])
        else:
            print('Apply Val-stage Transform!')

            Transforms = transforms.Compose([
                transforms.ToTensor(),
            ])

        if self.dataset_name == 'mimic':

            if train_test == 'train':
                misc_args = {'train_test': train_test,
                             'text_csv': self.train_csv,
                             }
            else:
                misc_args = {'train_test': train_test,
                             'text_csv': self.val_csv,
                             }

            dataset = MIMIC_ECG_Dataset(ecg_meta_path=self.data_path,
                                        transform=Transforms,
                                        **misc_args)
            print(f'{train_test} dataset length: ', len(dataset))

        return dataset


class SSLDataSet(Dataset):
    def __init__(self, data):
        super(SSLDataSet, self).__init__()
        self.data = data

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.float)

    def __len__(self):
        return self.data.shape[0]

class FTDataSet(Dataset):
    def __init__(self, data, label, multi_label=False):
        super(FTDataSet, self).__init__()
        self.data = data
        self.label = label
        self.multi_label = multi_label

    def __getitem__(self, index):
        if self.multi_label:
            return (torch.tensor(self.data[index], dtype=torch.float), torch.tensor(self.label[index], dtype=torch.float))
        else:
            return (torch.tensor(self.data[index], dtype=torch.float), torch.tensor(self.label[index], dtype=torch.long))

    def __len__(self):
        return self.data.shape[0]