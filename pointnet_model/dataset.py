from torch.utils.data import Dataset


# TODO: complete the dataset code


class MyDataset(Dataset):
    def __init__(self, root, transform=None):
        super(MyDataset, self).__init__()
        self.root = root
        self.transform = transform

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, index):
        raise NotImplementedError


class TrainDataset(MyDataset):
    def __init__(self, *args):
        super(TrainDataset, self).__init__(*args)


class TestDataset(MyDataset):
    def __init__(self, *args):
        super(TestDataset, self).__init__(*args)
