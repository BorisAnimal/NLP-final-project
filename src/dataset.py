from torch.utils import data


class BinaryDataset(data.Dataset):
    def __init__(self, x, y):
        """
        :param x: np.array
        :param y: np.array
        """
        self.x = x
        self.y = y.astype(int)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
