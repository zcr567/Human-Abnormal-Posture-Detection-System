import os
import numpy as np


class CollectedData:

    def __init__(self, path=None, data_shape=None, valid_tags=None, data_amount_magnitude=None):
        """A temporary data structure to store collected data.
The type of data should be numpy array and type of tags is int.
Below the root directory, there are every tag`s directory and a config text file. Data of the same tag are stored
in one directory, and they are in separate files for convenience. The config file holds basic properties such as
naming rules, valid tags and shape of the data.

:param path: The path of the root directory.
:param data_shape: The shape of the collected data, should be the type of tuple(int).
:param valid_tags: A list of valid tags.
:param data_amount_magnitude: Preserved. Assigning a value to it has no effect."""
        if not any((path, data_shape, valid_tags, data_amount_magnitude)):
            try:
                with open(path + "/config", "r") as f:
                    props = [eval(d[22:]) for d in f.read().splitlines()]
                    self.data_shape, self.valid_tags, self.data_amount_magnitude, self.data_amounts = props
            except FileNotFoundError:
                raise RuntimeError('Dataset filee broken or not exist.')
        else:
            self.path = path
            self.data_shape = data_shape
            self.data_amount_magnitude = data_amount_magnitude
            if valid_tags is None:
                self.valid_tags = [0, 1]
            else:
                self.valid_tags = valid_tags
            self.data_amounts = [0 for _ in range(len(self.valid_tags))]

            try:
                with open(path + "/config", "r") as f:
                    props = [eval(d[22:]) for d in f.read().splitlines()]
                    self.data_shape, self.valid_tags, self.data_amount_magnitude, self.data_amounts = props
            except FileNotFoundError:
                try:
                    self._update_config_file()
                except FileNotFoundError:
                    os.mkdir(self.path)
                    self._update_config_file()

            for name in self.valid_tags:
                try:
                    os.mkdir(f'{self.path}/{name}')
                except FileExistsError:
                    pass

    def _update_config_file(self):
        try:
            with open(self.path + "/config", "x") as f:
                f.write(f"data_shape            {self.data_shape}\n")
                f.write(f"valid_tags            {self.valid_tags}\n")
                f.write(f"data_amount_magnitude {self.data_amount_magnitude}\n")
                f.write(f"data_amounts          {self.data_amounts}\n")
        except FileExistsError:
            with open(self.path + "/config", "w") as f:
                f.write(f"data_shape            {self.data_shape}\n")
                f.write(f"valid_tags            {self.valid_tags}\n")
                f.write(f"data_amount_magnitude {self.data_amount_magnitude}\n")
                f.write(f"data_amounts          {self.data_amounts}\n")

    def get_amount(self, tag):
        return self.data_amounts[self.valid_tags.index(tag)]

    def add_data(self, data, tag):
        if not isinstance(data, np.ndarray) or data.shape != self.data_shape:
            raise ValueError(f'The data should be a numpy array with shape {self.data_shape}, not{data.shape}')
        if tag not in self.valid_tags:
            raise ValueError(f'The tag {tag} is not a valid tag.')

        try:
            np.save(self.path + f"/{tag}/{self.get_amount(tag) :05}", data)
            self.data_amounts[self.valid_tags.index(tag)] += 1
            self._update_config_file()
        except FileNotFoundError:
            pass

    def get_data(self, tag, idx):
        if tag not in self.valid_tags:
            raise ValueError(f'The tag {tag} is not a valid tag.')
        if not type(idx) is int or not -1 < idx < self.get_amount(tag):
            raise IndexError(f'The index out of range.')
        try:
            return np.load(self.path + f"/{tag}/{idx:05}.npy", allow_pickle=True)

        except FileNotFoundError:
            raise FileNotFoundError(f'This data is broken or not exist.')

    def get_all_data(self):
        raise NotImplemented

    def del_last_data(self, tag):
        try:
            os.remove(self.path + f"/{tag}/{self.get_amount(tag)-1 :05}.npy")
            self.data_amounts[self.valid_tags.index(tag)] -= 1
        except FileNotFoundError:
            pass


if __name__ == '__main__':
    test_set = CollectedData('test', data_shape=(3, 2), data_amount_magnitude=5)
    print(test_set.__dict__)
    test_set.add_data(np.array([[0, 1], [2, 3], [4, 5]]), 0)
    print(test_set.get_data(0, 1))
