import os
import json
from PIL import Image

import pickle
import imageio
import numpy as np
import jittor as jt
from jittor.dataset import Dataset
from .datasets_jt import register

def pil_to_jt_tensor(img_pil):
    arr = np.array(img_pil)                  # H, W, C
    if arr.ndim == 2:
        arr = arr[..., None]
    arr = arr.astype('float32') / 255.0
    arr = arr.transpose(2, 0, 1).copy()     # HWC -> CHW
    return jt.array(arr)                    # jt.Var (C, H, W), float32, [0,1]

@register('image-folder')
class ImageFolder(Dataset):
    #class jittor.dataset.Dataset(batch_size=16, shuffle=False, drop_last=False, num_workers=0, buffer_size=536870912, stop_grad=True, keep_numpy_array=False, endless=False)
    def __init__(self, root_path, split_file=None, split_key=None, first_k=None,
                 repeat=1, cache='none'):
        super().__init__()
        self.repeat = repeat
        self.cache = cache

        if split_file is None:
            filenames = sorted(os.listdir(root_path))
        else:
            with open(split_file, 'r') as f:
                filenames = json.load(f)[split_key]
        if first_k is not None:
            filenames = filenames[:first_k]

        self.files = []
        for filename in filenames:
            file = os.path.join(root_path, filename)

            if cache == 'none':
                self.files.append(file)

            elif cache == 'bin':
                bin_root = os.path.join(os.path.dirname(root_path),
                    '_bin_' + os.path.basename(root_path))
                if not os.path.exists(bin_root):
                    os.mkdir(bin_root)
                    print('mkdir', bin_root)
                bin_file = os.path.join(
                    bin_root, filename.split('.')[0] + '.pkl')
                if not os.path.exists(bin_file):
                    with open(bin_file, 'wb') as f:
                        pickle.dump(imageio.imread(file), f)
                    print('dump', bin_file)
                self.files.append(bin_file)

            elif cache == 'in_memory':
                self.files.append(pil_to_jt_tensor(
                    Image.open(file).convert('RGB')))
# 必须告诉 jittor 总长度，否则会报错
        self.set_attrs(total_len = len(self.files) * self.repeat)


    def __len__(self):
        return len(self.files) * self.repeat

    def __getitem__(self, idx):
        x = self.files[idx % len(self.files)]

        if self.cache == 'none':
            return pil_to_jt_tensor(Image.open(x).convert('RGB'))

        elif self.cache == 'bin':
            with open(x, 'rb') as f:
                x = pickle.load(f)
            # x = np.ascontiguousarray(x.transpose(2, 0, 1))
            # x = torch.from_numpy(x).float() / 255
            x = np.ascontiguousarray(x.transpose(2, 0, 1))
            x = jt.array(x).float() / 255.0
            return x

        elif self.cache == 'in_memory':
            return x


@register('paired-image-folders')
class PairedImageFolders(Dataset):

    def __init__(self, root_path_1, root_path_2, **kwargs):
        super().__init__()
        self.dataset_1 = ImageFolder(root_path_1, **kwargs)
        self.dataset_2 = ImageFolder(root_path_2, **kwargs)
# 必须告诉 jittor 总长度，否则会报错
        self.set_attrs(total_len = len(self.dataset_1))
    def __len__(self):
        return len(self.dataset_1)

    def __getitem__(self, idx):
        return self.dataset_1[idx], self.dataset_2[idx]