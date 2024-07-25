import os
from torch.utils.data import Dataset
import torch
import numpy as np
import trimesh
import cv2
import glob
import matplotlib.image as mpimg
import re
import open3d as o3d


# This is special function used for reading NYU pgm format
# as it is written in big endian byte order.
def read_nyu_pgm(filename, byteorder='>'):
    with open(filename, 'rb') as f:
        buffer = f.read()
    try:
        header, width, height, maxval = re.search(
            b"(^P5\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n]\s)*)", buffer).groups()
    except AttributeError:
        raise ValueError("Not a raw PGM file: '%s'" % filename)
    img = np.frombuffer(buffer,
                        dtype=byteorder + 'u2',
                        count=int(width) * int(height),
                        offset=len(header)).reshape((int(height), int(width)))
    img_out = img.astype('u2')
    return img_out


class CamRecording(Dataset):
    def __init__(self,
                 root_dir,
                 traj_file=None,
                 rgb_transform=None,
                 depth_transform=None,
                 col_ext=".jpg"):

        self.Ts = None
        if traj_file is not None:
            self.Ts = np.loadtxt(traj_file, delimiter=" ").reshape(-1, 4, 4)
        self.root_dir = root_dir
        all_files = glob.glob(f"{self.root_dir}depth*")
        self.num_files = len(all_files)
        self.rgb_transform = rgb_transform
        self.depth_transform = depth_transform
        self.col_ext = col_ext

    def __len__(self):
        try:
            return self.Ts.shape[0]
        except:
            return self.num_files

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        s = f"{idx:06}"  # int variable
        depth_file = self.root_dir + "depth" + s + ".png"
        rgb_file = self.root_dir + "frame" + s + self.col_ext

        depth = cv2.imread(depth_file, -1)
        image = cv2.imread(rgb_file)

        T = None
        if self.Ts is not None:
            T = self.Ts[idx]

        sample = {"image": image, "depth": depth, "T": T}

        if self.rgb_transform:
            sample["image"] = self.rgb_transform(sample["image"])

        if self.depth_transform:
            sample["depth"] = self.depth_transform(sample["depth"])

        return sample


class PpmPgmRecording(Dataset):
    def __init__(self,
                 root_dir,
                 traj_file=None,
                 rgb_transform=None,
                 depth_transform=None,
                 col_ext=".ppm"):

        self.Ts = None
        if traj_file is not None:
            self.Ts = np.loadtxt(traj_file, delimiter=" ").reshape(-1, 4, 4)
        self.root_dir = root_dir
        self.all_depth_files = sorted(glob.glob(f"{self.root_dir}*.pgm"))
        self.all_rgb_files = sorted(glob.glob(f"{self.root_dir}*.ppm"))
        self.num_files = len(self.all_rgb_files)
        self.rgb_transform = rgb_transform
        self.depth_transform = depth_transform
        self.col_ext = col_ext

    def __len__(self):
        try:
            return self.Ts.shape[0]
        except:
            return self.num_files

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        depth_file = self.all_depth_files[idx]
        rgb_file = self.all_rgb_files[idx]

        image = mpimg.imread(rgb_file)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        depth = read_nyu_pgm(depth_file)

        T = None
        if self.Ts is not None:
            T = self.Ts[idx]

        sample = {"image": image, "depth": depth, "T": T}

        if self.rgb_transform:
            sample["image"] = self.rgb_transform(sample["image"])

        if self.depth_transform:
            sample["depth"] = self.depth_transform(sample["depth"])

        return sample


class CamRecordingServer(Dataset):
    def __init__(self,
                 root_dir,
                 traj_file=None,
                 rgb_transform=None,
                 depth_transform=None,
                 col_ext=".jpg"):

        self.Ts = None
        self.root_dir = root_dir
        self.rgb_transform = rgb_transform
        self.depth_transform = depth_transform
        self.col_ext = col_ext

    def __len__(self):
        all_files = glob.glob(f"{self.root_dir}depth*")
        num_files = len(all_files)
        try:
            return self.Ts.shape[0]
        except:
            return num_files

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        s = f"{idx:06}"  # int variable
        depth_file = self.root_dir + "depth" + s + ".png"
        rgb_file = self.root_dir + "frame" + s + self.col_ext

        # wait until both files exist and ensure write is complete (next file write has started)
        num_files = len(self)
        while not os.path.exists(depth_file) or not os.path.exists(rgb_file) or num_files < (idx + 2):
            continue

        depth = cv2.imread(depth_file, -1)
        image = cv2.imread(rgb_file)

        T = None
        if self.Ts is not None:
            T = self.Ts[idx]

        sample = {"image": image, "depth": depth, "T": T}

        if self.rgb_transform:
            sample["image"] = self.rgb_transform(sample["image"])

        if self.depth_transform:
            sample["depth"] = self.depth_transform(sample["depth"])

        return sample


class TUMDataset(Dataset):
    def __init__(self,
                 root_dir,
                 traj_file=None,
                 rgb_transform=None,
                 depth_transform=None,
                 col_ext=None):

        self.t_poses = None
        if traj_file is not None:
            with open(traj_file) as f:
                lines = (line for line in f if not line.startswith('#'))
                self.t_poses = np.loadtxt(lines, delimiter=' ')

        self.root_dir = root_dir
        self.rgb_transform = rgb_transform
        self.depth_transform = depth_transform

        self.associations_file = root_dir + "associations.txt"
        with open(self.associations_file) as f:
            timestamps, self.rgb_files, self.depth_files = zip(
                *[(float(line.rstrip().split()[0]),
                    line.rstrip().split()[1],
                    line.rstrip().split()[3]) for line in f])

            self.timestamps = np.array(timestamps)

    def __len__(self):
        return len(self.rgb_files)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        depth_file = self.root_dir + self.depth_files[idx]
        rgb_file = self.root_dir + self.rgb_files[idx]

        depth = cv2.imread(depth_file, -1)
        image = cv2.imread(rgb_file)

        T = None
        if self.t_poses is not None:
            rgb_timestamp = self.timestamps[idx]
            timestamp_distance = np.abs(rgb_timestamp - self.t_poses[:, 0])
            gt_idx = timestamp_distance.argmin()
            quat = self.t_poses[gt_idx][4:]
            trans = self.t_poses[gt_idx][1:4]

            T = trimesh.transformations.quaternion_matrix(np.roll(quat, 1))
            T[:3, 3] = trans

        sample = {"image": image, "depth": depth, "T": T}

        if self.rgb_transform:
            sample["image"] = self.rgb_transform(sample["image"])

        if self.depth_transform:
            sample["depth"] = self.depth_transform(sample["depth"])

        return sample


class ReplicaSceneCache(Dataset):
    def __init__(self,
                 traj_file,
                 root_dir,
                 rgb_transform=None,
                 depth_transform=None):

        self.Ts = np.loadtxt(traj_file, delimiter=" ").reshape(-1, 4, 4)
        self.root_dir = root_dir
        self.rgb_transform = rgb_transform
        self.depth_transform = depth_transform
        self.samples = []

        for idx in range(self.Ts.shape[0]):
            s = f"{idx:06}"  # int variable
            depth_file = self.root_dir + "depth" + s + ".png"
            rgb_file = self.root_dir + "frame" + s + ".jpg"

            depth = cv2.imread(depth_file, -1)
            image = cv2.imread(rgb_file)

            if self.rgb_transform:
                image = self.rgb_transform(image)

            if self.depth_transform:
                depth = self.depth_transform(depth)

            self.samples.append((image, depth, self.Ts[idx]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = {"image": self.samples[idx][0],
                  "depth": self.samples[idx][1],
                  "T": self.samples[idx][2]}

        return sample
