# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import warnings

import torchvision
warnings.filterwarnings('ignore', category=DeprecationWarning)

import os
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MUJOCO_GL'] = 'egl'

from pathlib import Path

import hydra
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import IterableDataset
import pandas as pd
import json
import time
import pickle
from torchvision.utils import save_image
import json
import random
import h5py

def get_ind(vid, index, ds):
    if ds == "ego4d":
        return torchvision.io.read_image(f"{vid}/{index:06}.jpg")
    elif ds == 'droid':
        return torchvision.io.read_image(f"{vid}/{index}.png", mode=torchvision.io.image.ImageReadMode.RGB)
    elif ds == 'libero':
        return 0
    else:
        raise NameError('Invalid Dataset')


## Data Loader for Ego4D
class MCRBuffer(IterableDataset):
    def __init__(self, ego4dpath, num_workers, source1, source2, alpha, datasources, doaug = "none"):
        self._num_workers = max(1, num_workers)
        self.alpha = alpha
        self.curr_same = 0
        self.data_sources = datasources
        self.doaug = doaug

        # Augmentations
        if doaug in ["rc", "rctraj"]:
            self.aug = torch.nn.Sequential(
                transforms.RandomResizedCrop(224, scale = (0.2, 1.0)),
            )
        else:
            self.aug = lambda a : a

        # Load Data
        if "ego4d" in self.data_sources:
            print("Ego4D")
            self.manifest = pd.read_csv(f"{ego4dpath}manifest.csv")
            print(self.manifest)
            self.ego4dlen = len(self.manifest)
        else:
            raise NameError('Invalid Dataset')


    def _sample(self):
        t0 = time.time()
        ds = random.choice(self.data_sources)

        vidid = np.random.randint(0, self.ego4dlen)
        m = self.manifest.iloc[vidid]
        vidlen = m["len"]
        txt = m["txt"]
        label = txt[2:] ## Cuts of the "C " part of the text
        vid = m["path"]

        start_ind = np.random.randint(1, 2 + int(self.alpha * vidlen))
        end_ind = np.random.randint(int((1-self.alpha) * vidlen)-1, vidlen)
        s1_ind = np.random.randint(2, vidlen)
        s0_ind = np.random.randint(1, s1_ind)
        s2_ind = np.random.randint(s1_ind, vidlen+1) # start, s0, s1, s2, end

        if self.doaug == "rctraj":
            ### Encode each image in the video at once the same way
            im0 = get_ind(vid, start_ind, ds) 
            img = get_ind(vid, end_ind, ds)
            imts0 = get_ind(vid, s0_ind, ds)
            imts1 = get_ind(vid, s1_ind, ds)
            imts2 = get_ind(vid, s2_ind, ds)
            allims = torch.stack([im0, img, imts0, imts1, imts2], 0)
            allims_aug = self.aug(allims / 255.0) * 255.0

            im0 = allims_aug[0]
            img = allims_aug[1]
            imts0 = allims_aug[2]
            imts1 = allims_aug[3]
            imts2 = allims_aug[4]
        else:
            ### Encode each image individually
            im0 = self.aug(get_ind(vid, start_ind, ds) / 255.0) * 255.0
            img = self.aug(get_ind(vid, end_ind, ds) / 255.0) * 255.0
            imts0 = self.aug(get_ind(vid, s0_ind, ds) / 255.0) * 255.0
            imts1 = self.aug(get_ind(vid, s1_ind, ds) / 255.0) * 255.0
            imts2 = self.aug(get_ind(vid, s2_ind, ds) / 255.0) * 255.0

        im = torch.stack([im0, img, imts0, imts1, imts2])
        return (im, label)

    def __iter__(self):
        while True:
            yield self._sample()

## Data Loader for LIBERO
class MCRBufferLibero(IterableDataset):
    def __init__(
            self,
            liberopath,
            num_workers,
            alpha,
            datasources,
            doaug = "none",
            state_list_used = ['obs/ee_states', 'obs/gripper_states', 'obs/joint_states'],
            state_window = 1,
            use_action = False,
            view_keys_used = ['obs/agentview_rgb', 'obs/eye_in_hand_rgb'],
            tasks = ['libero_10', 'libero_90', 'goal', 'object', 'spatial'],
        ):
        
        self._num_workers = max(1, num_workers)
        self.alpha = alpha
        self.data_sources = datasources
        self.doaug = doaug
        self.dataset_path = liberopath
        self.state_list_used = state_list_used
        self.state_window = state_window
        self.use_action = use_action
        self.action_key = 'actions'
        self.view_keys = view_keys_used
        self.datasetlen = 0

        # Augmentations
        if doaug in ["rc", "rctraj"]:
            self.aug = torch.nn.Sequential(
                transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),
            )
        elif doaug in ["rctraj_eval"]:
            self.aug = torch.nn.Sequential(
            transforms.Resize(256),
            transforms.CenterCrop(224),
            )
        else:
            self.aug = lambda a : a

        # Load Data
        if "libero" in self.data_sources:
            print("Libero")
            self.loaded_dataset = []

            for task in tasks:
                task_dir = os.path.join(self.dataset_path, task)
                if not os.path.isdir(task_dir):
                    continue
                for fname in os.listdir(task_dir):
                    if not fname.endswith('.hdf5'):
                        continue
                    lang_inst = os.path.splitext(fname)[0]
                    path = os.path.join(task_dir, fname)
                    data = h5py.File(path, 'r')['data']

                    for demo in data:
                        if 
                        traj = {}
                        for key in state_list_used or view_keys_used:
                            traj[key] = data[demo][key]
                        
                        traj['actions'] = data[demo]['actions']
                        traj['lang'] = lang_inst

                        self.loaded_dataset.append(traj)
                        self.datasetlen += 1
        else:
            raise NameError('Invalid Dataset')
                

    def _sample(self):
        t0 = time.time()
        ds = random.choice(self.data_sources)

        vidid = np.random.randint(0, self.datasetlen)
        traj_path = self.loaded_dataset[vidid]


import os
import random
import h5py
import numpy as np
import torch
from torch.utils.data import IterableDataset
from torchvision import transforms

import os
import random
import h5py
import numpy as np
import torch
from torch.utils.data import IterableDataset
from torchvision import transforms

class MCRBufferLibero(IterableDataset):
    def __init__(
        self,
        liberopath: str,
        num_workers: int,
        alpha: float,
        datasources: list,
        doaug: str = 'none',
        state_list_used: list = ['states'],
        state_window: int = 1,
        use_action: bool = False,
        view_keys_used: list = ['obs/agentview_rgb', 'obs/eye_in_hand_rgb'],
        tasks: list = ['libero_10', 'libero_90', 'goal', 'object', 'spatial'],
    ):
        super().__init__()
        self._num_workers = max(1, num_workers)
        self.alpha = alpha
        self.data_sources = datasources
        self.doaug = doaug
        self.dataset_path = liberopath
        self.state_list_used = state_list_used
        self.state_window = state_window
        self.use_action = use_action
        self.action_key = 'actions'
        self.view_keys = view_keys_used

        # 이미지 증강 설정
        if doaug in ['rc', 'rctraj']:
            self.aug = transforms.RandomResizedCrop(224, scale=(0.5, 1.0))
        elif doaug == 'rctraj_eval':
            self.aug = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
            ])
        else:
            self.aug = lambda x: x

        # Libero task별 하위 디렉터리에서 HDF5 파일 로드
        if 'libero' in self.data_sources:
            self.files = {}       # key: (task, lang_inst) -> h5py.File
            self.episodes = []    # list of (task, lang_inst, ep_group)

            for task in tasks:
                task_dir = os.path.join(self.dataset_path, task)
                if not os.path.isdir(task_dir):
                    continue
                # 각 폴더 내 .hdf5 파일
                for fname in os.listdir(task_dir):
                    if not fname.endswith('.hdf5'):
                        continue
                    lang_inst = os.path.splitext(fname)[0]
                    path = os.path.join(task_dir, fname)
                    hf = h5py.File(path, 'r')
                    self.files[(task, lang_inst)] = hf
                    # 'data/demo_*' 그룹 순회
                    for demo in hf['data']:
                        ep_group = f"data/{demo}"
                        self.episodes.append((task, lang_inst, ep_group))
        else:
            raise NameError('Invalid Dataset')

    def _sample(self):
        # 무작위 에피소드 선택
        task, lang_inst, ep_group = random.choice(self.episodes)
        hf = self.files[(task, lang_inst)]
        grp = hf[ep_group]
        vidlen = grp[self.action_key].shape[0]

        # 시간 인덱스 샘플링
        start = random.randint(1, max(1, int(2 + self.alpha * vidlen) - 1))
        end   = random.randint(max(1, int((1 - self.alpha) * vidlen) - 1), vidlen - 1)
        s1    = random.randint(2, vidlen - 1)
        s0    = random.randint(1, s1 - 1)
        s2    = random.randint(s1, vidlen - 1)

        # 이미지 로드 및 증강
        ims = []
        for idx in [start, end, s0, s1, s2]:
            view = random.choice(self.view_keys)
            img_np = grp[view][idx]                  # HxWxC, uint8
            img = torch.from_numpy(img_np).permute(2, 0, 1).float()
            img = (self.aug(img / 255.0) * 255.0)
            ims.append(img)
        images = torch.stack(ims, dim=0)             # (5, C, H, W)

        # s0 시점 상태 벡터
        parts = [grp[k][s0] for k in self.state_list_used]
        state_array = torch.tensor(np.concatenate(parts)).float()

        # 슬라이딩 윈도우 상태 및 액션
        full_s0, full_s2 = [], []
        s0_start = max(0, s0 - self.state_window // 2)
        s2_start = max(0, s2 - self.state_window // 2)
        for i in range(self.state_window):
            i0 = min(s0_start + i, vidlen - 1)
            i2 = min(s2_start + i, vidlen - 1)
            for k in self.state_list_used:
                full_s0.append(grp[k][i0])
                full_s2.append(grp[k][i2])
            if self.use_action and i < self.state_window - 1:
                full_s0.append(grp[self.action_key][i0])
                full_s2.append(grp[self.action_key][i2])
        full_state = {
            's0': torch.tensor(np.concatenate(full_s0)).float(),
            's2': torch.tensor(np.concatenate(full_s2)).float(),
        }

        # 액션 시퀀스
        acts = [grp[self.action_key][i] for i in [start, end, s0, s1, s2]]
        actions = torch.tensor(np.stack(acts)).float()  # (5, action_dim)

        # label: (task, language instruction)
        label = f"{task}/{lang_inst}"

        return images, label, state_array, full_state, actions

    def __iter__(self):
        while True:
            yield self._sample()









## Data Loader for Droid
class MCRBufferDroid(IterableDataset):
    def __init__(self, droidpath, num_workers, source1, source2, alpha,
                    datasources, doaug = "none", state_list_used = None, state_window=1, use_action=False, view_keys_used = None):
        self._num_workers = max(1, num_workers)
        self.alpha = alpha
        self.curr_same = 0
        self.data_sources = datasources
        self.doaug = doaug
        self.dataset_path = droidpath
        self.state_keys = ['cartesian_position', 'gripper_position', 'joint_position']
        self.lang_keys = ['language_instruction', 'language_instruction_2', 'language_instruction_3']
        self.view_keys = view_keys_used # ['exterior_image_1_left', 'exterior_image_2_left', 'wrist_image_left']
        self.state_list_used = state_list_used
        self.state_window = state_window
        self.use_action = use_action

        # Augmentations
        if doaug in ["rc", "rctraj"]:
            self.aug = torch.nn.Sequential(
                transforms.RandomResizedCrop(224, scale = (0.5, 1.0)), # first crop, then resize
            )
        elif doaug in ["rctraj_eval"]:
            self.aug = torch.nn.Sequential(
                transforms.Resize(256),
                transforms.CenterCrop(224),
            )
        else:
            self.aug = lambda a : a

        # Load Data
        if "droid" in self.data_sources:
            print("Droid")
            self.loaded_dataset = os.listdir(droidpath)
            print(self.loaded_dataset[:5])
            self.datasetlen = len(self.loaded_dataset)
        else:
            raise NameError('Invalid Dataset')


    def _sample(self):
        t0 = time.time()
        ds = random.choice(self.data_sources)

        vidid = np.random.randint(0, self.datasetlen)
        traj_path = self.loaded_dataset[vidid] # 2023-02-28_Tue_Feb_28_20:31:42_2023
        vidlen = min(len(os.listdir(os.path.join(self.dataset_path, traj_path, 'exterior_image_1_left'))), len(os.listdir(os.path.join(self.dataset_path, traj_path, 'exterior_image_2_left'))))
        txt_path = random.choice(self.lang_keys)
        with open(os.path.join(self.dataset_path, traj_path, txt_path, '0.txt'), 'r') as file:
            label = file.read()
        vid_path = random.choice(self.view_keys) # time contrastive within same view
        vid = os.path.join(self.dataset_path, traj_path, vid_path) # video path
        otherdata_path = os.path.join(self.dataset_path, traj_path, 'other_data.pkl')

        start_ind = np.random.randint(1, 2 + int(self.alpha * vidlen)) # [low, high)
        end_ind = np.random.randint(int((1-self.alpha) * vidlen)-1, vidlen)
        s1_ind = np.random.randint(2, vidlen)
        s0_ind = np.random.randint(1, s1_ind)
        s2_ind = np.random.randint(s1_ind, vidlen) # start, s0, s1, s2, end

        # for state encode
        with open(otherdata_path, 'rb') as f:
            loaded_data = pickle.load(f)
        state_array, full_state_dict = np.empty(0), {'s0': np.empty(0), 's2': np.empty(0)}
        for key in self.state_list_used:
            state_array = np.concatenate((state_array, loaded_data[key][s0_ind]))

        s0wind_start = max(1, s0_ind - self.state_window // 2)
        s2wind_start = max(1, s2_ind - self.state_window // 2)
        for i in range(self.state_window):
            for key in self.state_keys:
                full_state_dict['s0'] = np.concatenate((full_state_dict['s0'], loaded_data[key][min(s0wind_start + i, vidlen - 1)]))
                full_state_dict['s2'] = np.concatenate((full_state_dict['s2'], loaded_data[key][min(s2wind_start + i, vidlen - 1)]))
            if self.use_action and i != self.state_window - 1:
                full_state_dict['s0'] = np.concatenate((full_state_dict['s0'], loaded_data['action'][min(s0wind_start + i, vidlen - 1)]))
                full_state_dict['s2'] = np.concatenate((full_state_dict['s2'], loaded_data['action'][min(s2wind_start + i, vidlen - 1)]))

        full_state_dict['s0'] = torch.tensor(full_state_dict['s0']).float()
        full_state_dict['s2'] = torch.tensor(full_state_dict['s2']).float()

        # for bc, sample action
        actions = torch.tensor(np.stack([loaded_data['action'][start_ind], 
                                loaded_data['action'][end_ind], 
                                loaded_data['action'][s0_ind], 
                                loaded_data['action'][s1_ind], 
                                loaded_data['action'][s2_ind]])).float()
        # actions = torch.tensor(loaded_data['action'][s0_ind]).float()

        if self.doaug == ["rctraj", "rctraj_eval"]:
            ### Encode each image in the video at once the same way
            im0 = get_ind(vid, start_ind, ds)
            img = get_ind(vid, end_ind, ds)
            imts0 = get_ind(vid, s0_ind, ds)
            imts1 = get_ind(vid, s1_ind, ds)
            imts2 = get_ind(vid, s2_ind, ds)
            allims = torch.stack([im0, img, imts0, imts1, imts2], 0)
            allims_aug = self.aug(allims / 255.0) * 255.0

            im0 = allims_aug[0]
            img = allims_aug[1]
            imts0 = allims_aug[2]
            imts1 = allims_aug[3]
            imts2 = allims_aug[4]
        else:
            ### Encode each image individually
            im0 = self.aug(get_ind(vid, start_ind, ds) / 255.0) * 255.0
            img = self.aug(get_ind(vid, end_ind, ds) / 255.0) * 255.0
            imts0 = self.aug(get_ind(vid, s0_ind, ds) / 255.0) * 255.0
            imts1 = self.aug(get_ind(vid, s1_ind, ds) / 255.0) * 255.0
            imts2 = self.aug(get_ind(vid, s2_ind, ds) / 255.0) * 255.0

        im = torch.stack([im0, img, imts0, imts1, imts2])
        return (im, label, torch.tensor(state_array).float(), full_state_dict, actions)

    def __iter__(self):
        while True:
            yield self._sample()