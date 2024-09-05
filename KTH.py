import os
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import cv2  # OpenCV for video读取
import numpy as np

class KTHDataset(Dataset):
    def __init__(self, root_dir, transform=None, clip_len=16):
        """
        初始化数据集类。
        Args:
            root_dir (str): KTH数据集的根目录，包含所有视频文件。
            transform (callable, optional): 可选的图像变换函数。
            clip_len (int): 每个样本的视频帧数量（clip length）。
        """
        self.root_dir = root_dir
        self.transform = transform
        self.clip_len = clip_len
        self.classes = ['handclapping', 'handwaving', 'jogging', 'running', 'walking']
        
        # 生成所有视频文件的路径和对应的标签
        self.videos = []
        self.labels = []
        for idx, class_name in enumerate(self.classes):
            class_videos = [os.path.join(root_dir, f) for f in os.listdir(root_dir) if class_name in f]
            self.videos.extend(class_videos)
            self.labels.extend([idx] * len(class_videos))

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        video_path = self.videos[idx]
        label = self.labels[idx]
        # 读取视频并采样固定数量的帧
        frames = self.load_video_frames(video_path)
        # 应用图像变换
        if self.transform:
            frames = self.transform(frames)

        return frames, label

    def load_video_frames(self, video_path):
        """
        读取视频并采样固定数量的帧。

        Args:
            video_path (str): 视频文件路径。
        
        Returns:
            frames (torch.Tensor): 采样的视频帧张量，形状为 [C, T, H, W]。
        """
        cap = cv2.VideoCapture(video_path)
        frames = []
        while len(frames) < self.clip_len:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)

        cap.release()

        # 如果视频帧数少于clip_len，循环补帧
        while len(frames) < self.clip_len:
            frames.append(frames[-1])

        # 将帧转换为张量
        frames = np.array(frames)
        frames = torch.from_numpy(frames).permute(3, 0, 1, 2)  # [C, T, H, W]
        return frames
