import torchvision.transforms as transforms
import KTH
from torch.utils.data import DataLoader

# 定义数据变换
transform = transforms.Compose([
    transforms.Resize((128, 171)),  # 调整视频帧大小
    transforms.CenterCrop(128),     # 裁剪中心区域
    transforms.ToTensor(),          # 转换为PyTorch张量
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 标准化
])

# 创建数据集实例
dataset = KTH.KTHDataset(root_dir='~/ConvLSTM/dataset',  # 数据集的根目录
                     transform=transform, 
                     clip_len=16)

# 创建 DataLoader
dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)

# 迭代数据
for inputs, labels in dataloader:
    # 在这里进行模型训练或测试
    print(inputs.size(), labels)
    break
