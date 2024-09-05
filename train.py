import ConvLSTM
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch
import torchvision.transforms as transforms

# traintransform = transforms.Compose([transforms.

input_dims = 3
output_dims = 3
num_layers = 2
hidden_dims = 64
kernel_size = [(3,3),(3,3)]

input_steps = 7
output_steps = 1


# trainLoader = 

model = ConvLSTM.ConvLSTM(input_dim=input_dims,hidden_dim=hidden_dims,
                          output_dims = output_dims,kernel_size=kernel_size,num_layers=num_layers,batch_first=True,input_steps=input_steps,output_steps=output_steps)

if torch.cuda.is_available():
    model = model.cuda()

criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001,weight_decay= 0.0001)

def train():
    
        if torch.cuda.is_available():
            data = Variable(data).cuda()
            target = Variable(target).cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output,target)
        loss.backward()
        optimizer.step()

