import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias
        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        combine = torch.cat([input_tensor, h_cur], dim=1)
        combined = self.conv(combine)
        CR_input, CR_forget, CR_output, CR_Wait = torch.split(combined,self.hidden_dim,dim=1) 

        i = torch.sigmoid(CR_input)
        f = torch.sigmoid(CR_forget)
        o = torch.sigmoid(CR_output)
        w = torch.tanh(CR_Wait)

        c_next = f * c_cur + i * w
        h_next = o * torch.tanh(c_next)

        return h_next, c_next
    
    def init_hidden(self, batch_size, image_size):

        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, 
                            device=self.conv.weight.device),   
                torch.zeros(batch_size, self.hidden_dim, height, width, 
                            device=self.conv.weight.device))
    
class ConvLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers, batch_first=False, bias=True, return_all_layers=False):
        super().__init__()

        self._check_kernel_size_consistency(kernel_size)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]

            cell_list.append(ConvLSTMCell(input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],
                                          bias=self.bias))
        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state=None):

        if not self.batch_first:
            # (time_steps, batch_size, channels, height, width) -> (batch_size, time_steps, channels, height, width)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        if hidden_state is None:
            hidden_state = self._init_hidden(batch_size=input_tensor.size(0),
                                             image_size=(input_tensor.size(3), input_tensor.size(4)))

        layer_output_list = []
        last_state_list = []

        seq_len = input_tensor.size(1)

        cur_layer_input = input_tensor

        # 循环每一层
        for layer_idx in range(self.num_layers):

            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :],  # 当前时间步输入
                                                 cur_state=[h, c])  # 当前层的状态
                output_inner.append(h)

            # 堆叠输出
            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append([h, c])

        if not self.return_all_layers:

            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]

        return layer_output_list, last_state_list

    def _init_hidden(self, batch_size, image_size):

        init_states = []
        for i in range(self.num_layers):

            init_states.append(self.cell_list[i].init_hidden(batch_size, image_size))

        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not isinstance(kernel_size, list) and not isinstance(kernel_size, tuple):
            raise ValueError('kernel_size 必须是列表或元组类型')
        
        if not all(isinstance(elem, tuple) for elem in kernel_size):
            raise ValueError('kernel_size 的每个元素必须是元组类型')