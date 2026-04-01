import torch
import torch.nn as nn
import torch.nn.functional as F

class DownstreamModel(nn.Module):
    def __init__(self, class_num):
        super(DownstreamModel, self).__init__()
        self.fc1 = nn.Linear(4096, 1024)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(1024, 256)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(256, class_num)

    def forward(self, input_l):
        """
        input_l: (batch_size, 4096)
        """
        if input_l.shape[1] > 4096:
            input_l = input_l[:, :4096]  # 截断
        output = self.fc1(input_l)
        output = self.relu1(output)
        output = self.dropout1(output)
        output = self.fc2(output)
        output = self.relu2(output)
        output = self.dropout2(output)
        output = self.fc3(output)
        return output

if __name__ == "__main__":
    # 测试数据
    raw_input = torch.randn(32, 5, 4096)  # batch_size=32
    a = raw_input.mean(dim=1) 

    model = DownstreamModel(class_num=2)
    output = model(a)
    print(output.shape)  # (32, 2)


