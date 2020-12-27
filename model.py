import torch.nn as nn


""" Optional conv block """
def conv_block(in_channels, out_channels):

    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
    )



""" Define your own model """
class FewShotModel(nn.Module):
    def __init__(self, x_dim=3, hid_dim=64, z_dim=64):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 96, 12 ,stride = 4)
        self.conv2u = nn.Conv2d(96, 128, 5)
        self.conv2d = nn.Conv2d(96, 128, 5)
        self.conv3u = nn.Conv2d(128, 192, 3)
        self.conv3d = nn.Conv2d(128, 192, 3)
        self.conv4u = nn.Conv2d(192, 384, 3,padding= 1)
        self.conv4d = nn.Conv2d(192, 384, 3,padding= 1)
        self.conv5u = nn.Conv2d(384, 256, 3,padding= 1)
        self.conv5d = nn.Conv2d(384, 256, 3,padding= 1)

        self.bn2 = nn.BatchNorm2d(256)
        self.bn3 = nn.BatchNorm2d(192)

        self.dr = nn.Dropout(p=0.5)
        self.pool = nn.MaxPool2d(2,2)

        self.fc = nn.Linear(11*11*256, 4096)
        #nn.init.xavier_uniform_(self.conv1.weight, gain = 1.0 )
        #nn.init.xavier_uniform_(self.conv2.weight, gain = 1.0 )
        #nn.init.uniform_(self.conv2u.weight, 0.0, 1.0 )
        #nn.init.uniform_(self.conv3u.weight, 0.0, 1.0 )
        #nn.init.uniform_(self.conv4u.weight, 0.0, 1.0 )
        nn.init.xavier_uniform_(self.conv2d.weight, gain = 1.0 )
        #nn.init.xavier_uniform_(self.conv3d.weight, gain = 1.0 )
        #nn.init.xavier_uniform_(self.conv4d.weight, gain = 1.0 )
        #nn.init.xavier_uniform_(self.conv4.weight, gain = 1.0 )
        #nn.init.xavier_uniform_(self.conv5.weight, gain = 1.0 )


    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))                       #in: 400,400,3 /out: 98,98,96
        #######  ######
        xu = self.pool(nn.functional.relu(self.conv2u(x)))  #in: 98,98,96 /out: 92,92,128-->46,46,128
        xd = self.pool(nn.functional.relu(self.conv2d(x)))  #in: 98,98,96 /out: 92,92,128-->46,46,128
        
        xu1 = self.pool(nn.functional.relu(self.conv3u(xu)))  #in: 46,46,128 /out: 44,44,192--> 22,22,192
        xd1 = self.pool(nn.functional.relu(self.conv3d(xd)))  #in: 46,46,128 /out: 44,44,192--> 22,22,192
        
        xu = nn.functional.relu(self.conv4u((xu1+xd1)/2))             #in: 22,22,192 /out: 22,22,384
        xd = nn.functional.relu(self.conv4d((xu1+xd1)/2))             #in: 22,22,192 /out: 22,22,384
        
        xu = self.pool(nn.functional.relu(self.conv5u(xu)))  #in: 22,22,384 /out: 22,22,128-->11,11,256
        xd = self.pool(nn.functional.relu(self.conv5d(xd)))  #in: 22,22,384 /out: 22,22,128-->11,11,256

        #########################
        
        x = (xu + xd)/2
        x = x.view(-1, 11*11*256)

        x = nn.functional.relu(self.fc(x))

        embedding_vector = x
        return embedding_vector