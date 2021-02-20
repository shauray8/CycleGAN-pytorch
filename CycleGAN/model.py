import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Sequential(
                    nn.ReflectionPad2d(1),
                    nn.Conv2d(in_features, in_features, 3),
                    nn.InstanceNorm2d(in_features),
                    nn.ReLU(inplace=True),
                    nn.ReflectionPad2d(1),
                    nn.Conv2d(in_features, in_features, 3),
                    nn.InstanceNorm2d(in_features),
                )

        def forward(self, x):
            x = self.conv1(x)
            return x

class Generator(nn.Module):
    def __init__(self, input_nc, output_nc, n_residual_blocks=9):
        super(Generator, self).__init__()
        model = [   nn.ReflectionPad2d(3),
                    nn.Conv2d(input_nc, 64, 7),
                    nn.InstanceNorm2d(64),
                    nn.ReLU(inplace=True)   ]
        
        in_features = 64
        out_features = in_features*2

        #downsampling

        for _ in range(2):
            model += [nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                    nn.InstanceNorm2d(out_features),
                    nn.ReLU(inplace=True)]

            in_features = out_features
            out_features = in_features*2

        #residual block

        for _ in range(n_residual_blocks):
            model += [ResidualBlock(in_features)]

        #upsampling

        out_features = in_features // 2
        for _ in range(2):
            model += [  nn.ReflectionPad2d(3),
                    nn.Conv2d(64, output_nc, 7),
                    nn.Tanh()   ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class Discriminator(nn.Module):
    def __init__(self, input_nc):
        super(Discriminator, self).__init__()

        #convolutions 

        model = [nn.Conv2d(input_nc, 64, 4, stride=2, padding=1),
                nn.LeakyReLU(.2, inplace=True),]

        model += [nn.Conv2d(64, 128, 4, stride=2, padding=1),
                nn.InstanceNorm2d(128),
                nn.LeakyReLU(.2, inplace=True)]

        model += [nn.Conv2d(128, 256, 4, stride=2, padding=1),
                nn.InstanceNorm2d(256),
                nn.LeakyReLU(.2, inplace=True)]

        model += [nn.Conv2d(256, 512, 4, padding=1),
                nn.InstanceNorm2d(512),
                nn.LeakyReLU(.2, inplace=True)]

        #FCN clasification layer
        model += [nn.Conv2d(512, 1, 4, padding=1)]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        x = self.model(x)

        return F.avg_pool(x, x.size()[2:]).view(x.size()[0], -1)


if __name__ == "__main__":
    g = Generator(28*28, 12)
    d = Discriminator(12)
    
    print(g)
    print(d)
    

