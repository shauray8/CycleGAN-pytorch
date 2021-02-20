from model import Generator, Discriminator
import torch

def test():
    N, input_nc, output_nc = 8,3,3
    z_dim =100
    disc = Discriminator(output_nc)
    gen = Generator(input_nc, output_nc)
    print("Working")


test()
