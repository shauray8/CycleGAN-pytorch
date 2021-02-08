import sys
from setuptools import setup, find_packages

sys.path[0:0] = ['CycleGAN-pytorch']
#from version import __version__

setup(
  name = 'CycelGAN-pytorch',
  packages = find_packages(),
  #version = __version__,
  license='MIT',
  description = 'CycleGan in Pytorch',
  author = 'Shauray Singh',
  author_email = 'shauray9@gmail.com',
  url = 'https://github.com/shauray8/CycleGAN-pytorch',
  keywords = ['generative adversarial networks', 'machine learning'],
  install_requires=[
      'numpy',
      'tqdm',
      'torch',
      'torchvision',
      'pillow',
  ],
)
