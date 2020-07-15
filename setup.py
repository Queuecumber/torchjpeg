from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension

setup(name='torchjpeg.codec',
      ext_modules=[CppExtension('torchjpeg', ['codec/torchjpeg.cpp', 'codec/jdatadst.cpp'], libraries=['jpeg'])],
      cmdclass={'build_ext': BuildExtension},
      install_requires=[
          'torch',
      ])
