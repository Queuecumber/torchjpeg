from setuptools import setup, find_packages
from torch.utils.cpp_extension import CppExtension, BuildExtension

setup(name='torchjpeg',
      packages=find_packages(),
      ext_modules=[CppExtension('torchjpeg.codec.codec_ops', ['torchjpeg/codec/codec_ops.cpp', 'torchjpeg/codec/jdatadst.cpp'], libraries=['jpeg'])],
      cmdclass={'build_ext': BuildExtension},
      install_requires=[
          'torch',
      ])
