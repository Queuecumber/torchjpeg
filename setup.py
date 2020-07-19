from setuptools import setup, find_packages
from torch.utils.cpp_extension import CppExtension, BuildExtension

setup(name='torchjpeg',
      packages=find_packages(),
      ext_modules=[CppExtension('torchjpeg.codec._codec_ops', ['torchjpeg/codec/codec_ops.cpp', 'torchjpeg/codec/jdatadst.cpp'], libraries=['jpeg'], extra_compile_args=['-std=c++17'])],
      cmdclass={'build_ext': BuildExtension},
      install_requires=[
          'torch',
      ])
