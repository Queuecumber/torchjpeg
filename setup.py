from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension

setup(name='torchjpeg',
      ext_modules=[CppExtension('torchjpeg', ['torchjpeg.cpp', 'jdatadst.c'], libraries=[':libjpeg.so.8'])],
      cmdclass={'build_ext': BuildExtension})
