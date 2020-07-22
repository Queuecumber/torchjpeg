from setuptools import setup, find_packages
from torch.utils.cpp_extension import CppExtension, BuildExtension

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(name='torchjpeg',
      version='0.9.4',
      author='Max Ehrlich',
      project_urls={
          'Documentation': 'https://queuecumber.gitlab.io/torchjpeg/',
          'Source': 'https://gitlab.com/Queuecumber/torchjpeg',
          'Tracker': 'https://gitlab.com/Queuecumber/torchjpeg/issues',
      },
      description='Utilities for JPEG data access and manipulation in pytorch',
      long_description=long_description,
      long_description_content_type="text/markdown",
      url='https://queuecumber.gitlab.io/torchjpeg/',
      classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX :: Linux",
        "Development Status :: 3 - Alpha"
      ],
      python_requires='>=3.6',
      packages=find_packages(),
      ext_modules=[CppExtension('torchjpeg.codec._codec_ops', ['torchjpeg/codec/codec_ops.cpp', 'torchjpeg/codec/jdatadst.cpp'], include_dirs=['/usr/local/include'], extra_objects=['/usr/local/lib/libjpeg.a'], extra_compile_args=['-std=c++17'])],
      cmdclass={'build_ext': BuildExtension},
      install_requires=[
          'torch',
          'torchvision',
          'Pillow',
          'numpy'
      ])
