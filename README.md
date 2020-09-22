# TorchJPEG

[![pipeline status](https://gitlab.com/Queuecumber/torchjpeg/badges/master/pipeline.svg)](https://gitlab.com/Queuecumber/torchjpeg/-/pipelines/latest)
[![coverage report](https://gitlab.com/Queuecumber/torchjpeg/badges/master/coverage.svg)](https://gitlab.com/Queuecumber/torchjpeg/-/pipelines/latest)
[![PyPI](https://img.shields.io/pypi/v/torchjpeg)](https://pypi.org/project/torchjpeg/)
[![License](https://img.shields.io/badge/license-MIT-blue)](https://gitlab.com/Queuecumber/torchjpeg/-/blob/master/LICENSE)

This package contains a C++ extension for pytorch that interfaces with libjpeg to allow for manipulation of low-level JPEG data.
By using libjpeg, quantization results are guaranteed to be consistent with other applications, like image viewers or MATLAB,
which use libjpeg to compress and decompress images. This is useful because JPEG images can be effected by round-off
errors or slight differences in the decompression procedure. Besides this, this library can be used to read and write
DCT coefficients, functionality which is not available from other python interfaces.

Besides this, the library includes many utilities related to JPEG compression, many of which are written using native pytorch code meaning
they can be differentiated or GPU accelerated. The library currently includes packages related to the DCT, quantization, metrics, and dataset
transformations.

## LIBJPEG

Currently builds against: `libjpeg-9d`. libjpeg is statically linked during the build process. See [http://www.ijg.org/files/](http://www.ijg.org/files/) for libjpeg source. 
The full libjpeg source is included with the torchjpeg source code and will be built during the install process (for a source or sdist install).

## Install

Packages are hosted on [pypi](https://pypi.org/project/torchjpeg/). Install using pip, note that only Linux builds are supported at the moment. 

```
pip install torchjpeg
```

If there is demand for builds on other platforms it may happen in the future. Also note that the wheel is intended to be compatible with manylinux2014
which means it should work on modern Linux systems, however, because of they way pytorch works, we can't actually build it using all of the manylinux2014
tools. So compliance is not guaranteed and YMMV.

```{warning}
torchjpeg is currently in pre-beta development and consists mostly of converted research code. The public facing API, including any and all names of
parameters and functions, is subject to change at any time. We follow semver for versioning and will adhere to that before making and breaking
changes.
```

## Citation

If you use our code in a publication, we ask that you cite the following paper ([bibtex](http://maxehr.umiacs.io/bibtex/ehrlich2020quantization.txt)):

> Max Ehrlich, Larry Davis, Ser-Nam Lim, and Abhinav Shrivastava. "Quantization Guided JPEG Artifact Correction." In Proceedings of the European Conference on Computer Vision, 2020