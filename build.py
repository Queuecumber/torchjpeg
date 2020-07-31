# pylint: disable=missing-function-docstring
from subprocess import check_call

from torch.utils.cpp_extension import BuildExtension, CppExtension


def build_libjpeg():
    check_call("cd src/libjpeg && ./configure --enable-static --with-pic && make", shell=True)


def build(setup_kwargs):
    build_libjpeg()

    setup_kwargs.update(
        {"ext_modules": [CppExtension("torchjpeg.codec._codec_ops", ["src/torchjpeg/codec/codec_ops.cpp",], include_dirs=["src/libjpeg"], extra_objects=["src/libjpeg/.libs/libjpeg.a"], extra_compile_args=["-std=c++17"],)], "cmdclass": {"build_ext": BuildExtension},}
    )


if __name__ == "__main__":
    build_libjpeg()
