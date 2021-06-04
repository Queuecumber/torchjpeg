# pylint: disable=missing-function-docstring
import pathlib
from subprocess import check_call

from torch.utils.cpp_extension import BuildExtension, CppExtension


def build_libjpeg():
    check_call("cd src/libjpeg && ./configure --enable-static --with-pic && make", shell=True)


def build(setup_kwargs):
    build_libjpeg()

    libjpeg_dir = pathlib.Path(__file__).parent.absolute() / "src" / "libjpeg"

    setup_kwargs.update(
        {
            "ext_modules": [
                CppExtension(
                    "torchjpeg.codec._codec_ops",
                    [
                        "src/torchjpeg/codec/codec_ops.cpp",
                    ],
                    include_dirs=[str(libjpeg_dir)],
                    extra_objects=[str(libjpeg_dir / ".libs" / "libjpeg.a")],
                    extra_compile_args=["-std=c++17"],
                )
            ],
            "cmdclass": {"build_ext": BuildExtension},
        }
    )


if __name__ == "__main__":
    build_libjpeg()
