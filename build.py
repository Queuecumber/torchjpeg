# pylint: disable=missing-function-docstring
from torch.utils.cpp_extension import BuildExtension, CppExtension


def build(setup_kwargs):
    setup_kwargs.update(
        {"ext_modules": [CppExtension("torchjpeg.codec._codec_ops", ["src/torchjpeg/codec/codec_ops.cpp",], include_dirs=["/usr/local/include"], extra_objects=["/usr/local/lib/libjpeg.a"], extra_compile_args=["-std=c++17"],)], "cmdclass": {"build_ext": BuildExtension},}
    )
