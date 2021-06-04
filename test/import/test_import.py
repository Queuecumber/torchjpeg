# The import tests are supposed to test importing only, also these tests don't need docstrings
# pylint: disable=missing-function-docstring,import-outside-toplevel,unused-import


def test_import_torchjpeg():
    import torchjpeg


def test_import_torchjpeg_codec():
    import torchjpeg.codec


def test_import_torchjpeg_dct():
    import torchjpeg.dct


def test_import_torchjpeg_quantization():
    import torchjpeg.quantization


def test_import_torchjpeg_data():
    import torchjpeg.data


def test_import_torchjpeg_transforms():
    import torchjpeg.data.transforms
