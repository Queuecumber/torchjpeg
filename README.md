# TorchJPEG

This C++ extension for pytorch interfaces with libjpeg to allow for manipulation of low-level JPEG data using pytorch.
By using libjpeg, quantization results are guaranteed to be consistent with other applications, like image viewers or MATLAB,
which use libjpeg to compress and decompress images. This is useful because JPEG images can be effected by round-off
errors or slight differences in the decompression procedure. Besides this, this library can be used to read and write
DCT coefficients, functionality which is not available from other python interfaces.

To quantize spatial domain pixels, the data can be provided in the pytorch native format e.g. CHW float tensors with
values in [0, 1]. For reading data, however, the results are largely untampered with and it is up to the user to
manipulate them as needed. Quantization matrices are represented as C x 8 x 8 short (int16) tensors and DCT coefficients are
given as C x H//8 x W//8 x 8 x 8 short tensors (e.g. blocks) of quantized coefficients, the same format used internally
by libjpeg. The DCT coefficients for the Y channel and CrCb channels are returned separately and will usually be
different sizes. It is up to the user of the library to dequantize, reshape, and otherwise convert them into pixels
if desired. See `test_decompress.py` for an example of how to do this.

## LIBJPEG

Currently builds against libjpeg-9d

## Exported Functions

`read_coefficients(path)` - Reads DCT coefficients from the JPEG file at the given path. Return values are

* `dimensions` - (C x 2 int) the downsampled height x width of the channels. This will often be different from the width/height of the
returned DCT coefficients if padding was added to the blocks
* `quantization` - (C x 8 x 8 short) The quantization matrix that was used for each channel
* `Y_coefficients` - (1 x H//8 x W//8 x 8 x 8 short) The Y channel coefficients organized row-major blockwise
* `CrCb_coefficients` - (2 x H//8 x W//8 x 8 x 8 short) The CrCb channel coefficients organized row-major blockwise if the image is color, otherwise empty

`write_coefficients (path, dimensions, quantization, Y_coefficients, CrCb_coefficients` - Writes the given DCT coefficients to a JPEG file, arguments must be
in the same format as the return value from `read_coefficients`

`quantize_at_quality(pixels, quality, baseline)` - Computes the quantized DCT coefficients for the given input pixels at the given quality.
*Baseline is true be default* and limits the quantization matrix entries to the [0, 255] range as mandated by the JPEG standard (see libjpeg documentation
for more details). Quality is an integer from [0, 100] where 0 is the worst quality and 100 is the best quality (note that 100 is *not* lossless). Pixels
is a tensor in the standard pytorch image format, e.g. CHW float in [0, 1]. Return values are the same as `read_coefficients`