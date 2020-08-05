# Quantizing DCT Coefficients

The {py:mod}`torchjpeg.quantization` package provides routines for quantizing DCT coefficients, including performing IJG (libjpeg) compatible quantization by implementing their quality to quantization matrix algorithm. The 
functions are entirely written in pytorch and can be differentiated adn GPU accelerated. This is in contrast to the {py:func}`torchjpeg.codec.quantize_at_quality` function which uses the libjpeg C API. Both have advantages and
disadvantages. Here we discuss only {py:mod}`torchjpeg.quantization`.

## General Quantization

The two functions {py:func}`torchjpeg.quantization.quantize_multichannel` and {py:func}`torchjpeg.quantization.dequantize_multichannel` implement the general case quantization. These funcions take a batch of DCT coefficients (shape $N \times 3 \times H \times W$) and quantization matrices of shape $1 \times 2 \times 8 \times 8$ and returns quantized coefficients for each image in the batch. This function also applies chroma subsampling. If chroma subsampling is not desired, then the single channel versions {py:func}`torchjpeg.quantization.quantize` and {py:func}`torchjpeg.quantization.dequantize` can be used instead. 

```{note}
While the quantization function is fully differentiable, because of the rounding function, the gradient will not be useful. The rounding function can be overridden using the `round_func` parameter to an approximation to true rounding which provides more useful gradient. It defaults to {py:func}`torch.round`.
```

## IJG Quantization

For differentiable libjpeg compatible quantization, use the {py:mod}`torchjpeg.quantization.ijg` package. The analog to the previous sections discussion are the functions {py:func}`torchjpeg.quantization.ijg.quantize_at_quality` and{py:func}`torchjpeg.quantization.ijg.dequantize_at_quality`. These functions take a batch of DCT coefficients (of a single channel) and a scalar quality in [0, 100] and quantize and dequantize the coefficients respectively. The parameter `table` allows the selection of either the 'luma' table for y channels or the 'chroma' table for color channels.