#include <pybind11/embed.h>
#include <torch/extension.h>

#include <algorithm>
#include <array>
#include <numeric>
#include <vector>

#include <jpeglib.h>

class libjpeg_exception : public std::exception {
private:
    char *error;

public:
    libjpeg_exception(j_common_ptr cinfo) {
        error = new char[JMSG_LENGTH_MAX];
        (cinfo->err->format_message)(cinfo, error);
    }

    virtual const char *what() const throw() {
        return error;
    }
};

void raise_libjpeg(j_common_ptr cinfo) {
    throw libjpeg_exception(cinfo);
}

long jdiv_round_up(long a, long b)
/* Compute a/b rounded up to next integer, ie, ceil(a/b) */
/* Assumes a >= 0, b > 0 */
{
    return (a + b - 1L) / b;
}

void extract_channel(const jpeg_decompress_struct &srcinfo,
                     jvirt_barray_ptr *src_coef_arrays,
                     int compNum,
                     torch::Tensor coefficients,
                     torch::Tensor quantization,
                     int &coefficients_written) {
    for (JDIMENSION rowNum = 0; rowNum < srcinfo.comp_info[compNum].height_in_blocks; rowNum++) {
        JBLOCKARRAY rowPtrs = srcinfo.mem->access_virt_barray((j_common_ptr)&srcinfo, src_coef_arrays[compNum],
                                                              rowNum, 1, FALSE);

        for (JDIMENSION blockNum = 0; blockNum < srcinfo.comp_info[compNum].width_in_blocks; blockNum++) {
            std::copy_n(rowPtrs[0][blockNum], DCTSIZE2, coefficients.data_ptr<int16_t>() + coefficients_written);
            coefficients_written += DCTSIZE2;
        }
    }

    std::copy_n(srcinfo.comp_info[compNum].quant_table->quantval, DCTSIZE2,
                quantization.data_ptr<int16_t>() + DCTSIZE2 * compNum);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, std::optional<torch::Tensor>> read_coefficients_using(jpeg_decompress_struct &srcinfo) {
    jpeg_read_header(&srcinfo, TRUE);

    // channels x 2
    auto dimensions = torch::empty({srcinfo.num_components, 2}, torch::kInt);
    auto dct_dim_a = dimensions.accessor<int, 2>();
    for (auto i = 0; i < srcinfo.num_components; i++) {
        dct_dim_a[i][0] = srcinfo.comp_info[i].downsampled_height;
        dct_dim_a[i][1] = srcinfo.comp_info[i].downsampled_width;
    }

    // read coefficients
    jvirt_barray_ptr *src_coef_arrays = jpeg_read_coefficients(&srcinfo);

    auto Y_coefficients = torch::empty({1,
                                        srcinfo.comp_info[0].height_in_blocks,
                                        srcinfo.comp_info[0].width_in_blocks,
                                        DCTSIZE,
                                        DCTSIZE},
                                       torch::kShort);

    auto quantization = torch::empty({srcinfo.num_components, DCTSIZE, DCTSIZE}, torch::kShort);

    // extract Y channel
    int cw = 0;
    extract_channel(srcinfo, src_coef_arrays, 0, Y_coefficients, quantization, cw);

    // extract CrCb channels
    auto CrCb_coefficients = std::optional<torch::Tensor>{};

    if (srcinfo.num_components > 1) {
        CrCb_coefficients = torch::empty({2,
                                          srcinfo.comp_info[1].height_in_blocks,
                                          srcinfo.comp_info[1].width_in_blocks,
                                          DCTSIZE,
                                          DCTSIZE},
                                         torch::kShort);

        cw = 0;
        extract_channel(srcinfo, src_coef_arrays, 1, *CrCb_coefficients, quantization, cw);
        extract_channel(srcinfo, src_coef_arrays, 2, *CrCb_coefficients, quantization, cw);
    }

    // cleanup
    jpeg_finish_decompress(&srcinfo);

    return {
        dimensions,
        quantization,
        Y_coefficients,
        CrCb_coefficients};
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, std::optional<torch::Tensor>> read_coefficients(const std::string &path) {
    // open the file
    FILE *infile;
    if ((infile = fopen(path.c_str(), "rb")) == nullptr) {
        std::ostringstream ss;
        ss << "Unable to open file for reading: " << path;
        throw std::runtime_error(ss.str());
    }

    // start decompression
    jpeg_decompress_struct cinfo{};

    struct jpeg_error_mgr jerr;
    cinfo.err = jpeg_std_error(&jerr);
    jerr.error_exit = raise_libjpeg;

    jpeg_create_decompress(&cinfo);

    jpeg_stdio_src(&cinfo, infile);

    auto ret = read_coefficients_using(cinfo);

    jpeg_destroy_decompress(&cinfo);
    fclose(infile);

    return ret;
}

void set_quantization(j_compress_ptr cinfo, torch::Tensor quantization) {
    int num_components = quantization.size(0);
    std::copy_n(quantization.data_ptr<int16_t>(), DCTSIZE2, cinfo->quant_tbl_ptrs[0]->quantval);

    if (num_components > 1) {
        std::copy_n(quantization.data_ptr<int16_t>() + DCTSIZE2, DCTSIZE2, cinfo->quant_tbl_ptrs[1]->quantval);
    }
}

jvirt_barray_ptr *request_block_storage(j_compress_ptr cinfo) {
    auto block_arrays = (jvirt_barray_ptr *)(*cinfo->mem->alloc_small)((j_common_ptr)cinfo,
                                                                       JPOOL_IMAGE,
                                                                       sizeof(jvirt_barray_ptr *) *
                                                                           cinfo->num_components);

    std::transform(cinfo->comp_info, cinfo->comp_info + cinfo->num_components, block_arrays,
                   [&](jpeg_component_info &compptr) {
                       int MCU_width = jdiv_round_up((long)cinfo->jpeg_width, (long)compptr.MCU_width);
                       int MCU_height = jdiv_round_up((long)cinfo->jpeg_height, (long)compptr.MCU_height);

                       return (cinfo->mem->request_virt_barray)((j_common_ptr)cinfo,
                                                                JPOOL_IMAGE,
                                                                TRUE,
                                                                MCU_width,
                                                                MCU_height,
                                                                compptr.v_samp_factor);
                   });

    return block_arrays;
}

void fill_extended_defaults(j_compress_ptr cinfo, int color_samp_factor_vertical = 2, int color_samp_factor_horizontal = 2) {

    cinfo->jpeg_width = cinfo->image_width;
    cinfo->jpeg_height = cinfo->image_height;

    jpeg_set_defaults(cinfo);

    cinfo->comp_info[0].component_id = 1;
    cinfo->comp_info[0].h_samp_factor = 1;
    cinfo->comp_info[0].v_samp_factor = 1;
    cinfo->comp_info[0].quant_tbl_no = 0;
    cinfo->comp_info[0].width_in_blocks = jdiv_round_up(cinfo->jpeg_width, DCTSIZE);
    cinfo->comp_info[0].height_in_blocks = jdiv_round_up(cinfo->jpeg_height, DCTSIZE);
    cinfo->comp_info[0].MCU_width = 1;
    cinfo->comp_info[0].MCU_height = 1;

    if (cinfo->num_components > 1) {
        cinfo->comp_info[0].h_samp_factor = color_samp_factor_horizontal;
        cinfo->comp_info[0].v_samp_factor = color_samp_factor_vertical;
        cinfo->comp_info[0].MCU_width = color_samp_factor_horizontal;
        cinfo->comp_info[0].MCU_height = color_samp_factor_vertical;

        for (int c = 1; c < cinfo->num_components; c++) {
            cinfo->comp_info[c].component_id = 1 + c;
            cinfo->comp_info[c].h_samp_factor = 1;
            cinfo->comp_info[c].v_samp_factor = 1;
            cinfo->comp_info[c].quant_tbl_no = 1;
            cinfo->comp_info[c].width_in_blocks = jdiv_round_up(cinfo->jpeg_width, DCTSIZE * color_samp_factor_horizontal);
            cinfo->comp_info[c].height_in_blocks = jdiv_round_up(cinfo->jpeg_width, DCTSIZE * color_samp_factor_vertical);
            cinfo->comp_info[c].MCU_width = 1;
            cinfo->comp_info[c].MCU_height = 1;
        }
    }

    cinfo->min_DCT_h_scaled_size = DCTSIZE;
    cinfo->min_DCT_v_scaled_size = DCTSIZE;
}

void set_channel(const jpeg_compress_struct &cinfo,
                 torch::Tensor coefficients,
                 jvirt_barray_ptr *dest_coef_arrays,
                 int compNum,
                 int &coefficients_written) {
    for (JDIMENSION rowNum = 0; rowNum < cinfo.comp_info[compNum].height_in_blocks; rowNum++) {
        JBLOCKARRAY rowPtrs = cinfo.mem->access_virt_barray((j_common_ptr)&cinfo, dest_coef_arrays[compNum],
                                                            rowNum, 1, TRUE);

        for (JDIMENSION blockNum = 0; blockNum < cinfo.comp_info[compNum].width_in_blocks; blockNum++) {
            std::copy_n(coefficients.data_ptr<int16_t>() + coefficients_written, DCTSIZE2, rowPtrs[0][blockNum]);
            coefficients_written += DCTSIZE2;
        }
    }
}

void write_coefficients(const std::string &path,
                        torch::Tensor dimensions,
                        torch::Tensor quantization,
                        torch::Tensor Y_coefficients,
                        std::optional<torch::Tensor> CrCb_coefficients = std::nullopt) {
    FILE *outfile;
    if ((outfile = fopen(path.c_str(), "wb")) == nullptr) {
        std::ostringstream ss;
        ss << "Unable to open file for reading: " << path;
        throw std::runtime_error(ss.str());
    }

    jpeg_compress_struct cinfo{};

    struct jpeg_error_mgr jerr;
    cinfo.err = jpeg_std_error(&jerr);
    jerr.error_exit = raise_libjpeg;

    jpeg_create_compress(&cinfo);
    jpeg_stdio_dest(&cinfo, outfile);

    auto dct_dim_a = dimensions.accessor<int, 2>();

    cinfo.image_height = dct_dim_a[0][0];
    cinfo.image_width = dct_dim_a[0][1];
    cinfo.input_components = CrCb_coefficients ? 3 : 1;
    cinfo.in_color_space = CrCb_coefficients ? JCS_RGB : JCS_GRAYSCALE;

    if (CrCb_coefficients) {
        fill_extended_defaults(&cinfo, (Y_coefficients.size(1) + CrCb_coefficients->size(1) - 1) / CrCb_coefficients->size(1), (Y_coefficients.size(2) + CrCb_coefficients->size(2) - 1) / CrCb_coefficients->size(2));
    } else {
        fill_extended_defaults(&cinfo);
    }

    set_quantization(&cinfo, quantization);

    jvirt_barray_ptr *coef_dest = request_block_storage(&cinfo);
    jpeg_write_coefficients(&cinfo, coef_dest);

    int cw = 0;
    set_channel(cinfo, Y_coefficients, coef_dest, 0, cw);
    
    if (CrCb_coefficients) {
        cw = 0;
        set_channel(cinfo, *CrCb_coefficients, coef_dest, 1, cw);
        set_channel(cinfo, *CrCb_coefficients, coef_dest, 2, cw);
    }

    jpeg_finish_compress(&cinfo);
    jpeg_destroy_compress(&cinfo);

    fclose(outfile);
}

extern "C" {
// On some machines, this free will be name-mangled if it isn't in extern "C" here
void free_buffer(unsigned char *buffer) {
    free(buffer);
}
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, std::optional<torch::Tensor>> quantize_at_quality(torch::Tensor pixels, int quality, int color_samp_factor_vertical = 2, int color_samp_factor_horizontal = 2, bool baseline = true) {
    // Use libjpeg to compress the pixels into a memory buffer, this is slightly wasteful
    // as it performs entropy coding
    struct jpeg_compress_struct cinfo {
    };

    struct jpeg_error_mgr jerr;
    cinfo.err = jpeg_std_error(&jerr);
    jerr.error_exit = raise_libjpeg;

    jpeg_create_compress(&cinfo);

    unsigned long compressed_size;
    unsigned char *buffer = nullptr;
    jpeg_mem_dest(&cinfo, &buffer, &compressed_size);

    cinfo.image_width = pixels.size(2);
    cinfo.image_height = pixels.size(1);
    cinfo.input_components = pixels.size(0);
    cinfo.in_color_space = pixels.size(0) > 1 ? JCS_RGB : JCS_GRAYSCALE;

    jpeg_set_defaults(&cinfo);
    jpeg_set_quality(&cinfo, quality, boolean(baseline));

    cinfo.comp_info[0].h_samp_factor = color_samp_factor_horizontal;
    cinfo.comp_info[0].v_samp_factor = color_samp_factor_vertical;

    // No way that I know of to pass planar images to libjpeg
    auto channel_interleaved = (pixels * 255.f).round().to(torch::kByte).transpose(0, 2).transpose(0, 1).contiguous();

    jpeg_start_compress(&cinfo, TRUE);

    JSAMPROW row_pointer[1];
    while (cinfo.next_scanline < cinfo.image_height) {
        row_pointer[0] = channel_interleaved.data_ptr<JSAMPLE>() +
                         cinfo.next_scanline * channel_interleaved.size(1) * channel_interleaved.size(2);
        jpeg_write_scanlines(&cinfo, row_pointer, 1);
    }

    jpeg_finish_compress(&cinfo);
    jpeg_destroy_compress(&cinfo);

    // Decompress memory buffer to DCT coefficients
    jpeg_decompress_struct srcinfo{};
    struct jpeg_error_mgr srcerr {
    };

    srcinfo.err = jpeg_std_error(&srcerr);
    jpeg_create_decompress(&srcinfo);

    jpeg_mem_src(&srcinfo, buffer, compressed_size);

    auto ret = read_coefficients_using(srcinfo);

    jpeg_destroy_decompress(&srcinfo);
    free_buffer(buffer);

    return ret;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    py::options options;
    options.disable_function_signatures();

    py::register_exception<libjpeg_exception>(m, "LibjpegError");

    m.import("torch");

    m.def("read_coefficients", &read_coefficients, R"(
            read_coefficients(path: str) -> Tuple[Tensor, Tensor, Tensor, Optional[Tensor]]

            Read DCT coefficients from a JPEG file

            Parameters
            ----------
            path : str
                The path to an existing JPEG file

            Returns
            -------
            Tensor
                A :math:`\left(C, 2 \right)` Tensor containing the size of the original image that produced the returned DCT coefficients, this is usually different from the size of the
                coefficient Tensor because padding is added during the compression process. The format is :math:`\left(H, W \right)`.
            Tensor
                A :math:`\left(C, 8, 8 \right)` Tensor containing the quantization matrices for each of the channels. Usually the color channels have the same quantization matrix.
            Tensor
                A :math:`\left(1, \frac{H}{8}, \frac{W}{8}, 8, 8 \right)` Tensor containing the Y channel DCT coefficients for each :math:`8 \times 8` block.
            Optional[Tensor]
                A :math:`\left(2, \frac{H}{8}, \frac{W}{8}, 8, 8 \right)` Tensor containing the Cb and Cr channel DCT coefficients for each :math:`8 \times 8` block, or `None` if the image is grayscale.

            Note
            -----
            The return values from this function are "raw" values, as output by libjpeg with no transformation. In particular, the DCT coefficients are quantized and will need
            to be dequantized using the returned quantization matrices before they can be converted into displayable image pixels. They will likely also need cropping and the chroma
            channels, if they exist, will probably be downsampled. The type of all Tensors is :code:`torch.short` except the dimensions (first return value) with are of type :code:`torch.int`.
          )",
          py::arg("path"));
    m.def("write_coefficients", &write_coefficients, R"(
            write_coefficients(path: str, dimensions: Tensor, quantization: Tensor, Y_coefficients: Tensor, CrCb_coefficients: Optional[Tensor] = None) -> None

            Write DCT coefficients to a JPEG file.

            Parameters
            ----------
            path : str
                The path to the JPEG file to write, will be overwritten
            dimensions : Tensor
                A :math:`\left(C, 2 \right)` Tensor containing the size of the original image before taking the DCT. If you padded the image to produce the coefficients, pass the size before padding here.
            quantization : Tensor
                A :math:`\left(C, 8, 8 \right)` Tensor containing the quantization matrices that were used to quantize the DCT coefficients.
            Y_coefficients : Tensor
                A :math:`\left(1, \frac{H}{8}, \frac{W}{8}, 8, 8 \right)` Tensor of Y channel DCT coefficients separated into :math:`8 \times 8` blocks.
            CbCr_coefficients : Optional[Tensor]
                A :math:`\left(2, \frac{H}{8}, \frac{W}{8}, 8, 8 \right)` Tensor of Cb and Cr channel DCT coefficients separated into :math:`8 \times 8` blocks.

            Note
            -----
            The parameters passed to this function are in the same "raw" format as returned by :py:func:`read_coefficients`. The DCT coefficients must be appropriately quantized and the color 
            channel coefficients must be downsampled if desired. The type of the Tensors must be :code:`torch.short` except the :code:`dimensions` parameter which must be :code:`torch.int`.
          )",
          py::arg("path"), py::arg("dimensions"), py::arg("quantization"), py::arg("Y_coefficients"),
          py::arg("CrCb_coefficients") = std::nullopt);
    m.def("quantize_at_quality", &quantize_at_quality, R"(
            quantize_at_quality(pixels: Tensor, quality: int, color_samp_factor_vertical: int = 2, color_samp_factor_horizontal: int = 2, baseline: bool = true) -> Tuple[Tensor, Tensor, Tensor, Optional[Tensor]]

            Quantize pixels using libjpeg at the given quality. By using this function instead of :py:mod:`torchjpeg.quantization` the result
            is guaranteed to be exactly the same as if the JPEG was quantized using an image library like Pillow and the coefficients are returned
            directly without needing to be recomputed from pixels.

            Parameters
            ----------
            pixels : Tensor
                A :math:`\left(C, H, W \right)` Tensor of image pixels in pytorch format (normalized to [0, 1]).
            quality : int
                The integer quality level to quantize to, in [0, 100] with 100 being maximum quality and 0 being minimal quality.
            color_samp_factor_vertical : int
                Vertical chroma subsampling factor. Defaults to 2.
            color_samp_factor_horizontal : int
                Horizontal chroma subsampling factor. Defaults to 2.
            baseline : bool
                Use the baseline quantization matrices, e.g. quantization matrix entries cannot be larger than 255. True by default, don't change it unless you know what you're doing.

            Returns
            -------
            Tensor
                A :math:`\left(C, 2 \right)` Tensor containing the size of the original image that produced the returned DCT coefficients, this is usually different from the size of the
                coefficient Tensor because padding is added during the compression process. The format is :math:`\left(H, W \right)`.
            Tensor
                A :math:`\left(C, 8, 8 \right)` Tensor containing the quantization matrices for each of the channels. Usually the color channels have the same quantization matrix.
            Tensor
                A :math:`\left(1, \frac{H}{8}, \frac{W}{8}, 8, 8 \right)` Tensor containing the Y channel DCT coefficients for each :math:`8 \times 8` block.
            Optional[Tensor]
                A :math:`\left(2, \frac{H}{8}, \frac{W}{8}, 8, 8 \right)` Tensor containing the Cb and Cr channel DCT coefficients for each :math:`8 \times 8` block, or `None` if the image is grayscale.

            Note
            -----
            The output format of this function is the same as that of :py:func:`read_coefficients`. 
          )",
          py::arg("pixels"), py::arg("quality"), py::arg("color_samp_factor_vertical") = 2, py::arg("color_samp_factor_horizontal") = 2, py::arg("baseline") = true);
}
