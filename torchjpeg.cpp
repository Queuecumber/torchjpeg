#include <torch/extension.h>
#include <jpeglib.h>
#include <numeric>
#include <algorithm>
#include "jdatadst.h"

int extract_channel(const jpeg_decompress_struct &srcinfo, jvirt_barray_ptr *src_coef_arrays, int compNum, torch::Tensor coefficients, torch::Tensor quantization, int coefficients_written) {
    for (JDIMENSION rowNum = 0; rowNum < srcinfo.comp_info[compNum].height_in_blocks; rowNum++) {
        JBLOCKARRAY rowPtrs = srcinfo.mem->access_virt_barray((j_common_ptr) &srcinfo, src_coef_arrays[compNum],
                                                              rowNum, 1, FALSE);

        for (JDIMENSION blockNum = 0; blockNum < srcinfo.comp_info[compNum].width_in_blocks; blockNum++) {
            std::copy_n(rowPtrs[0][blockNum], DCTSIZE2, coefficients.data<int16_t>() + coefficients_written);
            coefficients_written += DCTSIZE2;
        }
    }

    std::copy_n(srcinfo.comp_info[compNum].quant_table->quantval, DCTSIZE2, quantization.data<int16_t>() + DCTSIZE2 * compNum);

    return coefficients_written;
}

std::vector<torch::Tensor> read_coefficients_using(jpeg_decompress_struct &srcinfo) {
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
                                        DCTSIZE
                                       }, torch::kShort);

    auto quantization = torch::empty({srcinfo.num_components, DCTSIZE, DCTSIZE}, torch::kShort);

    // extract Y channel
    extract_channel(srcinfo, src_coef_arrays, 0, Y_coefficients, quantization, 0);

    // extract CrCb channels
    auto CrCb_coefficients = torch::empty({}, torch::kShort);

    if (srcinfo.num_components > 1) {
        CrCb_coefficients = torch::empty({2,
                                          srcinfo.comp_info[1].height_in_blocks,
                                          srcinfo.comp_info[1].width_in_blocks,
                                          DCTSIZE,
                                          DCTSIZE
                                         }, torch::kShort);

        auto start = extract_channel(srcinfo, src_coef_arrays, 1, CrCb_coefficients, quantization, 0);
        extract_channel(srcinfo, src_coef_arrays, 2, CrCb_coefficients, quantization, start);
    }


    // cleanup
    jpeg_finish_decompress(&srcinfo);

    return {
            dimensions,
            quantization,
            Y_coefficients,
            CrCb_coefficients
    };
}

std::vector<torch::Tensor> read_coefficients(const std::string &path) {
    // open the file
    FILE *infile;
    if ((infile = fopen(path.c_str(), "rb")) == nullptr) {
        return {};
    }

    // start decompression
    jpeg_decompress_struct srcinfo;
    struct jpeg_error_mgr srcerr;

    srcinfo.err = jpeg_std_error(&srcerr);
    srcinfo.err->output_message = [](j_common_ptr cinfo) {};
    jpeg_create_decompress(&srcinfo);

    jpeg_stdio_src(&srcinfo, infile);

    auto ret = read_coefficients_using(srcinfo);

    jpeg_destroy_decompress(&srcinfo);
    fclose(infile);

    return ret;
}

std::vector<torch::Tensor> quantize_at_quality(torch::Tensor pixels, int quality, bool baseline=true) {
    // Use libjpeg to compress the pixels into a memory buffer, this is slightly wasteful
    // as it performs entropy coding
    struct jpeg_compress_struct cinfo;
    struct jpeg_error_mgr jerr;

    cinfo.err = jpeg_std_error(&jerr);
    cinfo.err->output_message = [](j_common_ptr cinfo) {};
    jpeg_create_compress(&cinfo);

    unsigned long compressed_size;
    unsigned char *buffer = NULL;
    jpeg_mem_dest(&cinfo, &buffer, &compressed_size);

    cinfo.image_width = pixels.size(2);
    cinfo.image_height = pixels.size(1);
    cinfo.input_components = pixels.size(0);
    cinfo.in_color_space = pixels.size(0) > 1 ? JCS_RGB : JCS_GRAYSCALE;

    jpeg_set_defaults(&cinfo);
    jpeg_set_quality(&cinfo, quality, int(baseline));

    // No way that I know of to pass planar images to libjpeg
    auto channel_interleaved = (pixels * 255.f).to(torch::kByte).transpose(0,2).transpose(0,1).contiguous();

    jpeg_start_compress(&cinfo, TRUE);

    JSAMPROW row_pointer[1];
    while (cinfo.next_scanline < cinfo.image_height) {
        row_pointer[0] = channel_interleaved.data<JSAMPLE>() + cinfo.next_scanline * channel_interleaved.size(1) * channel_interleaved.size(2);
        jpeg_write_scanlines(&cinfo, row_pointer, 1);
    }

    jpeg_finish_compress(&cinfo);
    jpeg_destroy_compress(&cinfo);

    // Decompress memory buffer to DCT coefficients
    jpeg_decompress_struct srcinfo;
    struct jpeg_error_mgr srcerr;

    srcinfo.err = jpeg_std_error(&srcerr);
    jpeg_create_decompress(&srcinfo);

    jpeg_mem_src(&srcinfo, buffer, compressed_size);

    auto ret = read_coefficients_using(srcinfo);

    jpeg_destroy_decompress(&srcinfo);
    free(buffer);

    return ret;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("read_coefficients", &read_coefficients, "Read DCT coefficients from the a JPEG file",
        py::arg("path"));
    m.def("quantize_at_quality", &quantize_at_quality, "Quantize pixels using libjpeg at the given quality",
        py::arg("pixels"), py::arg("quality"), py::arg("baseline") = true);
}