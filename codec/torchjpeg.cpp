#include <torch/extension.h>

#include <numeric>
#include <algorithm>
#include <array>
#include <vector>

#include <jpeglib.h>
#include "jdatadst.hpp"

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
        JBLOCKARRAY rowPtrs = srcinfo.mem->access_virt_barray((j_common_ptr) &srcinfo, src_coef_arrays[compNum],
                                                              rowNum, 1, FALSE);

        for (JDIMENSION blockNum = 0; blockNum < srcinfo.comp_info[compNum].width_in_blocks; blockNum++) {
            std::copy_n(rowPtrs[0][blockNum], DCTSIZE2, coefficients.data_ptr<int16_t>() + coefficients_written);
            coefficients_written += DCTSIZE2;
        }
    }

    std::copy_n(srcinfo.comp_info[compNum].quant_table->quantval, DCTSIZE2,
                quantization.data_ptr<int16_t>() + DCTSIZE2 * compNum);
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
    int cw = 0;
    extract_channel(srcinfo, src_coef_arrays, 0, Y_coefficients, quantization, cw);

    // extract CrCb channels
    auto CrCb_coefficients = torch::empty({}, torch::kShort);

    if (srcinfo.num_components > 1) {
        CrCb_coefficients = torch::empty({2,
                                          srcinfo.comp_info[1].height_in_blocks,
                                          srcinfo.comp_info[1].width_in_blocks,
                                          DCTSIZE,
                                          DCTSIZE
                                         }, torch::kShort);

        cw = 0;
        extract_channel(srcinfo, src_coef_arrays, 1, CrCb_coefficients, quantization, cw);
        extract_channel(srcinfo, src_coef_arrays, 2, CrCb_coefficients, quantization, cw);
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
    jpeg_decompress_struct srcinfo{};
    struct jpeg_error_mgr srcerr{};

    srcinfo.err = jpeg_std_error(&srcerr);
    srcinfo.err->output_message = [](j_common_ptr cinfo) {};
    jpeg_create_decompress(&srcinfo);

    jpeg_stdio_src(&srcinfo, infile);

    auto ret = read_coefficients_using(srcinfo);

    jpeg_destroy_decompress(&srcinfo);
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
    auto block_arrays = (jvirt_barray_ptr *) (*cinfo->mem->alloc_small)((j_common_ptr) cinfo,
                                                                        JPOOL_IMAGE,
                                                                        sizeof(jvirt_barray_ptr *) *
                                                                        cinfo->num_components);

    std::transform(cinfo->comp_info, cinfo->comp_info + cinfo->num_components, block_arrays,
                   [&](jpeg_component_info &compptr) {
                       int MCU_width = jdiv_round_up((long) cinfo->jpeg_width, (long) compptr.MCU_width);
                       int MCU_height = jdiv_round_up((long) cinfo->jpeg_height, (long) compptr.MCU_height);

                       return (cinfo->mem->request_virt_barray)((j_common_ptr) cinfo,
                                                                JPOOL_IMAGE,
                                                                TRUE,
                                                                MCU_width,
                                                                MCU_height,
                                                                compptr.v_samp_factor);
                   });

    return block_arrays;
}

void fill_extended_defaults(j_compress_ptr cinfo, int color_samp_factor = 2) {

    cinfo->jpeg_width = cinfo->image_width;
    cinfo->jpeg_height = cinfo->image_height;

    jpeg_set_defaults(cinfo);

    cinfo->comp_info[0].component_id = 0;
    cinfo->comp_info[0].h_samp_factor = 1;
    cinfo->comp_info[0].v_samp_factor = 1;
    cinfo->comp_info[0].quant_tbl_no = 0;
    cinfo->comp_info[0].width_in_blocks = jdiv_round_up(cinfo->jpeg_width, DCTSIZE);
    cinfo->comp_info[0].height_in_blocks = jdiv_round_up(cinfo->jpeg_height, DCTSIZE);
    cinfo->comp_info[0].MCU_width = 1;
    cinfo->comp_info[0].MCU_height = 1;

    if (cinfo->num_components > 1) {
        cinfo->comp_info[0].h_samp_factor = color_samp_factor;
        cinfo->comp_info[0].v_samp_factor = color_samp_factor;
        cinfo->comp_info[0].MCU_width = color_samp_factor;
        cinfo->comp_info[0].MCU_height = color_samp_factor;

        for (int c = 1; c < cinfo->num_components; c++) {
            cinfo->comp_info[c].component_id = c;
            cinfo->comp_info[c].h_samp_factor = 1;
            cinfo->comp_info[c].v_samp_factor = 1;
            cinfo->comp_info[c].quant_tbl_no = 1;
            cinfo->comp_info[c].width_in_blocks = jdiv_round_up(cinfo->jpeg_width, DCTSIZE * color_samp_factor);
            cinfo->comp_info[c].height_in_blocks = jdiv_round_up(cinfo->jpeg_width, DCTSIZE * color_samp_factor);
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
        JBLOCKARRAY rowPtrs = cinfo.mem->access_virt_barray((j_common_ptr) &cinfo, dest_coef_arrays[compNum],
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
                        torch::Tensor CrCb_coefficients = torch::empty({}, torch::kShort)) {
    FILE *outfile;
    if ((outfile = fopen(path.c_str(), "wb")) == nullptr) {
        return;
    }

    jpeg_compress_struct cinfo{};
    struct jpeg_error_mgr srcerr{};

    cinfo.err = jpeg_std_error(&srcerr);
    jpeg_create_compress(&cinfo);
    jpeg_stdio_dest(&cinfo, outfile);

    auto dct_dim_a = dimensions.accessor<int, 2>();

    cinfo.image_height = dct_dim_a[0][0];
    cinfo.image_width = dct_dim_a[0][1];
    cinfo.input_components = dimensions.size(0);
    cinfo.in_color_space = dimensions.size(0) > 1 ? JCS_RGB : JCS_GRAYSCALE;

    fill_extended_defaults(&cinfo);

    set_quantization(&cinfo, quantization);

    jvirt_barray_ptr *coef_dest = request_block_storage(&cinfo);
    jpeg_write_coefficients(&cinfo, coef_dest);

    int cw = 0;
    set_channel(cinfo, Y_coefficients, coef_dest, 0, cw);

    if (cinfo.num_components > 1) {
        cw = 0;
        set_channel(cinfo, CrCb_coefficients, coef_dest, 1, cw);
        set_channel(cinfo, CrCb_coefficients, coef_dest, 2, cw);
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

std::vector<torch::Tensor> quantize_at_quality(torch::Tensor pixels, int quality, bool baseline = true) {
    // Use libjpeg to compress the pixels into a memory buffer, this is slightly wasteful
    // as it performs entropy coding
    struct jpeg_compress_struct cinfo{};
    struct jpeg_error_mgr jerr{};

    cinfo.err = jpeg_std_error(&jerr);
    jpeg_create_compress(&cinfo);

    unsigned long compressed_size;
    unsigned char *buffer = nullptr;
    jpeg_mem_dest(&cinfo, &buffer, &compressed_size);

    cinfo.image_width = pixels.size(2);
    cinfo.image_height = pixels.size(1);
    cinfo.input_components = pixels.size(0);
    cinfo.in_color_space = pixels.size(0) > 1 ? JCS_RGB : JCS_GRAYSCALE;

    jpeg_set_defaults(&cinfo);
    jpeg_set_quality(&cinfo, quality, int(baseline));

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
    struct jpeg_error_mgr srcerr{};

    srcinfo.err = jpeg_std_error(&srcerr);
    jpeg_create_decompress(&srcinfo);

    jpeg_mem_src(&srcinfo, buffer, compressed_size);

    auto ret = read_coefficients_using(srcinfo);

    jpeg_destroy_decompress(&srcinfo);
    free_buffer(buffer);

    return ret;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("read_coefficients", &read_coefficients, "Read DCT coefficients from the a JPEG file",
          py::arg("path"));
    m.def("write_coefficients", &write_coefficients, "Write DCT coefficients to a JPEG file",
          py::arg("path"), py::arg("dimensions"), py::arg("quantization"), py::arg("Y_coefficients"),
          py::arg("CrCb_coefficients") = torch::empty({}, torch::kShort));
    m.def("quantize_at_quality", &quantize_at_quality, "Quantize pixels using libjpeg at the given quality",
          py::arg("pixels"), py::arg("quality"), py::arg("baseline") = true);
}