#pragma once

#include <jpeglib.h>

EXTERN(void) jpeg_mem_dest(j_compress_ptr cinfo, unsigned char **outbuffer,
unsigned long *outsize);
EXTERN(void) jpeg_mem_src(j_decompress_ptr cinfo,
const unsigned char *inbuffer, unsigned long insize);