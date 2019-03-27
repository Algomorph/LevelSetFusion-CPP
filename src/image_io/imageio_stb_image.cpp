#include "../image_io/imageio_stb_image.hpp"

#define STB_IMAGE_IMPLEMENTATION
#include "../image_io/stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../image_io/stb_image_write.h"


unsigned char* image_io::stbi_load(char const* filename, int* x, int* y, int* comp, int req_comp){
  return ::stbi_load(filename, x, y, comp, req_comp);
}

unsigned short* image_io::stbi_load_16(char const* filename, int* x, int* y, int* comp, int req_comp){
	return ::stbi_load_16(filename, x, y, comp, req_comp);
}

void image_io::stbi_image_free(void* retval_from_stbi_load){
  ::stbi_image_free(retval_from_stbi_load);
}

int image_io::stbi_write_png(
  char const *filename, int w, int h, int comp, const void* data, int stride_in_bytes){
  return ::stbi_write_png(filename, w, h, comp, data, stride_in_bytes);
}
