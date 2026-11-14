#ifndef SUPERSCALE_IMAGE_H
#define SUPERSCALE_IMAGE_H

#include <stddef.h>
#include <stdint.h>

void sc_rgba_to_planar(const uint8_t* rgba, size_t pixels, float* planes);
void sc_planar_to_rgba(const float* planes, size_t pixels, uint8_t* rgba);

#endif
