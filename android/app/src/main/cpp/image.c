#include "image.h"

static uint8_t to_byte(float value) {
  const float scaled = value * 255.0f + 0.5f;
  if (scaled <= 0.0f) {
    return 0;
  }
  if (scaled >= 255.0f) {
    return 255;
  }
  return (uint8_t)scaled;
}

void sc_rgba_to_planar(const uint8_t* rgba, size_t pixels, float* planes) {
  for (size_t i = 0; i < pixels; i++) {
    planes[i] = (float)rgba[i * 4] / 255.0f;
    planes[pixels + i] = (float)rgba[i * 4 + 1] / 255.0f;
    planes[pixels * 2 + i] = (float)rgba[i * 4 + 2] / 255.0f;
  }
}

void sc_planar_to_rgba(const float* planes, size_t pixels, uint8_t* rgba) {
  for (size_t i = 0; i < pixels; i++) {
    rgba[i * 4] = to_byte(planes[i]);
    rgba[i * 4 + 1] = to_byte(planes[pixels + i]);
    rgba[i * 4 + 2] = to_byte(planes[pixels * 2 + i]);
    rgba[i * 4 + 3] = 255;
  }
}
