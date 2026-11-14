#ifndef SUPERSCALE_PLAN_H
#define SUPERSCALE_PLAN_H

#include <stddef.h>
#include <stdint.h>

#include "onnx.h"

typedef struct {
  uint32_t dst;
  uint32_t src;
  uint32_t count;
} sc_copy;

typedef struct {
  uint32_t off;
  const void* data;
  uint32_t bytes;
} sc_const_upload;

typedef struct {
  sc_op op;
  uint32_t out_off;
  uint32_t in_off;
  uint32_t in2_off;
  uint32_t w_off;
  uint32_t b_off;
  uint32_t n0;
  uint32_t n1;
  uint32_t n2;
  uint32_t n3;
  uint32_t n4;
  uint32_t n5;
  uint32_t n6;
  uint32_t n7;
  uint32_t s0[4];
  uint32_t s1[4];
  float f0;
  float f1;
  float f2;
  uint32_t kind;
  uint32_t copies;
  sc_copy copy[16];
} sc_step;

typedef struct {
  sc_step* steps;
  uint32_t n_steps;
  uint64_t arena_bytes;
  uint32_t arena_floats;
  uint32_t input_off;
  uint32_t output_off;
  uint32_t in_w;
  uint32_t in_h;
  uint32_t out_w;
  uint32_t out_h;
  uint32_t out_scale;
  sc_const_upload* consts;
  uint32_t n_consts;
} sc_plan;

int sc_weights_build(sc_model* model, float** out, size_t* count, char** err);
int sc_plan_build(const sc_model* model, uint32_t span, sc_plan* plan, char** err);
void sc_plan_free(sc_plan* plan);

#endif
