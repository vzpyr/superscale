#ifndef SUPERSCALE_ONNX_H
#define SUPERSCALE_ONNX_H

#include <stddef.h>
#include <stdint.h>

typedef enum {
  SC_OP_CONV = 1,
  SC_OP_LEAKY_RELU = 2,
  SC_OP_PRELU = 3,
  SC_OP_CLIP = 4,
  SC_OP_ADD = 5,
  SC_OP_MUL = 6,
  SC_OP_CONCAT = 7,
  SC_OP_RESIZE = 8,
  SC_OP_DEPTH_TO_SPACE = 9,
} sc_op;

typedef struct {
  char* name;
  int64_t dims[4];
  int rank;
  size_t count;
  uint8_t* data;
  int is_const;
} sc_tensor;

typedef struct {
  sc_op op;
  int in[16];
  int n_in;
  int out;
  int ks;
  int stride;
  int pad;
  int dil;
  int group;
  int axis;
  int has_bias;
  int has_min;
  int has_max;
  float alpha;
  float fmin;
  float fmax;
  int block;
  int mode;
  int nearest;
  int asymmetric;
  int slope_in;
  int scale_in;
  int w_off;
  int b_off;
  int slope_off;
} sc_node;

typedef struct {
  sc_tensor* tensors;
  int n_tensors;
  sc_node* nodes;
  int n_nodes;
  int input;
  int output;
} sc_model;

int sc_model_load(const char* path, sc_model* model, char** err);
void sc_model_free(sc_model* model);

#endif
