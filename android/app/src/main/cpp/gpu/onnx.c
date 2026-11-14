#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../err.h"
#include "onnx.h"

typedef struct {
  uint32_t key;
  int kind;
  int64_t i64;
  const uint8_t* data;
  size_t len;
} pb_field;

typedef struct {
  const uint8_t* p;
  size_t n;
  size_t i;
} pb;

static int pb_varint(const uint8_t* p, size_t n, size_t* i, uint64_t* out) {
  uint64_t v = 0;
  int shift = 0;
  while (*i < n && shift < 64) {
    uint8_t b = p[(*i)++];
    v |= (uint64_t)(b & 0x7fu) << shift;
    if (!(b & 0x80u)) {
      *out = v;
      return 0;
    }
    shift += 7;
  }
  return 1;
}

static int pb_next(pb* r, pb_field* f) {
  if (r->i >= r->n) {
    return 1;
  }
  uint64_t k = 0;
  if (pb_varint(r->p, r->n, &r->i, &k)) {
    return 1;
  }
  const uint32_t key = (uint32_t)(k >> 3);
  const int wire = (int)(k & 7u);
  memset(f, 0, sizeof(*f));
  f->key = key;
  if (wire == 0) {
    uint64_t v = 0;
    if (pb_varint(r->p, r->n, &r->i, &v)) {
      return 1;
    }
    f->kind = 0;
    f->i64 = (int64_t)v;
  } else if (wire == 2) {
    uint64_t len = 0;
    if (pb_varint(r->p, r->n, &r->i, &len)) {
      return 1;
    }
    if (r->i + len > r->n) {
      return 1;
    }
    f->kind = 2;
    f->data = r->p + r->i;
    f->len = (size_t)len;
    r->i += (size_t)len;
  } else if (wire == 5) {
    if (r->i + 4 > r->n) {
      return 1;
    }
    f->kind = 3;
    r->i += 4;
  } else if (wire == 1) {
    if (r->i + 8 > r->n) {
      return 1;
    }
    f->kind = 4;
    r->i += 8;
  } else {
    return 1;
  }
  return 0;
}

static const pb_field* pb_find(const pb_field* items, int n, uint32_t key) {
  for (int i = 0; i < n; i++) {
    if (items[i].key == key) {
      return &items[i];
    }
  }
  return NULL;
}

static int pb_text(const pb_field* items, int n, uint32_t key, char** out) {
  const pb_field* f = pb_find(items, n, key);
  if (!f || f->kind != 2) {
    return 1;
  }
  char* buf = (char*)malloc(f->len + 1);
  if (!buf) {
    return 1;
  }
  memcpy(buf, f->data, f->len);
  buf[f->len] = 0;
  *out = buf;
  return 0;
}

static int pb_int(const pb_field* items, int n, uint32_t key, int fallback) {
  const pb_field* f = pb_find(items, n, key);
  return f && f->kind == 0 ? (int)f->i64 : fallback;
}

static void pb_ints(const pb_field* items, int n, uint32_t key, int64_t* out, int cap, int* count) {
  int total = 0;
  for (int i = 0; i < n; i++) {
    if (items[i].key != key) {
      continue;
    }
    if (items[i].kind == 0) {
      if (total < cap) {
        out[total] = items[i].i64;
      }
      total++;
    } else if (items[i].kind == 2) {
      pb sub = {items[i].data, items[i].len, 0};
      pb_field item;
      while (!pb_next(&sub, &item)) {
        if (item.kind != 0) {
          continue;
        }
        if (total < cap) {
          out[total] = item.i64;
        }
        total++;
      }
    }
  }
  *count = total;
}

static void pb_floats(const pb_field* items, int n, uint32_t key, float* out, int cap, int* count) {
  int total = 0;
  for (int i = 0; i < n; i++) {
    if (items[i].key != key || items[i].kind != 2) {
      continue;
    }
    int avail = (int)(items[i].len / 4u);
    for (int j = 0; j < avail && total < cap; j++) {
      float v = 0.0f;
      memcpy(&v, items[i].data + (size_t)j * 4u, 4);
      out[total] = v;
      total++;
    }
  }
  *count = total;
}

static pb_field* pb_slice(const uint8_t* p, size_t n, int* count) {
  pb r = {p, n, 0};
  pb_field tmp;
  int c = 0;
  while (!pb_next(&r, &tmp)) {
    c++;
  }
  pb_field* items = (pb_field*)calloc((size_t)(c > 0 ? c : 1), sizeof(pb_field));
  if (!items) {
    *count = 0;
    return NULL;
  }
  r.i = 0;
  int idx = 0;
  while (idx < c && !pb_next(&r, &items[idx])) {
    idx++;
  }
  *count = idx;
  return items;
}

typedef struct {
  char** names;
  int n;
  int cap;
} big_pool;

static int big_find(big_pool* pool, const char* name) {
  for (int i = 0; i < pool->n; i++) {
    if (strcmp(pool->names[i], name) == 0) {
      return i;
    }
  }
  return -1;
}

static int big_add(big_pool* pool, const char* name) {
  int found = big_find(pool, name);
  if (found >= 0) {
    return found;
  }
  if (pool->n == pool->cap) {
    int cap = pool->cap ? pool->cap * 2 : 256;
    char** grown = (char**)realloc(pool->names, (size_t)cap * sizeof(char*));
    if (!grown) {
      return -1;
    }
    pool->names = grown;
    pool->cap = cap;
  }
  size_t len = strlen(name);
  char* copy = (char*)malloc(len + 1);
  if (!copy) {
    return -1;
  }
  memcpy(copy, name, len + 1);
  pool->names[pool->n] = copy;
  return pool->n++;
}

static int tensor_from_pb(const pb_field* t, int tn, char* name, sc_tensor* dst) {
  int64_t dims[4] = {1, 1, 1, 1};
  int total = 0;
  pb_ints(t, tn, 1, dims, 4, &total);
  if (total > 4) return 1;
  int64_t shape[4] = {1, 1, 1, 1};
  for (int d = 0; d < total && d < 4; d++) {
    shape[d] = dims[d] > 0 ? dims[d] : 1;
  }
  size_t count = (size_t)shape[0] * (size_t)shape[1] * (size_t)shape[2] * (size_t)shape[3];
  float* values = (float*)calloc(count ? count : 1, sizeof(float));
  if (!values) {
    return 1;
  }
  int data_type = pb_int(t, tn, 2, 1);
  if (data_type == 1) {
    int got = 0;
    pb_floats(t, tn, 4, values, (int)count, &got);
    if (got == 0) {
      const pb_field* raw = pb_find(t, tn, 9);
      if (raw && raw->kind == 2) {
        int avail = (int)(raw->len / 4u);
        for (int j = 0; j < avail && j < (int)count; j++) {
          memcpy(&values[j], raw->data + (size_t)j * 4u, 4);
        }
      }
    }
  } else if (data_type == 6 || data_type == 7) {
    int64_t* ints = (int64_t*)calloc(count ? count : 1, sizeof(int64_t));
    if (!ints) {
      free(values);
      return 1;
    }
    int got = 0;
    pb_ints(t, tn, data_type == 6 ? 5 : 7, ints, (int)count, &got);
    if (got == 0) {
      const pb_field* raw = pb_find(t, tn, 9);
      if (raw && raw->kind == 2) {
        const size_t width = data_type == 6 ? 4u : 8u;
        const int avail = (int)(raw->len / width);
        for (int j = 0; j < avail && j < (int)count; j++) {
          if (width == 4u) {
            int32_t v = 0;
            memcpy(&v, raw->data + (size_t)j * width, 4);
            ints[j] = v;
          } else {
            memcpy(&ints[j], raw->data + (size_t)j * width, 8);
          }
        }
      }
    }
    for (size_t j = 0; j < count; j++) {
      values[j] = (float)ints[j];
    }
    free(ints);
  } else {
    free(values);
    return 1;
  }
  dst->name = name;
  dst->rank = total > 4 ? 4 : total;
  for (int d = 0; d < 4; d++) {
    dst->dims[d] = shape[d];
  }
  dst->count = count;
  dst->data = (uint8_t*)values;
  dst->is_const = 1;
  return 0;
}

int sc_model_load(const char* path, sc_model* model, char** err) {
  FILE* fp = fopen(path, "rb");
  if (!fp) {
    return sc_err(err, "could not open model");
  }
  fseek(fp, 0, SEEK_END);
  long size = ftell(fp);
  fseek(fp, 0, SEEK_SET);
  if (size <= 0) {
    fclose(fp);
    return sc_err(err, "empty model");
  }
  uint8_t* buf = (uint8_t*)malloc((size_t)size);
  if (!buf) {
    fclose(fp);
    return sc_err(err, "out of memory");
  }
  if (fread(buf, 1, (size_t)size, fp) != (size_t)size) {
    free(buf);
    fclose(fp);
    return sc_err(err, "could not read model");
  }
  fclose(fp);
  memset(model, 0, sizeof(*model));

  int top = 0;
  pb_field* root = pb_slice(buf, (size_t)size, &top);
  if (!root) {
    free(buf);
    return sc_err(err, "could not parse model");
  }
  const pb_field* graph = pb_find(root, top, 7);
  if (!graph || graph->kind != 2) {
    free(root);
    free(buf);
    return sc_err(err, "model has no graph");
  }

  int gn = 0;
  pb_field* g = pb_slice(graph->data, graph->len, &gn);
  if (!g) {
    free(root);
    free(buf);
    return sc_err(err, "could not parse graph");
  }

  int node_count = 0;
  for (int i = 0; i < gn; i++) {
    if (g[i].key == 1 && g[i].kind == 2) {
      node_count++;
    }
  }

  sc_node* out_nodes = (sc_node*)calloc((size_t)(node_count ? node_count : 1), sizeof(sc_node));
  if (!out_nodes) {
    free(g);
    free(root);
    free(buf);
    return sc_err(err, "out of memory");
  }

  int inits = 0;
  for (int i = 0; i < gn; i++) {
    if (g[i].key == 5 && g[i].kind == 2) {
      inits++;
    }
  }

  big_pool pool = {NULL, 0, 0};
  int tensor_cap = inits + node_count + 16;
  sc_tensor* tensors = (sc_tensor*)calloc((size_t)tensor_cap, sizeof(sc_tensor));
  if (!tensors) {
    free(out_nodes);
    free(g);
    free(root);
    free(buf);
    return sc_err(err, "out of memory");
  }
  int n_tensors = 0;

  for (int i = 0; i < gn && n_tensors < tensor_cap; i++) {
    if (g[i].key != 5 || g[i].kind != 2) {
      continue;
    }
    int tn = 0;
    pb_field* t = pb_slice(g[i].data, g[i].len, &tn);
    if (!t) {
      continue;
    }
    char* name = NULL;
    if (pb_text(t, tn, 8, &name) || !name) {
      free(t);
      continue;
    }
    int id = big_add(&pool, name);
    free(name);
    if (id >= 0 && !tensor_from_pb(t, tn, pool.names[id], &tensors[n_tensors])) {
      n_tensors++;
    }
    free(t);
  }

  int produced = 0;
  int node_slot = 0;
  for (int i = 0; i < gn; i++) {
    if (g[i].key != 1 || g[i].kind != 2) {
      continue;
    }
    int nn = 0;
    pb_field* nd = pb_slice(g[i].data, g[i].len, &nn);
    if (!nd) {
      continue;
    }
    char* op_type = NULL;
    pb_text(nd, nn, 4, &op_type);
    if (!op_type) {
      free(nd);
      continue;
    }
    int in_count = 0;
    char* in_names[16];
    for (int k = 0; k < nn && in_count < 16; k++) {
      if (nd[k].key != 1 || nd[k].kind != 2) {
        continue;
      }
      char* tmp = (char*)malloc(nd[k].len + 1);
      if (!tmp) {
        continue;
      }
      memcpy(tmp, nd[k].data, nd[k].len);
      tmp[nd[k].len] = 0;
      in_names[in_count++] = tmp;
    }
    char* out_name = NULL;
    for (int k = 0; k < nn && !out_name; k++) {
      if (nd[k].key != 2 || nd[k].kind != 2) {
        continue;
      }
      char* tmp = (char*)malloc(nd[k].len + 1);
      if (!tmp) {
        continue;
      }
      memcpy(tmp, nd[k].data, nd[k].len);
      tmp[nd[k].len] = 0;
      out_name = tmp;
    }

    if (strcmp(op_type, "Constant") == 0 && out_name && n_tensors < tensor_cap) {
      for (int k = 0; k < nn; k++) {
        if (nd[k].key != 5 || nd[k].kind != 2) {
          continue;
        }
        int an = 0;
        pb_field* at = pb_slice(nd[k].data, nd[k].len, &an);
        if (!at) {
          continue;
        }
        const pb_field* tp = pb_find(at, an, 5);
        if (tp && tp->kind == 2) {
          int vn = 0;
          pb_field* vt = pb_slice(tp->data, tp->len, &vn);
          int id = vt ? big_add(&pool, out_name) : -1;
          if (id >= 0 && !tensor_from_pb(vt, vn, pool.names[id], &tensors[n_tensors])) {
            n_tensors++;
          }
          free(vt);
        }
        free(at);
      }
      free(op_type);
      for (int k = 0; k < in_count; k++) {
        free(in_names[k]);
      }
      free(out_name);
      free(nd);
      continue;
    }

    sc_node* node = &out_nodes[node_slot];
    memset(node, 0, sizeof(*node));
    node->n_in = in_count;
    node->ks = 0;
    node->stride = 1;
    node->pad = 0;
    node->dil = 1;
    node->group = 1;
    node->axis = 1;
    node->block = 1;
    node->mode = 0;
    node->nearest = 0;
    node->asymmetric = 0;
    node->slope_in = -1;
    node->scale_in = -1;
    node->w_off = -1;
    node->b_off = -1;
    node->slope_off = -1;
    node->out = big_add(&pool, out_name ? out_name : "");
    for (int k = 0; k < in_count; k++) {
      node->in[k] = in_names[k][0] ? big_add(&pool, in_names[k]) : -1;
    }

    if (strcmp(op_type, "Conv") == 0) {
      node->op = SC_OP_CONV;
      node->has_bias = in_count > 2;
    } else if (strcmp(op_type, "LeakyRelu") == 0) {
      node->op = SC_OP_LEAKY_RELU;
      node->alpha = 0.01f;
    } else if (strcmp(op_type, "PRelu") == 0) {
      node->op = SC_OP_PRELU;
      node->slope_in = in_count > 1 ? node->in[1] : -1;
    } else if (strcmp(op_type, "Clip") == 0) {
      node->op = SC_OP_CLIP;
      node->has_min = in_count > 1 ? 2 : 0;
      node->has_max = in_count > 2 ? 2 : 0;
    } else if (strcmp(op_type, "Add") == 0) {
      node->op = SC_OP_ADD;
    } else if (strcmp(op_type, "Mul") == 0) {
      node->op = SC_OP_MUL;
    } else if (strcmp(op_type, "Concat") == 0) {
      node->op = SC_OP_CONCAT;
    } else if (strcmp(op_type, "Resize") == 0) {
      node->op = SC_OP_RESIZE;
      node->scale_in = in_count > 2 ? node->in[2] : -1;
      if (in_count > 3 && node->in[3] < 0) {
        node->scale_in = -1;
      }
    } else if (strcmp(op_type, "DepthToSpace") == 0) {
      node->op = SC_OP_DEPTH_TO_SPACE;
    } else {
      free(op_type);
      for (int k = 0; k < in_count; k++) {
        free(in_names[k]);
      }
      free(out_name);
      free(nd);
      continue;
    }

    for (int k = 0; k < nn; k++) {
      if (nd[k].key != 5 || nd[k].kind != 2) {
        continue;
      }
      int an = 0;
      pb_field* at = pb_slice(nd[k].data, nd[k].len, &an);
      if (!at) {
        continue;
      }
      char* aname = NULL;
      pb_text(at, an, 1, &aname);
      if (aname) {
        int64_t ints[8];
        int nints = 0;
        pb_ints(at, an, 8, ints, 8, &nints);
        if (nints == 0) {
          const pb_field* scalar = pb_find(at, an, 3);
          if (scalar && scalar->kind == 0) {
            ints[0] = scalar->i64;
            nints = 1;
          }
        }
        if (strcmp(aname, "kernel_shape") == 0 && nints > 0) {
          node->ks = (int)ints[0];
        } else if (strcmp(aname, "pads") == 0 && nints > 0) {
          node->pad = (int)ints[0];
        } else if (strcmp(aname, "strides") == 0 && nints > 0) {
          node->stride = (int)ints[0];
        } else if (strcmp(aname, "dilations") == 0 && nints > 0) {
          node->dil = (int)ints[0];
        } else if (strcmp(aname, "group") == 0 && nints > 0) {
          node->group = (int)ints[0];
        } else if (strcmp(aname, "axis") == 0 && nints > 0) {
          node->axis = (int)ints[0];
        } else if (strcmp(aname, "blocksize") == 0 && nints > 0) {
          node->block = (int)ints[0];
        } else if (strcmp(aname, "mode") == 0) {
          char* mode = NULL;
          pb_text(at, an, 4, &mode);
          if (mode) {
            node->mode = strcmp(mode, "CRD") == 0 ? 1 : 0;
            node->nearest = strcmp(mode, "nearest") == 0 ? 1 : 0;
            free(mode);
          }
        } else if (strcmp(aname, "coordinate_transformation_mode") == 0) {
          char* coord = NULL;
          pb_text(at, an, 4, &coord);
          if (coord) {
            node->asymmetric = strcmp(coord, "asymmetric") == 0 ? 1 : 0;
            free(coord);
          }
        } else if (strcmp(aname, "alpha") == 0) {
          float alpha = 0.0f;
          const pb_field* fv = pb_find(at, an, 2);
          if (fv && fv->kind == 2 && fv->len >= 4) {
            memcpy(&alpha, fv->data, 4);
            node->alpha = alpha;
          }
        }
        free(aname);
      }
      free(at);
    }

    free(op_type);
    for (int k = 0; k < in_count; k++) {
      free(in_names[k]);
    }
    free(out_name);
    free(nd);
    node_slot++;
    produced++;
  }

  int n_inputs = 0;
  for (int i = 0; i < gn; i++) {
    if (g[i].key != 11 || g[i].kind != 2) {
      continue;
    }
    int vn = 0;
    pb_field* v = pb_slice(g[i].data, g[i].len, &vn);
    if (!v) {
      continue;
    }
    char* name = NULL;
    if (!pb_text(v, vn, 1, &name) && name) {
      int id = big_find(&pool, name);
      if (id < 0) {
        id = big_add(&pool, name);
      }
      if (n_inputs == 0) {
        model->input = id;
      }
      n_inputs++;
    }
    free(name);
    free(v);
  }
  for (int i = 0; i < gn; i++) {
    if (g[i].key != 12 || g[i].kind != 2) {
      continue;
    }
    int vn = 0;
    pb_field* v = pb_slice(g[i].data, g[i].len, &vn);
    if (!v) {
      continue;
    }
    char* name = NULL;
    if (!pb_text(v, vn, 1, &name) && name) {
      int id = big_find(&pool, name);
      if (id >= 0) {
        model->output = id;
      }
    }
    free(name);
    free(v);
  }

  model->n_nodes = produced;
  model->nodes = out_nodes;
  model->n_tensors = pool.n;
  model->tensors = (sc_tensor*)calloc((size_t)(pool.n > 0 ? pool.n : 1), sizeof(sc_tensor));
  if (!model->tensors) {
    for (int i = 0; i < n_tensors; i++) {
      free(tensors[i].data);
    }
    free(tensors);
    for (int i = 0; i < pool.n; i++) {
      free(pool.names[i]);
    }
    free(pool.names);
    free(out_nodes);
    free(g);
    free(root);
    free(buf);
    return sc_err(err, "out of memory");
  }
  for (int i = 0; i < pool.n; i++) {
    model->tensors[i].name = pool.names[i];
  }
  for (int i = 0; i < n_tensors; i++) {
    int id = big_find(&pool, tensors[i].name);
    if (id < 0) {
      continue;
    }
    model->tensors[id].dims[0] = tensors[i].dims[0];
    model->tensors[id].dims[1] = tensors[i].dims[1];
    model->tensors[id].dims[2] = tensors[i].dims[2];
    model->tensors[id].dims[3] = tensors[i].dims[3];
    model->tensors[id].rank = tensors[i].rank;
    model->tensors[id].count = tensors[i].count;
    model->tensors[id].data = tensors[i].data;
    model->tensors[id].is_const = 1;
  }
  free(tensors);
  free(pool.names);

  free(g);
  free(root);
  free(buf);
  if (model->n_nodes == 0) {
    sc_model_free(model);
    return sc_err(err, "model has no supported nodes");
  }
  return 0;
}

void sc_model_free(sc_model* model) {
  if (!model) {
    return;
  }
  if (model->tensors) {
    for (int i = 0; i < model->n_tensors; i++) {
      free(model->tensors[i].name);
      free(model->tensors[i].data);
    }
  }
  free(model->tensors);
  free(model->nodes);
  memset(model, 0, sizeof(*model));
}
