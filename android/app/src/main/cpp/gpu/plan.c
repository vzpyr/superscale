#include "plan.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "../err.h"

static float sc_at(const sc_tensor* t, size_t i) {
  float v = 0.0f;
  memcpy(&v, (const uint8_t*)t->data + i * 4, 4);
  return v;
}

typedef struct {
  uint32_t off;
  uint32_t size;
} sc_block;

typedef struct {
  sc_block* items;
  int n;
  int cap;
  uint64_t top;
  uint64_t peak;
} sc_arena;

static int sc_arena_init(sc_arena* a) {
  a->cap = 64;
  a->n = 0;
  a->top = 0;
  a->peak = 0;
  a->items = (sc_block*)malloc((size_t)a->cap * sizeof(sc_block));
  return a->items != NULL;
}

static void sc_arena_free(sc_arena* a) {
  free(a->items);
  a->items = NULL;
  a->n = 0;
}

static uint32_t sc_arena_alloc(sc_arena* a, uint32_t size) {
  if (size == 0) {
    size = 4;
  }
  for (int i = 0; i < a->n; i++) {
    if (a->items[i].size < size) {
      continue;
    }
    uint32_t off = a->items[i].off;
    if (a->items[i].size == size) {
      memmove(&a->items[i], &a->items[i + 1], (size_t)(a->n - i - 1) * sizeof(sc_block));
      a->n--;
    } else {
      a->items[i].off += size;
      a->items[i].size -= size;
    }
    return off;
  }
  uint32_t off = (uint32_t)a->top;
  a->top += size;
  if (a->top > a->peak) {
    a->peak = a->top;
  }
  return off;
}

static void sc_arena_release(sc_arena* a, uint32_t off, uint32_t size) {
  if (size == 0) {
    return;
  }
  if (a->n == a->cap) {
    int cap = a->cap * 2;
    sc_block* grown = (sc_block*)realloc(a->items, (size_t)cap * sizeof(sc_block));
    if (!grown) {
      return;
    }
    a->items = grown;
    a->cap = cap;
  }
  int i = 0;
  while (i < a->n && a->items[i].off < off) {
    i++;
  }
  memmove(&a->items[i + 1], &a->items[i], (size_t)(a->n - i) * sizeof(sc_block));
  a->items[i].off = off;
  a->items[i].size = size;
  a->n++;
  for (int j = 0; j + 1 < a->n;) {
    if (a->items[j].off + a->items[j].size == a->items[j + 1].off) {
      a->items[j].size += a->items[j + 1].size;
      memmove(&a->items[j + 1], &a->items[j + 2], (size_t)(a->n - j - 2) * sizeof(sc_block));
      a->n--;
    } else {
      j++;
    }
  }
}

static int sc_reserve(float** buf, size_t* cap, size_t used, size_t need, char** err) {
  if (used + need <= *cap) {
    return 0;
  }
  size_t next = *cap;
  while (used + need > next) {
    next *= 2;
  }
  float* grown = (float*)realloc(*buf, next * sizeof(float));
  if (!grown) {
    free(*buf);
    *buf = NULL;
    return sc_err(err, "out of memory");
  }
  *buf = grown;
  *cap = next;
  return 0;
}

int sc_weights_build(sc_model* model, float** out, size_t* count, char** err) {
  size_t cap = 1u << 20;
  size_t used = 0;
  float* buf = (float*)malloc(cap * sizeof(float));
  if (!buf) {
    return sc_err(err, "out of memory");
  }

  for (int i = 0; i < model->n_nodes; i++) {
    sc_node* n = &model->nodes[i];
    if (n->op == SC_OP_CONV) {
      if (n->in[1] < 0) {
        free(buf);
        return sc_err(err, "convolution without weights");
      }
      const sc_tensor* w = &model->tensors[n->in[1]];
      if (w->rank != 4) {
        free(buf);
        return sc_err(err, "unsupported convolution weights");
      }
      uint32_t co = (uint32_t)w->dims[0];
      uint32_t ci = (uint32_t)w->dims[1];
      uint32_t kh = (uint32_t)w->dims[2];
      uint32_t kw = (uint32_t)w->dims[3];
      if (kh != kw || n->ks != (int)kh) {
        free(buf);
        return sc_err(err, "unsupported convolution kernel");
      }
      uint32_t cop = (co + 15u) & ~15u;
      size_t need = (size_t)ci * kh * kw * cop + (n->has_bias ? co : 0) + 4;
      if (sc_reserve(&buf, &cap, used, need, err)) {
        return 1;
      }
      n->w_off = (int)used;
      for (uint32_t c = 0; c < ci; c++) {
        for (uint32_t k = 0; k < kh * kw; k++) {
          for (uint32_t o = 0; o < cop; o++) {
            buf[used++] = o < co ? sc_at(w, (size_t)(o * ci + c) * kh * kw + k) : 0.0f;
          }
        }
      }
      if (n->has_bias) {
        if (n->in[2] < 0) {
          free(buf);
          return sc_err(err, "convolution without bias");
        }
        const sc_tensor* b = &model->tensors[n->in[2]];
        n->b_off = (int)used;
        for (uint32_t o = 0; o < co; o++) {
          buf[used++] = sc_at(b, o);
        }
      } else {
        n->b_off = -1;
      }
    } else if (n->op == SC_OP_PRELU) {
      if (n->slope_in < 0) {
        free(buf);
        return sc_err(err, "prelu without slope");
      }
      const sc_tensor* s = &model->tensors[n->slope_in];
      size_t n_slope = s->count ? s->count : 1;
      if (sc_reserve(&buf, &cap, used, n_slope, err)) {
        return 1;
      }
      n->slope_off = (int)used;
      for (size_t k = 0; k < n_slope; k++) {
        buf[used++] = sc_at(s, k);
      }
    }
  }
  *out = buf;
  *count = used;
  return 0;
}

int sc_plan_build(const sc_model* model, uint32_t span, sc_plan* plan, char** err) {
  memset(plan, 0, sizeof(*plan));
  const int nt = model->n_tensors;
  const int nn = model->n_nodes;
  if (model->input < 0 || model->input >= nt || model->output < 0 || model->output >= nt) {
    return sc_err(err, "invalid model graph");
  }

  int64_t* dims = (int64_t*)calloc((size_t)nt * 4, sizeof(int64_t));
  uint32_t* off = (uint32_t*)calloc((size_t)nt, sizeof(uint32_t));
  uint8_t* placed = (uint8_t*)calloc((size_t)nt, 1);
  int* last = (int*)malloc((size_t)nt * sizeof(int));
  sc_step* steps = (sc_step*)calloc((size_t)nn, sizeof(sc_step));
  sc_const_upload* consts = (sc_const_upload*)calloc((size_t)nt, sizeof(sc_const_upload));
  sc_arena arena;
  int rc = 0;

  if (!dims || !off || !placed || !last || !steps || !consts || !sc_arena_init(&arena)) {
    free(dims);
    free(off);
    free(placed);
    free(last);
    free(steps);
    free(consts);
    return sc_err(err, "out of memory");
  }
  for (int i = 0; i < nt; i++) {
    last[i] = -1;
  }

  for (int i = 0; i < nt; i++) {
    if (model->tensors[i].is_const) {
      for (int d = 0; d < 4; d++) {
        dims[i * 4 + d] = model->tensors[i].dims[d];
      }
    } else {
      for (int d = 0; d < 4; d++) {
        dims[i * 4 + d] = 0;
      }
    }
  }
  dims[model->input * 4 + 0] = 1;
  dims[model->input * 4 + 1] = 3;
  dims[model->input * 4 + 2] = (int64_t)span;
  dims[model->input * 4 + 3] = (int64_t)span;

  for (int i = 0; i < nn && !rc; i++) {
    const sc_node* n = &model->nodes[i];
    sc_step* s = &steps[i];
    s->op = n->op;
    int64_t* o = &dims[n->out * 4];
    const int64_t* a = n->in[0] >= 0 ? &dims[n->in[0] * 4] : NULL;

    switch (n->op) {
    case SC_OP_CONV: {
      if (!a || n->in[1] < 0 || a[1] <= 0) {
        rc = sc_err(err, "convolution on unknown shape");
        break;
      }
      const int64_t* w = &dims[n->in[1] * 4];
      if (w[0] <= 0) {
        rc = sc_err(err, "convolution on unknown shape");
        break;
      }
      if (n->group != 1 || n->stride != 1 || n->dil != 1 || n->ks != 3) {
        rc = sc_err(err, "unsupported convolution configuration");
        break;
      }
      if (w[1] != a[1]) {
        rc = sc_err(err, "convolution channel mismatch");
        break;
      }
      o[0] = a[0];
      o[1] = w[0];
      o[2] = (a[2] + 2 * n->pad - (n->ks - 1) - 1) / n->stride + 1;
      o[3] = (a[3] + 2 * n->pad - (n->ks - 1) - 1) / n->stride + 1;
      s->n0 = (uint32_t)o[1];
      s->n1 = (uint32_t)a[1];
      s->n2 = (uint32_t)a[2];
      s->n3 = (uint32_t)a[3];
      s->n4 = (uint32_t)o[2];
      s->n5 = (uint32_t)o[3];
      s->n6 = (uint32_t)n->pad;
      s->s0[0] = (uint32_t)n->stride;
      s->s0[1] = n->has_bias ? 1u : 0u;
      s->n7 = (s->n0 + 15u) & ~15u;
      s->w_off = n->w_off < 0 ? 0u : (uint32_t)n->w_off;
      s->b_off = n->b_off < 0 ? 0u : (uint32_t)n->b_off;
      break;
    }
    case SC_OP_LEAKY_RELU:
    case SC_OP_PRELU:
    case SC_OP_CLIP: {
      if (!a || a[1] <= 0) {
        rc = sc_err(err, "activation on unknown shape");
        break;
      }
      o[0] = a[0];
      o[1] = a[1];
      o[2] = a[2];
      o[3] = a[3];
      s->n0 = (uint32_t)a[1];
      s->n1 = (uint32_t)a[2];
      s->n2 = (uint32_t)a[3];
      if (n->op == SC_OP_LEAKY_RELU) {
        s->kind = 0;
        s->f0 = n->alpha;
      } else if (n->op == SC_OP_CLIP) {
        float lo = -3.40282347e38f;
        float hi = 3.40282347e38f;
        s->kind = 1;
        if (n->has_min == 1) {
          lo = n->fmin;
        } else if (n->has_min == 2) {
          const sc_tensor* t = n->in[1] >= 0 ? &model->tensors[n->in[1]] : NULL;
          if (!t || !t->is_const) {
            rc = sc_err(err, "dynamic clip bounds unsupported");
            break;
          }
          lo = sc_at(t, 0);
        }
        if (n->has_max == 1) {
          hi = n->fmax;
        } else if (n->has_max == 2) {
          const sc_tensor* t = n->in[2] >= 0 ? &model->tensors[n->in[2]] : NULL;
          if (!t || !t->is_const) {
            rc = sc_err(err, "dynamic clip bounds unsupported");
            break;
          }
          hi = sc_at(t, 0);
        }
        s->f0 = lo;
        s->f1 = hi;
      } else {
        if (n->slope_off < 0) {
          rc = sc_err(err, "prelu without slope");
          break;
        }
        s->kind = 2;
        s->w_off = (uint32_t)n->slope_off;
      }
      break;
    }
    case SC_OP_ADD:
    case SC_OP_MUL: {
      if (!a || n->in[1] < 0) {
        rc = sc_err(err, "arithmetic on unknown shape");
        break;
      }
      const int64_t* b = &dims[n->in[1] * 4];
      int64_t ob[4];
      for (int d = 0; d < 4; d++) {
        int64_t da = a[d] > 0 ? a[d] : 1;
        int64_t db = b[d] > 0 ? b[d] : 1;
        if (da != db && da != 1 && db != 1) {
          rc = sc_err(err, "unsupported broadcast");
          break;
        }
        ob[d] = da > db ? da : db;
      }
      if (rc) {
        break;
      }
      for (int d = 0; d < 4; d++) {
        o[d] = ob[d];
      }
      for (int pass = 0; pass < 2; pass++) {
        const int64_t* shape = pass == 0 ? a : b;
        uint32_t* out = pass == 0 ? s->s0 : s->s1;
        const int64_t w = shape[3] > 0 ? shape[3] : 1;
        const int64_t h = shape[2] > 0 ? shape[2] : 1;
        const uint32_t sx = 1u;
        const uint32_t sy = (uint32_t)w;
        const uint32_t sc = (uint32_t)(w * h);
        out[0] = (shape[1] == 1 && ob[1] != 1) ? 0u : sc;
        out[1] = (shape[2] == 1 && ob[2] != 1) ? 0u : sy;
        out[2] = (shape[3] == 1 && ob[3] != 1) ? 0u : sx;
        out[3] = 0u;
      }
      s->n0 = (uint32_t)ob[1];
      s->n1 = (uint32_t)ob[2];
      s->n2 = (uint32_t)ob[3];
      s->kind = n->op == SC_OP_ADD ? 0u : 1u;
      break;
    }
    case SC_OP_CONCAT: {
      if (n->axis != 1) {
        rc = sc_err(err, "unsupported concat axis");
        break;
      }
      int64_t total = 0;
      for (int k = 0; k < n->n_in; k++) {
        if (n->in[k] < 0 || model->tensors[n->in[k]].is_const) {
          rc = sc_err(err, "unsupported concat input");
          break;
        }
        const int64_t* p = &dims[n->in[k] * 4];
        if (p[1] <= 0) {
          rc = sc_err(err, "concat on unknown shape");
          break;
        }
        total += p[1];
      }
      if (rc) {
        break;
      }
      o[0] = 1;
      o[1] = total;
      o[2] = dims[n->in[0] * 4 + 2];
      o[3] = dims[n->in[0] * 4 + 3];
      s->copies = (uint32_t)n->n_in;
      break;
    }
    case SC_OP_RESIZE: {
      if (!a || n->scale_in < 0) {
        rc = sc_err(err, "resize on unknown shape");
        break;
      }
      const sc_tensor* sc = &model->tensors[n->scale_in];
      if (!sc->is_const || sc->count < 4) {
        rc = sc_err(err, "unsupported resize scales");
        break;
      }
      float sy = sc_at(sc, 2);
      float sx = sc_at(sc, 3);
      if (sy <= 0.0f || sx <= 0.0f) {
        rc = sc_err(err, "invalid resize scale");
        break;
      }
      o[0] = a[0];
      o[1] = a[1];
      o[2] = (int64_t)floor((double)a[2] * (double)sy);
      o[3] = (int64_t)floor((double)a[3] * (double)sx);
      s->n0 = (uint32_t)o[1];
      s->n1 = (uint32_t)a[2];
      s->n2 = (uint32_t)a[3];
      s->n3 = (uint32_t)o[2];
      s->n4 = (uint32_t)o[3];
      s->f0 = (float)a[2] / (float)o[2];
      s->f1 = (float)a[3] / (float)o[3];
      s->n5 = n->asymmetric ? 1u : 0u;
      s->n6 = n->nearest ? 1u : 0u;
      break;
    }
    case SC_OP_DEPTH_TO_SPACE: {
      if (!a || a[1] <= 0) {
        rc = sc_err(err, "depth to space on unknown shape");
        break;
      }
      int64_t b = n->block > 0 ? n->block : 1;
      if (a[1] % (b * b) != 0) {
        rc = sc_err(err, "unsupported depth to space");
        break;
      }
      o[0] = a[0];
      o[1] = a[1] / (b * b);
      o[2] = a[2] * b;
      o[3] = a[3] * b;
      s->n0 = (uint32_t)o[1];
      s->n1 = (uint32_t)a[2];
      s->n2 = (uint32_t)a[3];
      s->n3 = (uint32_t)b;
      s->kind = n->mode == 1 ? 1u : 0u;
      break;
    }
    default:
      rc = sc_err(err, "unsupported operator");
      break;
    }
    if (rc) {
      break;
    }

    for (int k = 0; k < n->n_in; k++) {
      if (n->in[k] < 0) {
        continue;
      }
      last[n->in[k]] = i;
    }
  }

  if (!rc) {
    for (int i = 0; i < nt; i++) {
      int64_t* d = &dims[i * 4];
      if (d[0] <= 0 || d[1] <= 0 || d[2] <= 0 || d[3] <= 0) {
        if (i == model->output) {
          rc = sc_err(err, "output shape unknown");
        }
        continue;
      }
      if (i == model->output) {
        last[i] = nn;
      }
    }
  }

  if (!rc) {
    size_t n_consts = 0;
    for (int i = 0; i < nn && !rc; i++) {
      const sc_node* n = &model->nodes[i];
      if (n->op != SC_OP_ADD && n->op != SC_OP_MUL) {
        continue;
      }
      for (int k = 0; k < 2; k++) {
        int id = n->in[k];
        if (id < 0 || !model->tensors[id].is_const) {
          continue;
        }
        if (placed[id]) {
          continue;
        }
        size_t count = model->tensors[id].count ? model->tensors[id].count : 1;
        off[id] = sc_arena_alloc(&arena, (uint32_t)count);
        placed[id] = 1;
        consts[n_consts].off = off[id];
        consts[n_consts].data = model->tensors[id].data;
        consts[n_consts].bytes = (uint32_t)(count * sizeof(float));
        n_consts++;
      }
    }
    if (!rc) {
      off[model->input] = sc_arena_alloc(&arena, (uint32_t)(span * span * 3u));
      placed[model->input] = 1;
      plan->input_off = off[model->input];

      for (int i = 0; i < nn && !rc; i++) {
        const sc_node* n = &model->nodes[i];
        sc_step* s = &steps[i];
        int oid = n->out;
        size_t count = (size_t)dims[oid * 4 + 0] * (size_t)dims[oid * 4 + 1] * (size_t)dims[oid * 4 + 2] *
                       (size_t)dims[oid * 4 + 3];
        if (count == 0 || count > 0x3fffffffull) {
          rc = sc_err(err, "unsupported tensor size");
          break;
        }
        if (!placed[oid]) {
          off[oid] = sc_arena_alloc(&arena, (uint32_t)count);
          placed[oid] = 1;
        }
        s->out_off = off[oid];
        s->in_off = n->in[0] >= 0 ? off[n->in[0]] : 0u;
        s->in2_off = (n->n_in > 1 && n->in[1] >= 0) ? off[n->in[1]] : 0u;
        if (n->op == SC_OP_CONCAT) {
          uint32_t cursor = off[oid];
          for (int k = 0; k < n->n_in && k < 16; k++) {
            const int64_t* p = &dims[n->in[k] * 4];
            s->copy[k].dst = cursor;
            s->copy[k].src = off[n->in[k]];
            s->copy[k].count = (uint32_t)(p[1] * p[2] * p[3]);
            cursor += s->copy[k].count;
          }
          s->copies = (uint32_t)(n->n_in > 16 ? 16 : n->n_in);
        }
        for (int k = 0; k < n->n_in; k++) {
          int id = n->in[k];
          if (id < 0 || id == model->output) {
            continue;
          }
          if (last[id] != i || !placed[id] || model->tensors[id].is_const) {
            continue;
          }
          size_t fc = (size_t)dims[id * 4 + 0] * (size_t)dims[id * 4 + 1] * (size_t)dims[id * 4 + 2] *
                      (size_t)dims[id * 4 + 3];
          sc_arena_release(&arena, off[id], (uint32_t)fc);
          placed[id] = 0;
        }
      }
    }
    if (!rc) {
      for (int i = 0; i < nn && !rc; i++) {
        const sc_node* n = &model->nodes[i];
        size_t count = (size_t)dims[n->out * 4 + 0] * (size_t)dims[n->out * 4 + 1] * (size_t)dims[n->out * 4 + 2] *
                       (size_t)dims[n->out * 4 + 3];
        if ((uint64_t)steps[i].out_off + count > arena.peak) {
          rc = sc_err(err, "internal layout error");
          break;
        }
      }
      plan->out_w = (uint32_t)dims[model->output * 4 + 3];
      plan->out_h = (uint32_t)dims[model->output * 4 + 2];
      if (span == 0 || plan->out_w != plan->out_h || plan->out_w % span != 0) {
        rc = sc_err(err, "unsupported output scale");
      } else {
        plan->steps = steps;
        plan->n_steps = (uint32_t)nn;
        plan->output_off = off[model->output];
        plan->arena_floats = (uint32_t)arena.peak;
        plan->arena_bytes = arena.peak * sizeof(float);
        plan->in_w = span;
        plan->in_h = span;
        plan->out_scale = plan->out_w / span;
        plan->consts = consts;
        plan->n_consts = (uint32_t)n_consts;
        steps = NULL;
        consts = NULL;
      }
    }
  }

  free(steps);
  free(consts);
  free(dims);
  free(off);
  free(placed);
  free(last);
  sc_arena_free(&arena);
  return rc;
}

void sc_plan_free(sc_plan* plan) {
  free(plan->steps);
  free(plan->consts);
  memset(plan, 0, sizeof(*plan));
}
