#include "engine.h"

#include <stdlib.h>

#include "cpu.h"
#include "err.h"
#include "gpu.h"

struct sc_engine {
  const sc_backend_ops* ops;
  void* handle;
  int scale;
};

sc_engine* sc_engine_open(const char* model_path, sc_backend backend, int threads, int* scale, int* max_span,
                          char** err) {
  const sc_backend_ops* ops = backend == SC_BACKEND_GPU ? sc_gpu_ops() : sc_cpu_ops();
  if (!ops) {
    sc_err(err, "backend unavailable");
    return NULL;
  }
  void* handle = NULL;
  int resolved = 1;
  if (ops->open(model_path, threads, &handle, &resolved, max_span, err)) {
    return NULL;
  }
  sc_engine* engine = (sc_engine*)calloc(1, sizeof(sc_engine));
  if (!engine) {
    ops->close(handle);
    sc_err(err, "out of memory");
    return NULL;
  }
  engine->ops = ops;
  engine->handle = handle;
  engine->scale = resolved;
  if (scale) {
    *scale = resolved;
  }
  return engine;
}

int sc_engine_run(sc_engine* engine, const uint8_t* rgba, int width, int height, uint8_t* out, size_t out_len,
                  int* out_w, int* out_h, char** err) {
  if (!engine) {
    return sc_err(err, "no engine");
  }
  return engine->ops->run(engine->handle, rgba, width, height, out, out_len, out_w, out_h, err);
}

void sc_engine_close(sc_engine* engine) {
  if (!engine) {
    return;
  }
  engine->ops->close(engine->handle);
  free(engine);
}

void sc_engine_free_string(char* text) {
  free(text);
}
