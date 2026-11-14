#ifndef SUPERSCALE_ENGINE_H
#define SUPERSCALE_ENGINE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
  SC_BACKEND_CPU = 0,
  SC_BACKEND_GPU = 1,
} sc_backend;

typedef struct sc_engine sc_engine;

typedef struct {
  int (*open)(const char* model_path, int threads, void** handle, int* scale, int* max_span, char** err);
  int (*run)(void* handle, const uint8_t* rgba, int width, int height, uint8_t* out, size_t out_len, int* out_w,
             int* out_h, char** err);
  void (*close)(void* handle);
} sc_backend_ops;

sc_engine* sc_engine_open(const char* model_path, sc_backend backend, int threads, int* scale, int* max_span,
                          char** err);
int sc_engine_run(sc_engine* engine, const uint8_t* rgba, int width, int height, uint8_t* out, size_t out_len,
                  int* out_w, int* out_h, char** err);
void sc_engine_close(sc_engine* engine);
void sc_engine_free_string(char* text);

#ifdef __cplusplus
}
#endif

#endif
