#ifndef SUPERSCALE_CPU_H
#define SUPERSCALE_CPU_H

#include <stddef.h>
#include <stdint.h>

#include "engine.h"

typedef struct sc_cpu_session sc_cpu_session;

sc_cpu_session* sc_cpu_create(const char* model_path, int threads, char** err);
int sc_cpu_execute(sc_cpu_session* s, const uint8_t* rgba, int w, int h, uint8_t* out, size_t out_len, int* out_w,
                   int* out_h, char** err);
int sc_cpu_scale(const sc_cpu_session* s);
void sc_cpu_destroy(sc_cpu_session* s);

const sc_backend_ops* sc_cpu_ops(void);

#endif
