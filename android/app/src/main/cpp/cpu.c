#include <onnxruntime_c_api.h>
#include <stdint.h>
#include <stdlib.h>

#include "cpu.h"
#include "err.h"
#include "image.h"

struct sc_cpu_session {
  const OrtApi* api;
  OrtSession* session;
  OrtMemoryInfo* meminfo;
  OrtAllocator* allocator;
  char* input_name;
  char* output_name;
  float* input_buf;
  size_t input_cap;
  int scale;
};

static const OrtApi* g_api = NULL;
static OrtEnv* g_env = NULL;

static int fail(const OrtApi* api, OrtStatus* status, char** err) {
  int rc = sc_err(err, "%s", status ? api->GetErrorMessage(status) : "unknown error");
  if (status) api->ReleaseStatus(status);
  return rc;
}

static int fail_text(const char* msg, char** err) {
  return sc_err(err, "%s", msg);
}

static int sc_cpu_init(char** err) {
  if (g_env) return 0;
  const OrtApiBase* base = OrtGetApiBase();
  g_api = base->GetApi(ORT_API_VERSION);
  if (!g_api) return fail_text("onnx runtime api unavailable", err);
  OrtStatus* status = g_api->CreateEnv(ORT_LOGGING_LEVEL_ERROR, "superscale", &g_env);
  if (status) return fail(g_api, status, err);
  return 0;
}

static int sc_cpu_probe(sc_cpu_session* s, char** err) {
  enum { kProbe = 8, kMaxScale = 16 };
  uint8_t input[kProbe * kProbe * 4] = {0};
  uint8_t output[kProbe * kProbe * 4 * kMaxScale * kMaxScale];
  int w = 0;
  int h = 0;
  if (sc_cpu_execute(s, input, kProbe, kProbe, output, sizeof(output), &w, &h, err)) return 1;
  if (w != h || w % kProbe != 0) return fail_text("unsupported model output shape", err);
  s->scale = w / kProbe;
  if (s->scale < 1 || s->scale > kMaxScale) return fail_text("unsupported model scale", err);
  return 0;
}

sc_cpu_session* sc_cpu_create(const char* model_path, int threads, char** err) {
  if (!g_env && sc_cpu_init(err) != 0) return NULL;
  const OrtApi* api = g_api;

  OrtSessionOptions* options = NULL;
  OrtStatus* status = api->CreateSessionOptions(&options);
  if (status) {
    fail(api, status, err);
    return NULL;
  }

  status = api->SetIntraOpNumThreads(options, threads > 0 ? threads : 1);
  if (!status) status = api->SetSessionGraphOptimizationLevel(options, ORT_ENABLE_ALL);
  if (!status) status = api->DisableMemPattern(options);
  if (status) {
    api->ReleaseSessionOptions(options);
    fail(api, status, err);
    return NULL;
  }

  OrtSession* session = NULL;
  status = api->CreateSession(g_env, model_path, options, &session);
  api->ReleaseSessionOptions(options);
  if (status) {
    fail(api, status, err);
    return NULL;
  }

  OrtAllocator* allocator = NULL;
  status = api->GetAllocatorWithDefaultOptions(&allocator);
  if (status) {
    api->ReleaseSession(session);
    fail(api, status, err);
    return NULL;
  }

  char* input_name = NULL;
  char* output_name = NULL;
  status = api->SessionGetInputName(session, 0, allocator, &input_name);
  if (!status) status = api->SessionGetOutputName(session, 0, allocator, &output_name);
  if (status) {
    if (input_name) allocator->Free(allocator, input_name);
    api->ReleaseSession(session);
    fail(api, status, err);
    return NULL;
  }

  sc_cpu_session* s = (sc_cpu_session*)calloc(1, sizeof(sc_cpu_session));
  if (!s) {
    allocator->Free(allocator, input_name);
    allocator->Free(allocator, output_name);
    api->ReleaseSession(session);
    fail_text("out of memory", err);
    return NULL;
  }
  s->api = api;
  s->session = session;
  s->allocator = allocator;
  s->input_name = input_name;
  s->output_name = output_name;
  s->scale = 0;

  status = api->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &s->meminfo);
  if (status) {
    sc_cpu_session* tmp = s;
    if (tmp->input_name) allocator->Free(allocator, tmp->input_name);
    if (tmp->output_name) allocator->Free(allocator, tmp->output_name);
    api->ReleaseSession(session);
    free(tmp);
    fail(api, status, err);
    return NULL;
  }

  if (sc_cpu_probe(s, err)) {
    sc_cpu_destroy(s);
    return NULL;
  }
  return s;
}

int sc_cpu_execute(sc_cpu_session* s, const uint8_t* rgba, int w, int h, uint8_t* out, size_t out_len, int* out_w,
                   int* out_h, char** err) {
  if (!s || !rgba || !out || w <= 0 || h <= 0) return fail_text("invalid arguments", err);
  const OrtApi* api = s->api;

  size_t plane = (size_t)w * (size_t)h;
  size_t need = plane * 3;
  if (need > s->input_cap) {
    float* grown = (float*)realloc(s->input_buf, need * sizeof(float));
    if (!grown) return fail_text("out of memory", err);
    s->input_buf = grown;
    s->input_cap = need;
  }

  float* src = s->input_buf;
  sc_rgba_to_planar(rgba, plane, src);

  int64_t shape[4] = {1, 3, h, w};
  OrtValue* input = NULL;
  OrtStatus* status = api->CreateTensorWithDataAsOrtValue(s->meminfo, src, need * sizeof(float), shape, 4,
                                                          ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &input);
  if (status) return fail(api, status, err);

  const char* input_names[1] = {s->input_name};
  const char* output_names[1] = {s->output_name};
  const OrtValue* inputs[1] = {input};
  OrtValue* outputs[1] = {NULL};

  status = api->Run(s->session, NULL, input_names, inputs, 1, output_names, 1, outputs);
  api->ReleaseValue(input);
  if (status) return fail(api, status, err);

  OrtValue* output = outputs[0];
  OrtTensorTypeAndShapeInfo* info = NULL;
  status = api->GetTensorTypeAndShape(output, &info);
  if (status) {
    api->ReleaseValue(output);
    return fail(api, status, err);
  }

  size_t dims_count = 0;
  int64_t dims[8];
  status = api->GetDimensionsCount(info, &dims_count);
  if (status || dims_count != 4) {
    if (status) api->ReleaseStatus(status);
    api->ReleaseTensorTypeAndShapeInfo(info);
    api->ReleaseValue(output);
    return fail_text("unexpected output rank", err);
  }
  status = api->GetDimensions(info, dims, dims_count);
  if (status) {
    api->ReleaseTensorTypeAndShapeInfo(info);
    api->ReleaseValue(output);
    return fail(api, status, err);
  }
  api->ReleaseTensorTypeAndShapeInfo(info);

  int64_t oh = dims[2];
  int64_t ow = dims[3];
  size_t pixels = (size_t)ow * (size_t)oh;
  if (dims[0] != 1 || dims[1] != 3) {
    api->ReleaseValue(output);
    return fail_text("unexpected output shape", err);
  }
  if (pixels == 0 || pixels * 4 > out_len) {
    api->ReleaseValue(output);
    return fail_text("output buffer too small", err);
  }
  if (ow % w != 0 || oh % h != 0) {
    api->ReleaseValue(output);
    return fail_text("unsupported output scale", err);
  }

  float* dst = NULL;
  status = api->GetTensorMutableData(output, (void**)&dst);
  if (status) {
    api->ReleaseValue(output);
    return fail(api, status, err);
  }

  sc_planar_to_rgba(dst, pixels, out);

  api->ReleaseValue(output);

  if (out_w) *out_w = (int)ow;
  if (out_h) *out_h = (int)oh;
  s->scale = (int)(ow / w);
  return 0;
}

int sc_cpu_scale(const sc_cpu_session* s) {
  return s ? s->scale : 0;
}

void sc_cpu_destroy(sc_cpu_session* s) {
  if (!s) return;
  const OrtApi* api = s->api;
  if (s->input_name) s->allocator->Free(s->allocator, s->input_name);
  if (s->output_name) s->allocator->Free(s->allocator, s->output_name);
  if (s->meminfo) api->ReleaseMemoryInfo(s->meminfo);
  if (s->session) api->ReleaseSession(s->session);
  free(s->input_buf);
  free(s);
}

static int cpu_open(const char* model_path, int threads, void** handle, int* scale, int* max_span, char** err) {
  sc_cpu_session* session = sc_cpu_create(model_path, threads, err);
  if (!session) {
    return 1;
  }
  *handle = session;
  if (scale) {
    *scale = sc_cpu_scale(session);
  }
  if (max_span) {
    *max_span = 0;
  }
  return 0;
}

static int cpu_run(void* handle, const uint8_t* rgba, int width, int height, uint8_t* out, size_t out_len, int* out_w,
                   int* out_h, char** err) {
  return sc_cpu_execute((sc_cpu_session*)handle, rgba, width, height, out, out_len, out_w, out_h, err);
}

static void cpu_close(void* handle) {
  sc_cpu_destroy((sc_cpu_session*)handle);
}

const sc_backend_ops* sc_cpu_ops(void) {
  static const sc_backend_ops ops = {
      .open = cpu_open,
      .run = cpu_run,
      .close = cpu_close,
  };
  return &ops;
}
