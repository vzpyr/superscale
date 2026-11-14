#include <onnxruntime_c_api.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
  const OrtApi* api;
  OrtSession* session;
  OrtMemoryInfo* meminfo;
  OrtAllocator* allocator;
  char* input_name;
  char* output_name;
  float* input_buf;
  size_t input_cap;
  int scale;
} superscale_session;

static const OrtApi* g_api = NULL;
static OrtEnv* g_env = NULL;

static char* copy_string(const char* s) {
  if (!s) s = "unknown error";
  size_t n = strlen(s);
  char* out = (char*)malloc(n + 1);
  if (out) memcpy(out, s, n + 1);
  return out;
}

static int fail(const OrtApi* api, OrtStatus* status, char** err) {
  if (err) {
    *err = copy_string(status ? api->GetErrorMessage(status) : "unknown error");
  }
  if (status) api->ReleaseStatus(status);
  return 1;
}

static int fail_text(const char* msg, char** err) {
  if (err) *err = copy_string(msg);
  return 1;
}

const char* superscale_runtime_version(void) { return OrtGetApiBase()->GetVersionString(); }

int superscale_init(char** err) {
  if (g_env) return 0;
  const OrtApiBase* base = OrtGetApiBase();
  g_api = base->GetApi(ORT_API_VERSION);
  if (!g_api) return fail_text("onnx runtime api unavailable", err);
  OrtStatus* status = g_api->CreateEnv(ORT_LOGGING_LEVEL_ERROR, "superscale", &g_env);
  if (status) return fail(g_api, status, err);
  return 0;
}

superscale_session* superscale_open(const char* model_path, int threads, char** err) {
  if (!g_env && superscale_init(err) != 0) return NULL;
  const OrtApi* api = g_api;

  OrtSessionOptions* options = NULL;
  OrtStatus* status = api->CreateSessionOptions(&options);
  if (status) {
    fail(api, status, err);
    return NULL;
  }

  if (threads > 0) status = api->SetIntraOpNumThreads(options, threads);
  if (!status) status = api->SetSessionExecutionMode(options, ORT_SEQUENTIAL);
  if (!status) status = api->SetSessionGraphOptimizationLevel(options, ORT_ENABLE_ALL);
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

  superscale_session* s = (superscale_session*)calloc(1, sizeof(superscale_session));
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
  s->scale = 4;

  status = api->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &s->meminfo);
  if (status) {
    superscale_session* tmp = s;
    if (tmp->input_name) allocator->Free(allocator, tmp->input_name);
    if (tmp->output_name) allocator->Free(allocator, tmp->output_name);
    api->ReleaseSession(session);
    free(tmp);
    fail(api, status, err);
    return NULL;
  }

  return s;
}

int superscale_run(superscale_session* s, const uint8_t* rgba, int w, int h, uint8_t* out, size_t out_len,
               int* out_w, int* out_h, char** err) {
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
  for (size_t i = 0; i < plane; ++i) {
    const uint8_t* p = rgba + i * 4;
    src[i] = (float)p[0] / 255.0f;
    src[plane + i] = (float)p[1] / 255.0f;
    src[plane * 2 + i] = (float)p[2] / 255.0f;
  }

  int64_t shape[4] = {1, 3, h, w};
  OrtValue* input = NULL;
  OrtStatus* status = api->CreateTensorWithDataAsOrtValue(
      s->meminfo, src, need * sizeof(float), shape, 4, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &input);
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
  if (pixels == 0 || pixels * 4 > out_len) {
    api->ReleaseValue(output);
    return fail_text("output buffer too small", err);
  }

  float* dst = NULL;
  status = api->GetTensorMutableData(output, (void**)&dst);
  if (status) {
    api->ReleaseValue(output);
    return fail(api, status, err);
  }

  for (size_t i = 0; i < pixels; ++i) {
    float r = dst[i] * 255.0f;
    float g = dst[pixels + i] * 255.0f;
    float b = dst[pixels * 2 + i] * 255.0f;
    out[i * 4] = (uint8_t)(r < 0 ? 0 : (r > 255 ? 255 : r + 0.5f));
    out[i * 4 + 1] = (uint8_t)(g < 0 ? 0 : (g > 255 ? 255 : g + 0.5f));
    out[i * 4 + 2] = (uint8_t)(b < 0 ? 0 : (b > 255 ? 255 : b + 0.5f));
    out[i * 4 + 3] = 255;
  }

  api->ReleaseValue(output);

  if (out_w) *out_w = (int)ow;
  if (out_h) *out_h = (int)oh;
  s->scale = (int)(ow / w);
  return 0;
}

int superscale_scale(superscale_session* s) { return s ? s->scale : 0; }

void superscale_close(superscale_session* s) {
  if (!s) return;
  const OrtApi* api = s->api;
  if (s->input_name) s->allocator->Free(s->allocator, s->input_name);
  if (s->output_name) s->allocator->Free(s->allocator, s->output_name);
  if (s->meminfo) api->ReleaseMemoryInfo(s->meminfo);
  if (s->session) api->ReleaseSession(s->session);
  free(s->input_buf);
  free(s);
}

void superscale_free_string(char* s) { free(s); }
