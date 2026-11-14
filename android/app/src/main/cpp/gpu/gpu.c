#include "gpu.h"

#include <stdlib.h>
#include <string.h>
#include <vulkan/vulkan.h>

#include "../err.h"
#include "../image.h"
#include "plan.h"
#include "spirv.h"

typedef struct sc_vk_session sc_vk_session;

typedef struct {
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
} sc_pc;

_Static_assert(sizeof(sc_pc) == 100, "push constant layout");

#define SC_TILE 32u
#define SC_COV 16u
#define SC_CICH 4u
#define SC_SPAN_CANDIDATES 6
#define SC_ARENA_BUDGET (128u << 20)

typedef enum {
  SC_PIPE_CONV = 0,
  SC_PIPE_UNARY = 1,
  SC_PIPE_BINARY = 2,
  SC_PIPE_RESIZE = 3,
  SC_PIPE_D2S = 4,
  SC_PIPE_COUNT = 5,
} sc_pipe_id;

typedef struct {
  int ready;
  VkInstance instance;
  VkPhysicalDevice physical;
  VkDevice device;
  VkQueue queue;
  uint32_t family;
  VkCommandPool pool;
  VkDescriptorSetLayout set_layout;
  VkPipelineLayout layout;
  VkPipeline pipes[SC_PIPE_COUNT];
  VkPhysicalDeviceMemoryProperties mem;
  VkDeviceSize max_range;
} sc_gpu_state;

static sc_gpu_state g_gpu;

typedef struct {
  uint32_t span;
  sc_plan plan;
  VkBuffer arena;
  VkDeviceMemory arena_mem;
  VkBuffer stage;
  VkDeviceMemory stage_mem;
  void* stage_map;
  VkDeviceSize stage_size;
  VkDeviceSize in_bytes;
  VkDeviceSize const_bytes;
  VkDeviceSize out_bytes;
  VkCommandBuffer cmd;
  VkDescriptorSet set;
  VkDescriptorPool desc_pool;
  VkFence fence;
} sc_prepared;

struct sc_vk_session {
  int max_span;
  sc_model model;
  float* weights;
  size_t weight_count;
  VkBuffer wbuf;
  VkDeviceMemory wbuf_mem;
  sc_prepared plans[4];
  int n_plans;
};

static uint32_t sc_mem_type(uint32_t bits, VkMemoryPropertyFlags want) {
  for (uint32_t i = 0; i < g_gpu.mem.memoryTypeCount; i++) {
    if (!(bits & (1u << i))) {
      continue;
    }
    if ((g_gpu.mem.memoryTypes[i].propertyFlags & want) == want) {
      return i;
    }
  }
  return UINT32_MAX;
}

static int sc_buffer_create(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags props, VkBuffer* buf,
                            VkDeviceMemory* mem, char** err) {
  VkBufferCreateInfo bci = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
  bci.size = size ? size : 4;
  bci.usage = usage;
  bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  if (vkCreateBuffer(g_gpu.device, &bci, NULL, buf) != VK_SUCCESS) {
    return sc_err(err, "could not create buffer");
  }
  VkMemoryRequirements req;
  vkGetBufferMemoryRequirements(g_gpu.device, *buf, &req);
  const uint32_t type = sc_mem_type(req.memoryTypeBits, props);
  if (type == UINT32_MAX) {
    vkDestroyBuffer(g_gpu.device, *buf, NULL);
    *buf = VK_NULL_HANDLE;
    return sc_err(err, "no suitable memory type");
  }
  VkMemoryAllocateInfo mai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
  mai.allocationSize = req.size;
  mai.memoryTypeIndex = type;
  if (vkAllocateMemory(g_gpu.device, &mai, NULL, mem) != VK_SUCCESS) {
    vkDestroyBuffer(g_gpu.device, *buf, NULL);
    *buf = VK_NULL_HANDLE;
    return sc_err(err, "could not allocate memory");
  }
  vkBindBufferMemory(g_gpu.device, *buf, *mem, 0);
  return 0;
}

static void sc_buffer_destroy(VkBuffer* buf, VkDeviceMemory* mem) {
  if (*buf) {
    vkDestroyBuffer(g_gpu.device, *buf, NULL);
    *buf = VK_NULL_HANDLE;
  }
  if (*mem) {
    vkFreeMemory(g_gpu.device, *mem, NULL);
    *mem = VK_NULL_HANDLE;
  }
}

static int sc_pipeline_create(const uint32_t* code, uint32_t words, VkPipeline* out, char** err) {
  VkShaderModuleCreateInfo smci = {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
  smci.codeSize = (size_t)words * 4u;
  smci.pCode = code;
  VkShaderModule module = VK_NULL_HANDLE;
  if (vkCreateShaderModule(g_gpu.device, &smci, NULL, &module) != VK_SUCCESS) {
    return sc_err(err, "could not create shader module");
  }
  VkComputePipelineCreateInfo ci = {VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
  ci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  ci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
  ci.stage.module = module;
  ci.stage.pName = "main";
  ci.layout = g_gpu.layout;
  const VkResult r = vkCreateComputePipelines(g_gpu.device, VK_NULL_HANDLE, 1, &ci, NULL, out);
  vkDestroyShaderModule(g_gpu.device, module, NULL);
  if (r != VK_SUCCESS) {
    return sc_err(err, "could not create compute pipeline");
  }
  return 0;
}

static int sc_gpu_init(char** err) {
  if (g_gpu.ready) {
    return 0;
  }
  if (g_gpu.instance) {
    return sc_err(err, "vulkan unavailable");
  }
  VkApplicationInfo app = {VK_STRUCTURE_TYPE_APPLICATION_INFO};
  app.apiVersion = VK_API_VERSION_1_1;
  VkInstanceCreateInfo ici = {VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
  ici.pApplicationInfo = &app;
  if (vkCreateInstance(&ici, NULL, &g_gpu.instance) != VK_SUCCESS) {
    return sc_err(err, "vulkan unavailable");
  }
  uint32_t count = 0;
  if (vkEnumeratePhysicalDevices(g_gpu.instance, &count, NULL) != VK_SUCCESS || count == 0) {
    return sc_err(err, "no vulkan device");
  }
  VkPhysicalDevice* devices = (VkPhysicalDevice*)calloc(count, sizeof(VkPhysicalDevice));
  if (!devices) {
    return sc_err(err, "out of memory");
  }
  vkEnumeratePhysicalDevices(g_gpu.instance, &count, devices);
  VkPhysicalDevice chosen = VK_NULL_HANDLE;
  uint32_t chosen_family = UINT32_MAX;
  int best = -1;
  for (uint32_t i = 0; i < count; i++) {
    uint32_t families = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(devices[i], &families, NULL);
    VkQueueFamilyProperties* queue = (VkQueueFamilyProperties*)calloc(families, sizeof(VkQueueFamilyProperties));
    if (!queue) {
      continue;
    }
    vkGetPhysicalDeviceQueueFamilyProperties(devices[i], &families, queue);
    uint32_t family = UINT32_MAX;
    for (uint32_t k = 0; k < families; k++) {
      if (queue[k].queueFlags & VK_QUEUE_COMPUTE_BIT) {
        family = k;
        break;
      }
    }
    free(queue);
    if (family == UINT32_MAX) {
      continue;
    }
    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(devices[i], &props);
    int score = 0;
    if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
      score = 3;
    } else if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU) {
      score = 2;
    } else if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_CPU) {
      score = 1;
    }
    if (score > best) {
      best = score;
      chosen = devices[i];
      chosen_family = family;
      g_gpu.max_range = props.limits.maxStorageBufferRange;
    }
  }
  free(devices);
  if (!chosen) {
    return sc_err(err, "no gpu device");
  }
  g_gpu.physical = chosen;
  g_gpu.family = chosen_family;

  float priority = 1.0f;
  VkDeviceQueueCreateInfo qci = {VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
  qci.queueFamilyIndex = g_gpu.family;
  qci.queueCount = 1;
  qci.pQueuePriorities = &priority;
  VkDeviceCreateInfo dci = {VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
  dci.queueCreateInfoCount = 1;
  dci.pQueueCreateInfos = &qci;
  if (vkCreateDevice(chosen, &dci, NULL, &g_gpu.device) != VK_SUCCESS) {
    return sc_err(err, "could not create device");
  }
  vkGetDeviceQueue(g_gpu.device, g_gpu.family, 0, &g_gpu.queue);
  vkGetPhysicalDeviceMemoryProperties(chosen, &g_gpu.mem);

  VkCommandPoolCreateInfo pci = {VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
  pci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
  pci.queueFamilyIndex = g_gpu.family;
  if (vkCreateCommandPool(g_gpu.device, &pci, NULL, &g_gpu.pool) != VK_SUCCESS) {
    return sc_err(err, "could not create command pool");
  }

  VkDescriptorSetLayoutBinding bindings[2];
  memset(bindings, 0, sizeof(bindings));
  for (int i = 0; i < 2; i++) {
    bindings[i].binding = (uint32_t)i;
    bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[i].descriptorCount = 1;
    bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  }
  VkDescriptorSetLayoutCreateInfo lci = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
  lci.bindingCount = 2;
  lci.pBindings = bindings;
  if (vkCreateDescriptorSetLayout(g_gpu.device, &lci, NULL, &g_gpu.set_layout) != VK_SUCCESS) {
    return sc_err(err, "could not create descriptor layout");
  }
  VkPushConstantRange range;
  range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  range.offset = 0;
  range.size = sizeof(sc_pc);
  VkPipelineLayoutCreateInfo plci = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
  plci.setLayoutCount = 1;
  plci.pSetLayouts = &g_gpu.set_layout;
  plci.pushConstantRangeCount = 1;
  plci.pPushConstantRanges = &range;
  if (vkCreatePipelineLayout(g_gpu.device, &plci, NULL, &g_gpu.layout) != VK_SUCCESS) {
    return sc_err(err, "could not create pipeline layout");
  }

  if (sc_pipeline_create(sc_spv_conv, sc_spv_conv_words, &g_gpu.pipes[SC_PIPE_CONV], err) ||
      sc_pipeline_create(sc_spv_unary, sc_spv_unary_words, &g_gpu.pipes[SC_PIPE_UNARY], err) ||
      sc_pipeline_create(sc_spv_binary, sc_spv_binary_words, &g_gpu.pipes[SC_PIPE_BINARY], err) ||
      sc_pipeline_create(sc_spv_resize, sc_spv_resize_words, &g_gpu.pipes[SC_PIPE_RESIZE], err) ||
      sc_pipeline_create(sc_spv_d2s, sc_spv_d2s_words, &g_gpu.pipes[SC_PIPE_D2S], err)) {
    return 1;
  }
  g_gpu.ready = 1;
  return 0;
}

static void sc_barrier(VkCommandBuffer cmd, VkPipelineStageFlags src, VkAccessFlags src_acc, VkPipelineStageFlags dst,
                       VkAccessFlags dst_acc) {
  VkMemoryBarrier mb = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
  mb.srcAccessMask = src_acc;
  mb.dstAccessMask = dst_acc;
  vkCmdPipelineBarrier(cmd, src, dst, 0, 1, &mb, 0, NULL, 0, NULL);
}

static void sc_fill_pc(sc_pc* pc, const sc_step* st) {
  memset(pc, 0, sizeof(*pc));
  pc->out_off = st->out_off;
  pc->in_off = st->in_off;
  pc->in2_off = st->in2_off;
  pc->w_off = st->w_off;
  pc->b_off = st->b_off;
  pc->n0 = st->n0;
  pc->n1 = st->n1;
  pc->n2 = st->n2;
  pc->n3 = st->n3;
  pc->n4 = st->n4;
  pc->n5 = st->n5;
  pc->n6 = st->n6;
  pc->n7 = st->n7;
  for (int i = 0; i < 4; i++) {
    pc->s0[i] = st->s0[i];
    pc->s1[i] = st->s1[i];
  }
  pc->f0 = st->f0;
  pc->f1 = st->f1;
  pc->f2 = st->f2;
  pc->kind = st->kind;
}

static VkPipeline sc_pipe_for(const sc_step* st, char** err) {
  switch (st->op) {
  case SC_OP_CONV:
    return g_gpu.pipes[SC_PIPE_CONV];
  case SC_OP_LEAKY_RELU:
  case SC_OP_PRELU:
  case SC_OP_CLIP:
    return g_gpu.pipes[SC_PIPE_UNARY];
  case SC_OP_ADD:
  case SC_OP_MUL:
    return g_gpu.pipes[SC_PIPE_BINARY];
  case SC_OP_RESIZE:
    return g_gpu.pipes[SC_PIPE_RESIZE];
  case SC_OP_DEPTH_TO_SPACE:
    return g_gpu.pipes[SC_PIPE_D2S];
  default:
    sc_err(err, "unsupported operator");
    return VK_NULL_HANDLE;
  }
}

static void sc_step_groups(const sc_step* st, uint32_t* gx, uint32_t* gy, uint32_t* gz) {
  *gx = 1;
  *gy = 1;
  *gz = 1;
  if (st->op == SC_OP_CONV) {
    *gx = (st->n5 + SC_TILE - 1u) / SC_TILE;
    *gy = (st->n4 + SC_TILE - 1u) / SC_TILE;
    *gz = (st->n0 + SC_COV - 1u) / SC_COV;
  } else if (st->op == SC_OP_RESIZE || st->op == SC_OP_DEPTH_TO_SPACE) {
    const uint32_t ho = st->op == SC_OP_RESIZE ? st->n3 : st->n1 * st->n3;
    const uint32_t wo = st->op == SC_OP_RESIZE ? st->n4 : st->n2 * st->n3;
    *gx = (wo + 7u) / 8u;
    *gy = (ho + 7u) / 8u;
    *gz = (st->n0 + 3u) / 4u;
  } else {
    const uint32_t total = st->n0 * st->n1 * st->n2;
    *gx = (total + 255u) / 256u;
  }
}

static int sc_record(sc_prepared* p, char** err) {
  VkCommandBufferBeginInfo bi = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
  if (vkBeginCommandBuffer(p->cmd, &bi) != VK_SUCCESS) {
    return sc_err(err, "could not begin recording");
  }
  VkBufferCopy in_copy;
  in_copy.srcOffset = 0;
  in_copy.dstOffset = (VkDeviceSize)p->plan.input_off * 4u;
  in_copy.size = (VkDeviceSize)p->span * p->span * 3u * 4u;
  vkCmdCopyBuffer(p->cmd, p->stage, p->arena, 1, &in_copy);
  VkDeviceSize stage_off = p->in_bytes;
  for (uint32_t i = 0; i < p->plan.n_consts; i++) {
    VkBufferCopy c;
    c.srcOffset = stage_off;
    c.dstOffset = (VkDeviceSize)p->plan.consts[i].off * 4u;
    c.size = p->plan.consts[i].bytes;
    vkCmdCopyBuffer(p->cmd, p->stage, p->arena, 1, &c);
    stage_off += c.size;
  }
  sc_barrier(p->cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);

  for (uint32_t i = 0; i < p->plan.n_steps; i++) {
    const sc_step* st = &p->plan.steps[i];
    if (st->op == SC_OP_CONCAT) {
      for (uint32_t k = 0; k < st->copies; k++) {
        VkBufferCopy c;
        c.srcOffset = (VkDeviceSize)st->copy[k].src * 4u;
        c.dstOffset = (VkDeviceSize)st->copy[k].dst * 4u;
        c.size = (VkDeviceSize)st->copy[k].count * 4u;
        vkCmdCopyBuffer(p->cmd, p->arena, p->arena, 1, &c);
      }
      sc_barrier(p->cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
                 VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
      continue;
    }
    VkPipeline pipe = sc_pipe_for(st, err);
    if (pipe == VK_NULL_HANDLE) {
      return 1;
    }
    sc_pc pc;
    sc_fill_pc(&pc, st);
    vkCmdBindPipeline(p->cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
    vkCmdBindDescriptorSets(p->cmd, VK_PIPELINE_BIND_POINT_COMPUTE, g_gpu.layout, 0, 1, &p->set, 0, NULL);
    vkCmdPushConstants(p->cmd, g_gpu.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    uint32_t gx, gy, gz;
    sc_step_groups(st, &gx, &gy, &gz);
    vkCmdDispatch(p->cmd, gx, gy, gz);
    sc_barrier(p->cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
               VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
               VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT);
  }

  VkBufferCopy out_copy;
  out_copy.srcOffset = (VkDeviceSize)p->plan.output_off * 4u;
  out_copy.dstOffset = p->in_bytes + p->const_bytes;
  out_copy.size = (VkDeviceSize)p->plan.out_w * p->plan.out_h * 3u * 4u;
  vkCmdCopyBuffer(p->cmd, p->arena, p->stage, 1, &out_copy);

  if (vkEndCommandBuffer(p->cmd) != VK_SUCCESS) {
    return sc_err(err, "could not finish recording");
  }
  return 0;
}

static void sc_prepared_destroy(sc_prepared* p) {
  if (p->fence) {
    vkDestroyFence(g_gpu.device, p->fence, NULL);
    p->fence = VK_NULL_HANDLE;
  }
  if (p->cmd) {
    vkFreeCommandBuffers(g_gpu.device, g_gpu.pool, 1, &p->cmd);
    p->cmd = VK_NULL_HANDLE;
  }
  if (p->desc_pool) {
    vkDestroyDescriptorPool(g_gpu.device, p->desc_pool, NULL);
    p->desc_pool = VK_NULL_HANDLE;
  }
  if (p->stage_map) {
    vkUnmapMemory(g_gpu.device, p->stage_mem);
    p->stage_map = NULL;
  }
  sc_buffer_destroy(&p->stage, &p->stage_mem);
  sc_buffer_destroy(&p->arena, &p->arena_mem);
  sc_plan_free(&p->plan);
}

static int sc_prepared_create(sc_vk_session* s, uint32_t span, sc_prepared* p, char** err) {
  memset(p, 0, sizeof(*p));
  p->span = span;
  if (sc_plan_build(&s->model, span, &p->plan, err)) {
    return 1;
  }
  p->in_bytes = (VkDeviceSize)span * span * 3u * 4u;
  for (uint32_t i = 0; i < p->plan.n_consts; i++) {
    p->const_bytes += p->plan.consts[i].bytes;
  }
  p->out_bytes = (VkDeviceSize)p->plan.out_w * p->plan.out_h * 3u * 4u;
  p->stage_size = p->in_bytes + p->const_bytes + p->out_bytes;
  if (p->stage_size == 0 || p->plan.arena_bytes == 0 || p->plan.arena_bytes > g_gpu.max_range ||
      p->plan.arena_bytes > SC_ARENA_BUDGET) {
    return sc_err(err, "image tile too large for this device");
  }
  if (sc_buffer_create(p->plan.arena_bytes,
                       VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                           VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                       VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, &p->arena, &p->arena_mem, err)) {
    return 1;
  }
  const VkMemoryPropertyFlags host = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
  if (sc_buffer_create(p->stage_size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, host,
                       &p->stage, &p->stage_mem, err)) {
    sc_prepared_destroy(p);
    return 1;
  }
  if (vkMapMemory(g_gpu.device, p->stage_mem, 0, VK_WHOLE_SIZE, 0, &p->stage_map) != VK_SUCCESS) {
    sc_prepared_destroy(p);
    return sc_err(err, "could not map staging memory");
  }

  VkDescriptorPoolSize size;
  size.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  size.descriptorCount = 2;
  VkDescriptorPoolCreateInfo dpci = {VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
  dpci.maxSets = 1;
  dpci.poolSizeCount = 1;
  dpci.pPoolSizes = &size;
  if (vkCreateDescriptorPool(g_gpu.device, &dpci, NULL, &p->desc_pool) != VK_SUCCESS) {
    sc_prepared_destroy(p);
    return sc_err(err, "could not create descriptor pool");
  }
  VkDescriptorSetAllocateInfo ai = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
  ai.descriptorPool = p->desc_pool;
  ai.descriptorSetCount = 1;
  ai.pSetLayouts = &g_gpu.set_layout;
  if (vkAllocateDescriptorSets(g_gpu.device, &ai, &p->set) != VK_SUCCESS) {
    sc_prepared_destroy(p);
    return sc_err(err, "could not allocate descriptor set");
  }
  VkDescriptorBufferInfo wi;
  wi.buffer = s->wbuf;
  wi.offset = 0;
  wi.range = VK_WHOLE_SIZE;
  VkDescriptorBufferInfo ar;
  ar.buffer = p->arena;
  ar.offset = 0;
  ar.range = VK_WHOLE_SIZE;
  VkWriteDescriptorSet writes[2];
  memset(writes, 0, sizeof(writes));
  for (int i = 0; i < 2; i++) {
    writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[i].dstSet = p->set;
    writes[i].dstBinding = (uint32_t)i;
    writes[i].descriptorCount = 1;
    writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  }
  writes[0].pBufferInfo = &wi;
  writes[1].pBufferInfo = &ar;
  vkUpdateDescriptorSets(g_gpu.device, 2, writes, 0, NULL);

  VkCommandBufferAllocateInfo cai = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
  cai.commandPool = g_gpu.pool;
  cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  cai.commandBufferCount = 1;
  if (vkAllocateCommandBuffers(g_gpu.device, &cai, &p->cmd) != VK_SUCCESS) {
    sc_prepared_destroy(p);
    return sc_err(err, "could not allocate command buffer");
  }
  VkFenceCreateInfo fci = {VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
  if (vkCreateFence(g_gpu.device, &fci, NULL, &p->fence) != VK_SUCCESS) {
    sc_prepared_destroy(p);
    return sc_err(err, "could not create fence");
  }
  if (sc_record(p, err)) {
    sc_prepared_destroy(p);
    return 1;
  }
  return 0;
}

static int sc_session_max_span(sc_vk_session* s) {
  static const uint32_t candidates[SC_SPAN_CANDIDATES] = {320u, 240u, 160u, 120u, 80u, 64u};
  for (int i = 0; i < SC_SPAN_CANDIDATES; i++) {
    sc_plan probe;
    char* ignored = NULL;
    if (sc_plan_build(&s->model, candidates[i], &probe, &ignored)) {
      free(ignored);
      continue;
    }
    const uint64_t arena = probe.arena_bytes;
    sc_plan_free(&probe);
    if (arena <= SC_ARENA_BUDGET && arena <= g_gpu.max_range) {
      return (int)candidates[i];
    }
  }
  return 0;
}

static void sc_trim_initializers(sc_model* model) {
  uint8_t* keep = (uint8_t*)calloc((size_t)(model->n_tensors ? model->n_tensors : 1), 1);
  if (!keep) return;
  for (int i = 0; i < model->n_nodes; i++) {
    const sc_node* n = &model->nodes[i];
    if (n->op == SC_OP_ADD || n->op == SC_OP_MUL) {
      if (n->in[0] >= 0) keep[n->in[0]] = 1;
      if (n->in[1] >= 0) keep[n->in[1]] = 1;
    } else if (n->op == SC_OP_RESIZE) {
      if (n->scale_in >= 0) keep[n->scale_in] = 1;
    } else if (n->op == SC_OP_CLIP) {
      if (n->in[1] >= 0) keep[n->in[1]] = 1;
      if (n->in[2] >= 0) keep[n->in[2]] = 1;
    }
  }
  for (int i = 0; i < model->n_tensors; i++) {
    if (!keep[i] && model->tensors[i].data) {
      free(model->tensors[i].data);
      model->tensors[i].data = NULL;
      model->tensors[i].count = 0;
    }
  }
  free(keep);
}

static int gpu_open(const char* model_path, int threads, void** handle, int* scale, int* max_span, char** err) {
  (void)threads;
  if (sc_gpu_init(err)) {
    return 1;
  }
  *handle = NULL;
  sc_vk_session* s = (sc_vk_session*)calloc(1, sizeof(sc_vk_session));
  if (!s) {
    sc_err(err, "out of memory");
    return 1;
  }
  if (sc_model_load(model_path, &s->model, err)) {
    free(s);
    return 1;
  }
  if (sc_weights_build(&s->model, &s->weights, &s->weight_count, err)) {
    sc_model_free(&s->model);
    free(s);
    return 1;
  }
  sc_trim_initializers(&s->model);
  sc_plan probe_plan;
  if (sc_plan_build(&s->model, 32, &probe_plan, err)) {
    free(s->weights);
    sc_model_free(&s->model);
    free(s);
    return 1;
  }
  const int out_scale = (int)probe_plan.out_scale;
  sc_plan_free(&probe_plan);

  VkDeviceSize wsize = (VkDeviceSize)s->weight_count * 4u;
  if (wsize > g_gpu.max_range) {
    free(s->weights);
    sc_model_free(&s->model);
    free(s);
    sc_err(err, "model too large for this device");
    return 1;
  }
  if (sc_buffer_create(wsize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                       VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, &s->wbuf, &s->wbuf_mem, err)) {
    free(s->weights);
    sc_model_free(&s->model);
    free(s);
    return 1;
  }
  VkBuffer stage = VK_NULL_HANDLE;
  VkDeviceMemory stage_mem = VK_NULL_HANDLE;
  const VkMemoryPropertyFlags host = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
  if (sc_buffer_create(wsize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, host, &stage, &stage_mem, err)) {
    sc_buffer_destroy(&s->wbuf, &s->wbuf_mem);
    free(s->weights);
    sc_model_free(&s->model);
    free(s);
    return 1;
  }
  void* map = NULL;
  if (vkMapMemory(g_gpu.device, stage_mem, 0, VK_WHOLE_SIZE, 0, &map) != VK_SUCCESS) {
    sc_buffer_destroy(&stage, &stage_mem);
    sc_buffer_destroy(&s->wbuf, &s->wbuf_mem);
    free(s->weights);
    sc_model_free(&s->model);
    free(s);
    sc_err(err, "could not map staging memory");
    return 1;
  }
  memcpy(map, s->weights, s->weight_count * 4u);
  free(s->weights);
  s->weights = NULL;

  VkCommandBuffer cmd = VK_NULL_HANDLE;
  VkCommandBufferAllocateInfo cai = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
  cai.commandPool = g_gpu.pool;
  cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  cai.commandBufferCount = 1;
  VkFence fence = VK_NULL_HANDLE;
  VkFenceCreateInfo fci = {VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
  int ok = 1;
  if (vkAllocateCommandBuffers(g_gpu.device, &cai, &cmd) != VK_SUCCESS) {
    ok = 0;
  }
  if (ok && vkCreateFence(g_gpu.device, &fci, NULL, &fence) != VK_SUCCESS) {
    ok = 0;
  }
  if (ok) {
    VkCommandBufferBeginInfo bi = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    vkBeginCommandBuffer(cmd, &bi);
    VkBufferCopy c;
    c.srcOffset = 0;
    c.dstOffset = 0;
    c.size = wsize ? wsize : 4;
    vkCmdCopyBuffer(cmd, stage, s->wbuf, 1, &c);
    vkEndCommandBuffer(cmd);
    VkSubmitInfo si = {VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cmd;
    if (vkQueueSubmit(g_gpu.queue, 1, &si, fence) != VK_SUCCESS ||
        vkWaitForFences(g_gpu.device, 1, &fence, VK_TRUE, UINT64_MAX) != VK_SUCCESS) {
      ok = 0;
    }
  }
  if (fence) {
    vkDestroyFence(g_gpu.device, fence, NULL);
  }
  if (cmd) {
    vkFreeCommandBuffers(g_gpu.device, g_gpu.pool, 1, &cmd);
  }
  vkUnmapMemory(g_gpu.device, stage_mem);
  sc_buffer_destroy(&stage, &stage_mem);
  if (!ok) {
    sc_buffer_destroy(&s->wbuf, &s->wbuf_mem);
    sc_model_free(&s->model);
    free(s);
    sc_err(err, "could not upload weights");
    return 1;
  }

  s->max_span = sc_session_max_span(s);
  if (s->max_span <= 0) {
    sc_buffer_destroy(&s->wbuf, &s->wbuf_mem);
    sc_model_free(&s->model);
    free(s);
    sc_err(err, "model does not fit in device memory");
    return 1;
  }
  if (scale) {
    *scale = out_scale > 0 ? out_scale : 1;
  }
  if (max_span) {
    *max_span = s->max_span;
  }
  *handle = s;
  return 0;
}

static sc_prepared* sc_prepared_get(sc_vk_session* s, uint32_t span, char** err) {
  for (int i = 0; i < s->n_plans; i++) {
    if (s->plans[i].span == span) {
      return &s->plans[i];
    }
  }
  if (s->n_plans >= 4) {
    vkDeviceWaitIdle(g_gpu.device);
    sc_prepared_destroy(&s->plans[0]);
    memmove(&s->plans[0], &s->plans[1], (size_t)(s->n_plans - 1) * sizeof(sc_prepared));
    s->n_plans--;
  }
  sc_prepared* p = &s->plans[s->n_plans];
  if (sc_prepared_create(s, span, p, err)) {
    return NULL;
  }
  s->n_plans++;
  return p;
}

static int gpu_run(void* handle, const uint8_t* rgba, int w, int h, uint8_t* out, size_t out_len, int* out_w,
                   int* out_h, char** err) {
  sc_vk_session* s = (sc_vk_session*)handle;
  if (!s || !rgba || !out || w <= 0 || h <= 0 || w != h) {
    return sc_err(err, "invalid arguments");
  }
  sc_prepared* p = sc_prepared_get(s, (uint32_t)w, err);
  if (!p) {
    return 1;
  }
  if ((size_t)p->plan.out_w * p->plan.out_h * 4u > out_len) {
    return sc_err(err, "output buffer too small");
  }

  float* src = (float*)p->stage_map;
  const size_t plane = (size_t)w * (size_t)h;
  sc_rgba_to_planar(rgba, plane, src);
  uint8_t* cptr = (uint8_t*)p->stage_map + p->in_bytes;
  VkDeviceSize cursor = 0;
  for (uint32_t i = 0; i < p->plan.n_consts; i++) {
    memcpy(cptr + cursor, p->plan.consts[i].data, p->plan.consts[i].bytes);
    cursor += p->plan.consts[i].bytes;
  }
  if (vkResetFences(g_gpu.device, 1, &p->fence) != VK_SUCCESS) {
    return sc_err(err, "could not reset fence");
  }
  VkSubmitInfo si = {VK_STRUCTURE_TYPE_SUBMIT_INFO};
  si.commandBufferCount = 1;
  si.pCommandBuffers = &p->cmd;
  if (vkQueueSubmit(g_gpu.queue, 1, &si, p->fence) != VK_SUCCESS) {
    return sc_err(err, "could not submit work");
  }
  if (vkWaitForFences(g_gpu.device, 1, &p->fence, VK_TRUE, UINT64_MAX) != VK_SUCCESS) {
    return sc_err(err, "device lost during upscale");
  }
  const float* dst = (const float*)((const uint8_t*)p->stage_map + p->in_bytes + p->const_bytes);
  sc_planar_to_rgba(dst, (size_t)p->plan.out_w * p->plan.out_h, out);
  if (out_w) {
    *out_w = (int)p->plan.out_w;
  }
  if (out_h) {
    *out_h = (int)p->plan.out_h;
  }
  return 0;
}

static void gpu_close(void* handle) {
  sc_vk_session* s = (sc_vk_session*)handle;
  if (!s) {
    return;
  }
  if (g_gpu.device) {
    vkDeviceWaitIdle(g_gpu.device);
  }
  for (int i = 0; i < s->n_plans; i++) {
    sc_prepared_destroy(&s->plans[i]);
  }
  sc_buffer_destroy(&s->wbuf, &s->wbuf_mem);
  free(s->weights);
  sc_model_free(&s->model);
  free(s);
}

const sc_backend_ops* sc_gpu_ops(void) {
  static const sc_backend_ops ops = {
      .open = gpu_open,
      .run = gpu_run,
      .close = gpu_close,
  };
  return &ops;
}
