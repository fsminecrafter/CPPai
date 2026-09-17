#include "vulkan_ops.h"

#ifdef WITH_VULKAN
#include <vulkan/vulkan.h>

#include <array>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Context {
    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice physical = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkQueue queue = VK_NULL_HANDLE;
    uint32_t queue_family = 0;
    VkCommandPool command_pool = VK_NULL_HANDLE;
    VkDescriptorSetLayout set_layout = VK_NULL_HANDLE;
    VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;

    ~Context() {
        if (device) {
            vkDeviceWaitIdle(device);
            if (descriptor_pool) vkDestroyDescriptorPool(device, descriptor_pool, nullptr);
            if (pipeline) vkDestroyPipeline(device, pipeline, nullptr);
            if (pipeline_layout) vkDestroyPipelineLayout(device, pipeline_layout, nullptr);
            if (set_layout) vkDestroyDescriptorSetLayout(device, set_layout, nullptr);
            if (command_pool) vkDestroyCommandPool(device, command_pool, nullptr);
            vkDestroyDevice(device, nullptr);
        }
        if (instance) vkDestroyInstance(instance, nullptr);
    }
};

Context* active = nullptr;

void check(VkResult result, const char* what) {
    if (result != VK_SUCCESS) throw std::runtime_error(std::string("Vulkan ") + what + " failed: " + std::to_string(result));
}

VkInstance make_instance() {
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.pApplicationName = "pyai";
    app.apiVersion = VK_API_VERSION_1_0;
    VkInstanceCreateInfo ci{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO}; ci.pApplicationInfo = &app;
    VkInstance instance = VK_NULL_HANDLE;
    check(vkCreateInstance(&ci, nullptr, &instance), "instance creation");
    return instance;
}

uint32_t compute_queue(VkPhysicalDevice device) {
    uint32_t count = 0; vkGetPhysicalDeviceQueueFamilyProperties(device, &count, nullptr);
    std::vector<VkQueueFamilyProperties> props(count);
    vkGetPhysicalDeviceQueueFamilyProperties(device, &count, props.data());
    for (uint32_t i = 0; i < count; ++i)
        if (props[i].queueFlags & VK_QUEUE_COMPUTE_BIT) return i;
    return UINT32_MAX;
}

std::vector<uint32_t> read_spirv() {
    std::ifstream file(VULKAN_SHADER_PATH, std::ios::binary | std::ios::ate);
    if (!file) throw std::runtime_error("Vulkan shader is missing: " VULKAN_SHADER_PATH);
    auto size = file.tellg();
    if (size <= 0 || size % 4) throw std::runtime_error("Invalid Vulkan SPIR-V shader");
    std::vector<uint32_t> code(static_cast<size_t>(size) / 4);
    file.seekg(0); file.read(reinterpret_cast<char*>(code.data()), size);
    return code;
}

void initialise_pipeline(Context& c) {
    std::array<VkDescriptorSetLayoutBinding, 4> bindings{};
    for (uint32_t i = 0; i < bindings.size(); ++i) {
        bindings[i].binding = i; bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].descriptorCount = 1; bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo sl{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    sl.bindingCount = static_cast<uint32_t>(bindings.size()); sl.pBindings = bindings.data();
    check(vkCreateDescriptorSetLayout(c.device, &sl, nullptr, &c.set_layout), "descriptor layout creation");

    VkPushConstantRange range{}; range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT; range.size = 12;
    VkPipelineLayoutCreateInfo pl{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pl.setLayoutCount = 1; pl.pSetLayouts = &c.set_layout; pl.pushConstantRangeCount = 1; pl.pPushConstantRanges = &range;
    check(vkCreatePipelineLayout(c.device, &pl, nullptr, &c.pipeline_layout), "pipeline layout creation");

    auto code = read_spirv();
    VkShaderModuleCreateInfo sm{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO}; sm.codeSize = code.size() * sizeof(uint32_t); sm.pCode = code.data();
    VkShaderModule module = VK_NULL_HANDLE; check(vkCreateShaderModule(c.device, &sm, nullptr, &module), "shader module creation");
    VkPipelineShaderStageCreateInfo stage{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    stage.stage = VK_SHADER_STAGE_COMPUTE_BIT; stage.module = module; stage.pName = "main";
    VkComputePipelineCreateInfo cp{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO}; cp.stage = stage; cp.layout = c.pipeline_layout;
    VkResult result = vkCreateComputePipelines(c.device, VK_NULL_HANDLE, 1, &cp, nullptr, &c.pipeline);
    vkDestroyShaderModule(c.device, module, nullptr); check(result, "compute pipeline creation");

    VkDescriptorPoolSize pool_size{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 4};
    VkDescriptorPoolCreateInfo dp{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO}; dp.maxSets = 1; dp.poolSizeCount = 1; dp.pPoolSizes = &pool_size;
    check(vkCreateDescriptorPool(c.device, &dp, nullptr, &c.descriptor_pool), "descriptor pool creation");
}

uint32_t memory_type(Context& c, uint32_t bits) {
    VkPhysicalDeviceMemoryProperties props{}; vkGetPhysicalDeviceMemoryProperties(c.physical, &props);
    for (uint32_t i = 0; i < props.memoryTypeCount; ++i)
        if ((bits & (1u << i)) && (props.memoryTypes[i].propertyFlags & (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) ==
                                      (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) return i;
    throw std::runtime_error("Vulkan device has no host-visible coherent memory");
}

struct Buffer { VkBuffer buffer = VK_NULL_HANDLE; VkDeviceMemory memory = VK_NULL_HANDLE; void* mapped = nullptr; };
Buffer make_buffer(Context& c, VkDeviceSize size) {
    Buffer out; VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO}; bi.size = size; bi.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT; bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    check(vkCreateBuffer(c.device, &bi, nullptr, &out.buffer), "buffer creation");
    VkMemoryRequirements req{}; vkGetBufferMemoryRequirements(c.device, out.buffer, &req);
    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO}; ai.allocationSize = req.size; ai.memoryTypeIndex = memory_type(c, req.memoryTypeBits);
    check(vkAllocateMemory(c.device, &ai, nullptr, &out.memory), "memory allocation");
    check(vkBindBufferMemory(c.device, out.buffer, out.memory, 0), "buffer binding");
    check(vkMapMemory(c.device, out.memory, 0, size, 0, &out.mapped), "memory mapping"); return out;
}
void destroy_buffer(Context& c, Buffer& b) { if (b.mapped) vkUnmapMemory(c.device, b.memory); if (b.buffer) vkDestroyBuffer(c.device, b.buffer, nullptr); if (b.memory) vkFreeMemory(c.device, b.memory, nullptr); }

} // namespace

std::vector<GpuDevice> vulkan_enumerate_devices() {
    std::vector<GpuDevice> out;
    VkInstance instance = VK_NULL_HANDLE;
    try { instance = make_instance(); } catch (...) { return out; }
    uint32_t count = 0;
    if (vkEnumeratePhysicalDevices(instance, &count, nullptr) == VK_SUCCESS) {
        std::vector<VkPhysicalDevice> devices(count); vkEnumeratePhysicalDevices(instance, &count, devices.data());
        for (uint32_t i = 0; i < count; ++i) {
            VkPhysicalDeviceProperties props{}; vkGetPhysicalDeviceProperties(devices[i], &props);
            if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_CPU || compute_queue(devices[i]) == UINT32_MAX) continue;
            VkPhysicalDeviceMemoryProperties memory{}; vkGetPhysicalDeviceMemoryProperties(devices[i], &memory);
            size_t bytes = 0; for (uint32_t h = 0; h < memory.memoryHeapCount; ++h) if (memory.memoryHeaps[h].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) bytes += memory.memoryHeaps[h].size;
            out.push_back({static_cast<int>(i), props.deviceName, bytes});
        }
    }
    vkDestroyInstance(instance, nullptr); return out;
}

bool vulkan_set_device(int id) {
    delete active; active = new Context();
    try {
        active->instance = make_instance(); uint32_t count = 0; check(vkEnumeratePhysicalDevices(active->instance, &count, nullptr), "device enumeration");
        std::vector<VkPhysicalDevice> devices(count); check(vkEnumeratePhysicalDevices(active->instance, &count, devices.data()), "device enumeration");
        if (id < 0 || static_cast<uint32_t>(id) >= count || compute_queue(devices[id]) == UINT32_MAX) throw std::runtime_error("Vulkan compute device not found");
        active->physical = devices[id]; active->queue_family = compute_queue(active->physical); float priority = 1.f;
        VkDeviceQueueCreateInfo qi{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO}; qi.queueFamilyIndex = active->queue_family; qi.queueCount = 1; qi.pQueuePriorities = &priority;
        VkDeviceCreateInfo di{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO}; di.queueCreateInfoCount = 1; di.pQueueCreateInfos = &qi;
        check(vkCreateDevice(active->physical, &di, nullptr, &active->device), "device creation"); vkGetDeviceQueue(active->device, active->queue_family, 0, &active->queue);
        VkCommandPoolCreateInfo pi{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO}; pi.queueFamilyIndex = active->queue_family;
        check(vkCreateCommandPool(active->device, &pi, nullptr, &active->command_pool), "command pool creation"); initialise_pipeline(*active); return true;
    } catch (...) { delete active; active = nullptr; return false; }
}

bool vulkan_head_projection(const float* x, const float* w, const float* b, int rows, int d, int v, float* out) {
    if (!active && !vulkan_set_device(0)) return false;
    try {
        Context& c = *active; Buffer bx = make_buffer(c, static_cast<VkDeviceSize>(rows) * d * 4), bw = make_buffer(c, static_cast<VkDeviceSize>(d) * v * 4), bb = make_buffer(c, static_cast<VkDeviceSize>(v) * 4), by = make_buffer(c, static_cast<VkDeviceSize>(rows) * v * 4);
        std::memcpy(bx.mapped, x, static_cast<size_t>(rows)*d*4); std::memcpy(bw.mapped, w, static_cast<size_t>(d)*v*4); std::memcpy(bb.mapped, b, static_cast<size_t>(v)*4);
        VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO}; ai.descriptorPool = c.descriptor_pool; ai.descriptorSetCount = 1; ai.pSetLayouts = &c.set_layout; VkDescriptorSet set; check(vkAllocateDescriptorSets(c.device, &ai, &set), "descriptor allocation");
        std::array<VkDescriptorBufferInfo,4> info{{{bx.buffer,0,VK_WHOLE_SIZE},{bw.buffer,0,VK_WHOLE_SIZE},{bb.buffer,0,VK_WHOLE_SIZE},{by.buffer,0,VK_WHOLE_SIZE}}}; std::array<VkWriteDescriptorSet,4> writes{};
        for(uint32_t i=0;i<4;++i){writes[i]={VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};writes[i].dstSet=set;writes[i].dstBinding=i;writes[i].descriptorCount=1;writes[i].descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;writes[i].pBufferInfo=&info[i];} vkUpdateDescriptorSets(c.device,4,writes.data(),0,nullptr);
        VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};cai.commandPool=c.command_pool;cai.level=VK_COMMAND_BUFFER_LEVEL_PRIMARY;cai.commandBufferCount=1;VkCommandBuffer cmd;check(vkAllocateCommandBuffers(c.device,&cai,&cmd),"command allocation");VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};check(vkBeginCommandBuffer(cmd,&begin),"command begin");vkCmdBindPipeline(cmd,VK_PIPELINE_BIND_POINT_COMPUTE,c.pipeline);vkCmdBindDescriptorSets(cmd,VK_PIPELINE_BIND_POINT_COMPUTE,c.pipeline_layout,0,1,&set,0,nullptr);uint32_t shape[3]={static_cast<uint32_t>(rows),static_cast<uint32_t>(d),static_cast<uint32_t>(v)};vkCmdPushConstants(cmd,c.pipeline_layout,VK_SHADER_STAGE_COMPUTE_BIT,0,sizeof(shape),shape);vkCmdDispatch(cmd,(shape[0]*shape[2]+63)/64,1,1);check(vkEndCommandBuffer(cmd),"command end");VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};submit.commandBufferCount=1;submit.pCommandBuffers=&cmd;check(vkQueueSubmit(c.queue,1,&submit,VK_NULL_HANDLE),"queue submit");check(vkQueueWaitIdle(c.queue),"queue wait");std::memcpy(out,by.mapped,static_cast<size_t>(rows)*v*4);vkFreeCommandBuffers(c.device,c.command_pool,1,&cmd);vkFreeDescriptorSets(c.device,c.descriptor_pool,1,&set);destroy_buffer(c,bx);destroy_buffer(c,bw);destroy_buffer(c,bb);destroy_buffer(c,by);return true;
    } catch (...) { return false; }
}
#endif
