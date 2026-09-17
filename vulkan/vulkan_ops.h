#pragma once

#include "neurallm.h"
#include <vector>

#ifdef WITH_VULKAN
std::vector<GpuDevice> vulkan_enumerate_devices();
bool vulkan_set_device(int device_id);
bool vulkan_head_projection(const float* activations, const float* weights,
                            const float* bias, int rows, int embed_dim,
                            int vocab_size, float* logits_out);
#endif
