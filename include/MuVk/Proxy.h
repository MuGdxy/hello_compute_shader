#pragma once
#include <vulkan/vulkan.h>
#include <stdexcept>

namespace MuVk
{
	struct Proxy
	{
		static VkResult CreateDebugUtilsMessengerEXT(VkInstance instance,
			const VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo,
			const VkAllocationCallbacks* pAllocator,
			VkDebugUtilsMessengerEXT* pDebugMessenger)
		{
			auto func = (PFN_vkCreateDebugUtilsMessengerEXT)
				vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT");
			if (func) return func(instance, pCreateInfo, pAllocator, pDebugMessenger);
			else return VK_ERROR_EXTENSION_NOT_PRESENT;
		}

		static void DestoryDebugUtilsMessengerEXT(VkInstance instance,
			VkDebugUtilsMessengerEXT messenger, const VkAllocationCallbacks* pAllocator)
		{
			auto func = (PFN_vkDestroyDebugUtilsMessengerEXT)
				vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT");
			if (func) func(instance, messenger, pAllocator);
			else throw std::runtime_error("can't find proc");
		}
	};
}