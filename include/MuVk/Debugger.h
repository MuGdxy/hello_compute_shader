#pragma once
#include "CreateInfo.h"
#include "Proxy.h"
#include <iostream>
#include <stdexcept>

namespace MuVk
{
	const std::vector<const char*> validationLayers =
	{
		"VK_LAYER_KHRONOS_validation"
	};

	inline VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(
		VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
		VkDebugUtilsMessageTypeFlagsEXT messageType,
		const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
		void* pUserData)
	{
		if (messageSeverity >= VkDebugUtilsMessageSeverityFlagBitsEXT::VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT)
		{
			std::cerr << "validation layer:\n\t" << pCallbackData->pMessage << "\n" << std::endl;
			__debugbreak();
		}
		return VK_FALSE;
	}

	inline VkDebugUtilsMessengerCreateInfoEXT populateDebugMessengerCreateInfo()
	{
		auto createInfo = debugUtilsMessengerCreateInfoEXT();
		createInfo.messageSeverity =
			VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT
			| VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT
			| VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT
			;

		createInfo.messageType =
			VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT
			| VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT
			| VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
		createInfo.pfnUserCallback = debugCallback;
		createInfo.pUserData = nullptr;
		return createInfo;
	}

	inline bool checkValidationLayerSupport()
	{
		uint32_t layerCount;
		vkEnumerateInstanceLayerProperties(&layerCount, nullptr);
		std::vector<VkLayerProperties> availableLayers(layerCount);
		vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());

		for (const char* layerName : validationLayers)
		{
			bool layerFound = false;
			std::cout << "find validation layer:" << std::endl;
			for (const auto& layerProperty : availableLayers)
			{
				if (strcmp(layerName, layerProperty.layerName) == 0)
				{
					layerFound = true;
					std::cout << "\t" << layerName << std::endl;
					break;
				}
			}
			if (!layerFound)
			{
				std::cout << "Can't find validation layer[" << layerName << "]" << std::endl;
				return false;
			}
		}
		return true;
	}
}