#pragma once
#include <vulkan/vulkan.h>
#include <vector>
#include "CreateInfo.h"
#include "Query.h"
namespace MuVk
{
	namespace Utils
	{
		_NODISCARD inline std::vector<VkImage> getSwapchainImages(VkDevice device, VkSwapchainKHR swapchain)
		{
			uint32_t count;
			vkGetSwapchainImagesKHR(device, swapchain, &count, nullptr);
			std::vector<VkImage> swapChainImages(count);
			vkGetSwapchainImagesKHR(device, swapchain, &count, swapChainImages.data());
			return swapChainImages;
		}

		/*! @brief fill VkSwapchainCreateInfoKHR by common config. 
		*	unset:
			{
				.imageSharingMode,
				.queueFamilyIndexCount,
				.pQueueFamilyIndices
			}
			@param <support> Call MuVk::Query::querySwapChainSupport() to get the return struct.
			@param <actualExtent> The window size
			@param <surface> Surface handle
		*/
		_NODISCARD inline VkSwapchainCreateInfoKHR fillSwapchainCreateInfo(
			const Query::SwapChainSupportDetails& support, 
			VkSurfaceKHR surface,
			VkExtent2D actualExtent, 
			VkFormat format = VK_FORMAT_B8G8R8A8_SRGB,
			VkColorSpaceKHR colorSpace = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR)
		{
			auto formatColorSpace = MuVk::Query::chooseSwapSurfaceFormat(support.formats, format, colorSpace);
			auto presentMode = MuVk::Query::chooseSwapPresentMode(support.presentModes);
			auto extent = MuVk::Query::chooseSwapExtent(support.capabilities, actualExtent);
			auto imageCount = MuVk::Query::chooseSwapchainImageCount(support.capabilities);

			VkSwapchainCreateInfoKHR createInfo = MuVk::swapchainCreateInfo();
			createInfo.surface = surface;
			createInfo.minImageCount = imageCount;
			createInfo.imageFormat = formatColorSpace.format;
			createInfo.imageColorSpace = formatColorSpace.colorSpace;
			createInfo.imageExtent = extent;
			createInfo.imageArrayLayers = 1;
			createInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
			createInfo.preTransform = support.capabilities.currentTransform;
			createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
			createInfo.presentMode = presentMode;
			createInfo.clipped = VK_TRUE;
			createInfo.oldSwapchain = VK_NULL_HANDLE;

			return createInfo;
		}

		inline bool checkDeviceExtensionSupport(VkPhysicalDevice device, const std::vector<const char*>& deviceExtensions)
		{
			auto availableExtensions = MuVk::Query::deviceExtensionProperties(device);
			std::set<std::string> requiredExtensions(deviceExtensions.begin(), deviceExtensions.end());
			for (const auto& extension : availableExtensions)
				requiredExtensions.erase(extension.extensionName);
			return requiredExtensions.empty();
		}

		inline void beginSingleTimeCommand(VkDevice device, VkCommandPool commandPool, VkCommandBuffer& commandBuffer)
		{
			VkCommandBufferAllocateInfo allocInfo = MuVk::commandBufferAllocateInfo();
			allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
			allocInfo.commandPool = commandPool;
			allocInfo.commandBufferCount = 1;
			vkAllocateCommandBuffers(device, &allocInfo, &commandBuffer);
			VkCommandBufferBeginInfo beginInfo{};
			beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
			beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
			vkBeginCommandBuffer(commandBuffer, &beginInfo);
		}

		inline void endSingleTimeCommand(VkDevice device, VkQueue queue, VkCommandPool commandPool, VkCommandBuffer commandBuffer)
		{
			vkEndCommandBuffer(commandBuffer);
			VkSubmitInfo submitInfo = MuVk::submitInfo();
			submitInfo.commandBufferCount = 1;
			submitInfo.pCommandBuffers = &commandBuffer;
			vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE);
			vkQueueWaitIdle(queue);
			vkFreeCommandBuffers(device, commandPool, 1, &commandBuffer);
		}

		struct SingleTimeCommandGuard
		{
			VkDevice device;
			VkQueue queue;
			VkCommandPool commandPool;
			VkCommandBuffer commandBuffer;
			SingleTimeCommandGuard(VkDevice device, VkQueue queue, VkCommandPool commandPool)
				:device(device), queue(queue), commandPool(commandPool)
			{
				beginSingleTimeCommand(device, commandPool, commandBuffer);
			}
			~SingleTimeCommandGuard()
			{
				endSingleTimeCommand(device, queue, commandPool, commandBuffer);
			}
		};
	}
}

#ifdef GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
namespace MuVk
{
	namespace Utils
	{
		/*! @brief Append the glfw required extensions to the extension list
		*/
		void appendGLFWRequiredExtensions(std::vector<const char*>& extensions)
		{
			uint32_t glfwExtensionCount = 0;
			const char** glfwExtensions;
			glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);
			for (uint32_t i = 0; i < glfwExtensionCount; ++i) extensions.push_back(glfwExtensions[i]);
		}
	}
}
#endif