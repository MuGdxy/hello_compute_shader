// ComputeShader.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include <optional>
#include "MuVk/MuVK.h"
#include <array>

class ComputeShaderExample
{
	std::array<float, 1024> inputData;
	std::array<float, 1024> outputData;
	constexpr VkDeviceSize inputDataSize() { return sizeof(inputData); }
	constexpr uint32_t computeShaderProcessUnit() { return 256; }
public:
	ComputeShaderExample()
	{
		inputData.fill(1.0f);
		outputData.fill(0.0f);
	}

	VkInstance instance;
	VkDebugUtilsMessengerEXT debugMessenger;
	void createInstance()
	{
		VkApplicationInfo appInfo = MuVk::applicationInfo();
		appInfo.pApplicationName = "Hello Compute Shader";
		appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
		appInfo.pEngineName = "No Engine";
		appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
		appInfo.apiVersion = VK_API_VERSION_1_0;

		VkInstanceCreateInfo instanceCreateInfo = MuVk::instanceCreateInfo();
		instanceCreateInfo.pApplicationInfo = &appInfo;
		instanceCreateInfo.enabledLayerCount = static_cast<uint32_t>(MuVk::validationLayers.size());
		instanceCreateInfo.ppEnabledLayerNames = MuVk::validationLayers.data();

		if (!MuVk::checkValidationLayerSupport())
			throw std::runtime_error("validation layers requested, but not available!");

		VkDebugUtilsMessengerCreateInfoEXT debugCreateInfo = MuVk::populateDebugMessengerCreateInfo();
		//to debug instance
		instanceCreateInfo.pNext = &debugCreateInfo;

		//get extension properties
		auto extensionProperties = MuVk::Query::instanceExtensionProperties();
		std::cout << extensionProperties << std::endl;

		//required extension
		std::vector<const char*> extensions = { VK_EXT_DEBUG_UTILS_EXTENSION_NAME };
		instanceCreateInfo.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
		instanceCreateInfo.ppEnabledExtensionNames = extensions.data();

		VkResult result = vkCreateInstance(&instanceCreateInfo, nullptr, &instance);
		if (result != VkResult::VK_SUCCESS)
			throw std::runtime_error("failed to create instance");

		if (MuVk::Proxy::CreateDebugUtilsMessengerEXT(
			instance, &debugCreateInfo, nullptr, &debugMessenger) != VK_SUCCESS)
			throw std::runtime_error("failed to setup debug messenger");
	}

	VkPhysicalDevice physicalDevice;

	std::optional<uint32_t> computeTransferQueueFamilyIndex;

	void pickPhyscialDevice()
	{
		auto physicalDevices = MuVk::Query::physicalDevices(instance);
		std::cout << physicalDevices << std::endl;
		for (const auto device : physicalDevices)
		{
			auto queueFamilies = MuVk::Query::queueFamilies(device);
			std::cout << queueFamilies << std::endl;
			for (size_t i = 0; i < queueFamilies.size(); ++i)
			{
				if (queueFamilies[i].queueFlags & (VK_QUEUE_COMPUTE_BIT) &&
					queueFamilies[i].queueFlags & (VK_QUEUE_TRANSFER_BIT))
				{
					computeTransferQueueFamilyIndex = i;
					physicalDevice = device;
					break;
				}
			}
			if (computeTransferQueueFamilyIndex.has_value()) break;
		}
		if (!computeTransferQueueFamilyIndex.has_value())
			throw std::runtime_error("can't find a family that contains compute&transfer queue!");
		else
		{
			std::cout << "Select Physical Device:" << physicalDevice << std::endl;
			std::cout << "Select Queue Index:" << computeTransferQueueFamilyIndex.value() << std::endl;
		}
	}

	VkDevice device;
	VkQueue computeTransferQueue;
	void createLogicalDevice()
	{
		VkDeviceCreateInfo createInfo = MuVk::deviceCreateInfo();
		createInfo.enabledExtensionCount = 0;
		createInfo.ppEnabledExtensionNames = nullptr;
		createInfo.enabledLayerCount = 0;
		createInfo.ppEnabledLayerNames = nullptr;
		createInfo.pEnabledFeatures = nullptr;

		float priority = 1.0f;//default
		VkDeviceQueueCreateInfo queueCreateInfo = MuVk::deviceQueueCreateInfo();
		queueCreateInfo.queueCount = 1;
		queueCreateInfo.pQueuePriorities = &priority;
		queueCreateInfo.queueFamilyIndex = computeTransferQueueFamilyIndex.value();

		createInfo.queueCreateInfoCount = 1;
		createInfo.pQueueCreateInfos = &queueCreateInfo;
		if (vkCreateDevice(physicalDevice, &createInfo, nullptr, &device) != VK_SUCCESS)
		{
			throw std::runtime_error("failed to create logical device");
		}

		vkGetDeviceQueue(device, computeTransferQueueFamilyIndex.value(), 0, &computeTransferQueue);
	}

	VkBuffer storageBuffer;
	VkDeviceMemory storageBufferMemory;
	void createStorageBuffer()
	{

	}

	uint32_t findMemoryType(const VkMemoryRequirements& requirements, VkMemoryPropertyFlags properties)
	{
		VkPhysicalDeviceMemoryProperties memProperties = MuVk::Query::physicalDeviceMemoryProperties(physicalDevice);
		std::cout << memProperties << std::endl;
		for (uint32_t i = 0; i < memProperties.memoryTypeCount; ++i)
		{
			if (requirements.memoryTypeBits & (1 << i) &&
				(memProperties.memoryTypes[i].propertyFlags & properties) == properties)
			{
				std::cout << "pick memory type [" << i << "]\n";
				return i;
			}
		}
	}

	void writeMemoryFromHost()
	{

	}

	VkDescriptorSetLayout descriptorSetLayout;
	void createDescriptorSetLayout()
	{

	}

	VkShaderModule createShaderModule(const std::vector<char>& code)
	{
		VkShaderModule shaderModule;
		VkShaderModuleCreateInfo createInfo = MuVk::shaderModuleCreateInfo();
		createInfo.codeSize = code.size();
		createInfo.pCode = reinterpret_cast<const uint32_t*>(code.data());
		if (vkCreateShaderModule(device, &createInfo, nullptr, &shaderModule) != VK_SUCCESS)
			throw std::runtime_error("fail to create shader module");
		return shaderModule;
	}

	VkPipelineLayout pipelineLayout;
	VkPipeline computePipeline;
	void createComputePipeline()
	{

	}

	VkDescriptorPool descriptorPool;
	void createDescriptorPool()
	{

	}

	VkDescriptorSet descriptorSet;
	void createDescriptorSet()
	{

	}

	VkCommandPool commandPool;
	void createCommandPool()
	{

	}

	VkCommandBuffer commandBuffer;
	void execute()
	{
		std::cout << "input data:\n";
		for (size_t i = 0; i < inputData.size(); ++i)
		{
			if (i % 64 == 0 && i != 0) std::cout << '\n';
			std::cout << inputData[i];
		}
		std::cout << "\n";


		// execute


		std::cout << "output data:\n";
		for (size_t i = 0; i < outputData.size(); ++i)
		{
			if (i % 64 == 0 && i != 0) std::cout << '\n';
			std::cout << outputData[i];
		}
	}

	void Run()
    {
		createInstance();
		pickPhyscialDevice();
		createLogicalDevice();

		createStorageBuffer();
		writeMemoryFromHost();
		createDescriptorSetLayout();
		createComputePipeline();

		createDescriptorPool();
		createDescriptorSet();
		createCommandPool();

		execute();

		cleanUp();
    }

	void cleanUp()
	{
		MuVk::Proxy::DestoryDebugUtilsMessengerEXT(instance, debugMessenger, nullptr);
		vkDestroyDevice(device, nullptr);
		vkDestroyInstance(instance, nullptr);
	}
};

int main()
{
    ComputeShaderExample program;
    program.Run();
}

