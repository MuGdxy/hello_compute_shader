#include <iostream>
#include <optional>
#include <array>
#include<glm/glm.hpp>
#include <glm/gtc/random.hpp>
#include "DataDump.h"
#include <chrono>
#include <vulkan/vk_mem_alloc.h>
#include <filesystem>
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include "MuVk/MuVK.h"

#ifndef MU_SHADER_PATH
#define MU_SHADER_PATH "./shader/"
#endif

class ComputeShaderExample
{
	uint32_t computeShaderProcessUnit;
	TargetBuffer target;
	Camera camera;
	glm::ivec2 screenSize;
public:
	ComputeShaderExample()
	{
		
	}
	HittableDump hittables;
	MaterialDump materials;
	PushConstantData pushConstantData;
	const size_t maxSamplesForSingleShader = 50;
	void GreatScene()
	{
		//image
		const auto aspectRatio = 16.0 / 9.0;
		const int imageWidth = 800;
		const int imageHeight = static_cast<int>(imageWidth / aspectRatio);

		pushConstantData.screenSize = { imageWidth, imageHeight };
		pushConstantData.maxDepth = 50;
		pushConstantData.totalSamples = 800;

		target = TargetBuffer(imageWidth, imageHeight);

		auto groundMaterial = materials.Allocate<Lambertian>(color(249.0/255.0, 189.0/255.0, 219.0/255.0));
		hittables.Allocate<Sphere>(point3(0, -1000, 0), 1000)->mat = groundMaterial;

		for (int a = -11; a < 11; a++) {
			for (int b = -11; b < 11; b++) {
				using namespace glm;
				auto choose_mat = linearRand(0.0, 1.0);
				point3 center(a + 0.9 * linearRand(0.0, 1.0), 0.2, b + 0.9 * linearRand(0.0, 1.0));

				if (distance(center, point3(4, 0.2, 0)) > 0.9) {
					Material* mat;
					auto percent = 0.0;
					//orange
					if (choose_mat/0.7 < (percent+=0.2)) {
						// diffuse
						auto albedo = linearRand(vec3(-0.1), vec3(0.1)) * linearRand(vec3(-0.1), vec3(0.1)) 
							+ vec3(254.0/255.0, 193.0/255.0, 172.0/255.0);
						mat = materials.Allocate<Lambertian>(albedo);
						hittables.Allocate<Sphere>(center, 0.2)->mat = mat;
					}
					//purple
					else if (choose_mat/0.7 < (percent += 0.15)) {
						// diffuse
						auto albedo = linearRand(vec3(-0.1), vec3(0.1)) * linearRand(vec3(-0.1), vec3(0.1))
							+ vec3(249.0 / 255.0, 205.0 / 255.0, 255.0 / 255.0);
						mat = materials.Allocate<Lambertian>(albedo);
						hittables.Allocate<Sphere>(center, 0.2)->mat = mat;
					}
					//blue
					else if (choose_mat/0.7 < (percent += 0.20)) {
						// diffuse
						auto albedo = linearRand(vec3(-0.1), vec3(0.1)) * linearRand(vec3(-0.1), vec3(0.1))
							+ vec3(187.0 / 255.0, 240.0 / 255.0, 239.0 / 255.0);
						mat = materials.Allocate<Lambertian>(albedo);
						hittables.Allocate<Sphere>(center, 0.2)->mat = mat;
					}
					//dark blue
					else if (choose_mat/0.7 < (percent += 0.10)) {
						// diffuse
						auto albedo = linearRand(vec3(-0.1), vec3(0.1)) * linearRand(vec3(-0.1), vec3(0.1))
							+ vec3(185.0 / 255.0, 203.0 / 255.0, 255.0 / 255.0);
						mat = materials.Allocate<Lambertian>(albedo);
						hittables.Allocate<Sphere>(center, 0.2)->mat = mat;
					}
					//green
					else if (choose_mat / 0.7 < (percent += 0.15)) {
						// diffuse
						auto albedo = linearRand(vec3(-0.1), vec3(0.1)) * linearRand(vec3(-0.1), vec3(0.1))
							+ vec3(197.0 / 255.0, 243.0 / 255.0, 195.0 / 255.0);
						mat = materials.Allocate<Lambertian>(albedo);
						hittables.Allocate<Sphere>(center, 0.2)->mat = mat;
					}
					//yellow
					else if (choose_mat / 0.7 < (percent += 0.20)) {
						// diffuse
						auto albedo = linearRand(vec3(-0.1), vec3(0.1)) * linearRand(vec3(-0.1), vec3(0.1))
							+ vec3(245.0 / 255.0, 241.0 / 255.0, 185.0 / 255.0);
						mat = materials.Allocate<Lambertian>(albedo);
						hittables.Allocate<Sphere>(center, 0.2)->mat = mat;
					}
					////red
					//else if (choose_mat / 0.8 < (percent += 0.10)) {
					//	// diffuse
					//	auto albedo = linearRand(vec3(-0.1), vec3(0.1)) * linearRand(vec3(-0.1), vec3(0.1))
					//		+ vec3(251.0 / 255.0, 197.0 / 255.0, 201.0 / 255.0);
					//	mat = materials.Allocate<Lambertian>(albedo);
					//	hittables.Allocate<Sphere>(center, 0.2)->mat = mat;
					//}
					else if (choose_mat < 0.9) {
						// metal
						auto albedo = linearRand(vec3(0.5), vec3(1));
						auto fuzz = linearRand(0.0, 0.5);
						mat = materials.Allocate<Metal>(albedo, fuzz);
						hittables.Allocate<Sphere>(center, 0.2)->mat = mat;
					}
					else {
						//glass
						mat = materials.Allocate<Dielectric>(1.5);
						hittables.Allocate<Sphere>(center, 0.2)->mat = mat;
					}
				}
			}
		}

		auto material1 = materials.Allocate<Dielectric>(1.5);
		hittables.Allocate<Sphere>(point3(0, 1, 0), 1.0)->mat = material1;

		auto material2 = materials.Allocate<Lambertian>(color(242.0/255.0, 220.0/255.0, 196.0/255.0));
		hittables.Allocate<Sphere>(point3(-4, 1, 0), 1.0)->mat = material2;

		auto material3 = materials.Allocate<Metal>(color(253.0/255.0, 236.0/255.0, 223.0/255.0), 0.0);
		hittables.Allocate<Sphere>(point3(4, 1, 0), 1.0)->mat = material3;
		// Camera
		point3 lookfrom(13, 2, 3);
		point3 lookat(0, 0, 0);
		vec3 vup(0, 1, 0);
		auto distTofocus = 10.0;
		auto aperture = 0;

		camera = Camera(lookfrom, lookat, vup, 30, aspectRatio, aperture, distTofocus);


		pushConstantData.hittableCount = hittables.Count();
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
		//instanceCreateInfo.enabledLayerCount = static_cast<uint32_t>(MuVk::validationLayers.size());
		//instanceCreateInfo.ppEnabledLayerNames = MuVk::validationLayers.data();
		instanceCreateInfo.enabledLayerCount = 0;
		instanceCreateInfo.ppEnabledLayerNames = nullptr;
		

		if (!MuVk::checkValidationLayerSupport())
			throw std::runtime_error("validation layers requested, but not available!");

		VkDebugUtilsMessengerCreateInfoEXT debugCreateInfo = MuVk::populateDebugMessengerCreateInfo();
		//to debug instance
		instanceCreateInfo.pNext = &debugCreateInfo;
		instanceCreateInfo.pNext = nullptr;

		//get extension properties
		auto extensionProperties = MuVk::Query::instanceExtensionProperties();
		std::cout << extensionProperties << std::endl;

		//required extension
		std::vector<const char*> extensions = 
		{ 
			VK_EXT_DEBUG_UTILS_EXTENSION_NAME
		};
		MuVk::Utils::appendGLFWRequiredExtensions(extensions);

		instanceCreateInfo.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
		instanceCreateInfo.ppEnabledExtensionNames = extensions.data();

		//VkValidationFeaturesEXT features{};
		//features.sType = VK_STRUCTURE_TYPE_VALIDATION_FEATURES_EXT;
		//const std::vector<VkValidationFeatureEnableEXT> enabledValidationFeatures = 
		//{
		//	VK_VALIDATION_FEATURE_ENABLE_DEBUG_PRINTF_EXT
		//};
		//features.enabledValidationFeatureCount = enabledValidationFeatures.size();
		//features.pEnabledValidationFeatures = enabledValidationFeatures.data();
		//instanceCreateInfo.pNext = &features;

		VkResult result = vkCreateInstance(&instanceCreateInfo, nullptr, &instance);
		if (result != VkResult::VK_SUCCESS)
			throw std::runtime_error("failed to create instance");

		if (MuVk::Proxy::createDebugUtilsMessengerEXT(
			instance, &debugCreateInfo, nullptr, &debugMessenger) != VK_SUCCESS)
			throw std::runtime_error("failed to setup debug messenger");
	}

	VkPhysicalDevice physicalDevice;
	std::optional<uint32_t> queueFamilyIndex;
	void pickPhyscialDevice()
	{
		auto physicalDevices = MuVk::Query::physicalDevices(instance);
		std::cout << physicalDevices << std::endl;
		for (const auto device : physicalDevices)
		{
			auto queueFamilies = MuVk::Query::physicalDeviceQueueFamilyProperties(device);
			std::cout << queueFamilies << std::endl;
			for (size_t i = 0; i < queueFamilies.size(); ++i)
			{
				if (queueFamilies[i].queueFlags & (VK_QUEUE_COMPUTE_BIT) &&
					queueFamilies[i].queueFlags & (VK_QUEUE_TRANSFER_BIT))
				{
					VkBool32 presentSupport;
					vkGetPhysicalDeviceSurfaceSupportKHR(device, i, surface, &presentSupport);
					if (presentSupport)
					{
						queueFamilyIndex = i;
						physicalDevice = device;
						break;
					}
				}
			}
			if (queueFamilyIndex.has_value()) break;
		}
		auto properties = MuVk::Query::deviceExtensionProperties(physicalDevice);
		std::cout << properties << std::endl;
		if (!queueFamilyIndex.has_value())
			throw std::runtime_error("can't find a family that contains compute&transfer queue!");
		else
		{
			std::cout << "Select Physical Device:" << physicalDevice << std::endl;
			std::cout << "Select Queue Index:" << queueFamilyIndex.value() << std::endl;
		}
		auto p = MuVk::Query::physicalDeviceProperties(physicalDevice);
		std::cout << "maxComputeWorkGroupInvocations:" << p.limits.maxComputeWorkGroupInvocations << std::endl;
		computeShaderProcessUnit = sqrt(p.limits.maxComputeWorkGroupInvocations);
	}

	VkDevice device;
	VkQueue universalQueue;
	void createLogicalDevice()
	{
		VkDeviceCreateInfo createInfo = MuVk::deviceCreateInfo();
		const std::vector<const char*> extensions = 
		{
			VK_KHR_SHADER_NON_SEMANTIC_INFO_EXTENSION_NAME,
			VK_KHR_SWAPCHAIN_EXTENSION_NAME
		};
		createInfo.enabledExtensionCount = extensions.size();
		createInfo.ppEnabledExtensionNames = extensions.data();
		float priority = 1.0f;//default
		VkDeviceQueueCreateInfo queueCreateInfo = MuVk::deviceQueueCreateInfo();
		queueCreateInfo.queueCount = 1;
		queueCreateInfo.pQueuePriorities = &priority;
		queueCreateInfo.queueFamilyIndex = queueFamilyIndex.value();

		createInfo.queueCreateInfoCount = 1;
		createInfo.pQueueCreateInfos = &queueCreateInfo;
		if (vkCreateDevice(physicalDevice, &createInfo, nullptr, &device) != VK_SUCCESS)
		{
			throw std::runtime_error("failed to create logical device");
		}
		vkGetDeviceQueue(device, queueFamilyIndex.value(), 0, &universalQueue);
	}

	std::array<VkBuffer,5> storageBuffers;
	
	std::array <VmaAllocation,5>  storageAllocation;
	VkBuffer uniformBuffer;
	VmaAllocation uniformAllocation;
	void createBuffer(VkBufferUsageFlags usage, VkBuffer& buffer, VkDeviceMemory& memory)
	{
		VkBufferCreateInfo createInfo = MuVk::bufferCreateInfo();
		createInfo.size = target.DumpSize();
		createInfo.usage = usage;
		createInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
		createInfo.queueFamilyIndexCount = 1;
		createInfo.pQueueFamilyIndices = &queueFamilyIndex.value();
		createInfo.pNext = nullptr;

		if (vkCreateBuffer(device, &createInfo, nullptr, &buffer) != VK_SUCCESS)
			throw std::runtime_error("failed to create buffer!");

		VkMemoryRequirements requirements = MuVk::Query::memoryRequirements(device, buffer);
		std::cout << requirements << std::endl;

		VkMemoryAllocateInfo allocInfo = MuVk::memoryAllocateInfo();
		allocInfo.allocationSize = requirements.size;
		allocInfo.memoryTypeIndex = findMemoryType(requirements,
			VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
		if (vkAllocateMemory(device, &allocInfo, nullptr, &memory) != VK_SUCCESS)
			throw std::runtime_error("failed to allocate buffer memory");

		vkBindBufferMemory(device, buffer, memory, 0);
	}
	void createBuffer(VkBufferUsageFlags usage, VkBuffer& buffer, VmaAllocation& allocation)
	{
		VkBufferCreateInfo bufferInfo = MuVk::bufferCreateInfo();
		bufferInfo.size = target.DumpSize();
		bufferInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
		bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
		bufferInfo.queueFamilyIndexCount = VK_QUEUE_FAMILY_IGNORED;
		bufferInfo.pQueueFamilyIndices = nullptr;

		VmaAllocationCreateInfo allocInfo = {};
		allocInfo.usage = VMA_MEMORY_USAGE_CPU_ONLY;

		//VkBuffer buffer;
		//VmaAllocation allocation;
		if (vmaCreateBuffer(allocator, &bufferInfo, &allocInfo, &buffer, &allocation, nullptr)
			!= VK_SUCCESS)
			throw std::runtime_error("failed to create buffer!");
	}
	void createBuffers()
	{
		for (size_t i = 0; i < storageBuffers.size(); i++)
		{
			createBuffer(VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, storageBuffers[i], storageAllocation[i]);
		}
		createBuffer(VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, uniformBuffer, uniformAllocation);
	}
	
	VkImage storageImage;
	VmaAllocation storageImageAllocation;
	void createStorageImage()
	{
		VkImageCreateInfo imageInfo = MuVk::imageCreateInfo();
		imageInfo.imageType = VK_IMAGE_TYPE_2D;
		imageInfo.extent.width = static_cast<uint32_t>(WIDTH);
		imageInfo.extent.height = static_cast<uint32_t>(HEIGHT);
		imageInfo.extent.depth = 1;
		imageInfo.mipLevels = 1;
		imageInfo.arrayLayers = 1;
		imageInfo.format = VK_FORMAT_R8G8B8A8_SRGB;
		imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
		imageInfo.mipLevels = 1;
		imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		imageInfo.usage = VK_IMAGE_USAGE_STORAGE_BIT;
		imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
		imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
		imageInfo.flags = 0;
		VmaAllocationCreateInfo allocInfo = {};
		allocInfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;
		vmaCreateImage(allocator, &imageInfo, &allocInfo, &storageImage, &storageImageAllocation, nullptr);
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

	void WriteMemory(VkDeviceMemory memory, void* dataBlock, VkDeviceSize size)
	{
		void* data;
		if (vkMapMemory(device, memory, 0, size, 0, &data) != VK_SUCCESS)
			throw std::runtime_error("failed to map memory");
		memcpy(data, dataBlock, size);
		vkUnmapMemory(device, memory);
	}

	void WriteMemory(VmaAllocation allocation, void* dataBlock, VkDeviceSize size)
	{
		VmaAllocationInfo info;
		vmaGetAllocationInfo(allocator, allocation, &info);
		void* data;
		if (vkMapMemory(device, info.deviceMemory, info.offset, size, 0, &data) != VK_SUCCESS)
			throw std::runtime_error("failed to map memory");
		memcpy(data, dataBlock, size);
		vkUnmapMemory(device, info.deviceMemory);
	}

	void ReadMemory(VmaAllocation allocation, void* dataBlock, VkDeviceSize size)
	{
		VmaAllocationInfo info;
		vmaGetAllocationInfo(allocator, allocation, &info);
		void* data;
		if (vkMapMemory(device, info.deviceMemory, info.offset, size, 0, &data) != VK_SUCCESS)
			throw std::runtime_error("failed to map memory");
		memcpy(dataBlock, data, size);
		vkUnmapMemory(device, info.deviceMemory);
	}

	void writeMemoryFromHost()
	{
		//WriteMemory(storageBufferMemorys[0], target.dump.data(), target.DumpSize());
		//materials.Dump();
		//hittables.Dump();
		//materials.WriteMemory(device, storageBufferMemorys[1], storageBufferMemorys[2]);
		//hittables.WriteMemory(device, storageBufferMemorys[3], storageBufferMemorys[4]);
		//WriteMemory(uniformBufferMemory, &camera, sizeof(camera));

		WriteMemory(storageAllocation[0], target.dump.data(), target.DumpSize());
		materials.Dump();
		hittables.Dump();
		materials.WriteMemory(device, allocator, storageAllocation[1], storageAllocation[2]);
		hittables.WriteMemory(device, allocator, storageAllocation[3], storageAllocation[4]);
		WriteMemory(uniformAllocation, &camera, sizeof(camera));
	}

	std::array<VkDescriptorSetLayout, 2> descriptorSetLayouts;
	void createDescriptorSetLayout()
	{
		{
			std::array<VkDescriptorSetLayoutBinding, 5> bindings;
			for (size_t i = 0; i < bindings.size(); i++)
			{
				bindings[i].binding = i;
				bindings[i].descriptorCount = 1;
				bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
				bindings[i].pImmutableSamplers = nullptr;
				bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
			}

			VkDescriptorSetLayoutCreateInfo createInfo = MuVk::descriptorSetLayoutCreateInfo();
			createInfo.bindingCount = bindings.size();
			createInfo.pBindings = bindings.data();

			if (vkCreateDescriptorSetLayout(
				device, &createInfo, nullptr, &descriptorSetLayouts[0]) != VK_SUCCESS)
				throw std::runtime_error("failed to create descriptorSetLayout");
		}

		{
			std::array<VkDescriptorSetLayoutBinding, 1> bindings;
			for (size_t i = 0; i < bindings.size(); i++)
			{
				bindings[i].binding = 0;
				bindings[i].descriptorCount = 1;
				bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
				bindings[i].pImmutableSamplers = nullptr;
				bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
			}

			VkDescriptorSetLayoutCreateInfo createInfo = MuVk::descriptorSetLayoutCreateInfo();
			createInfo.bindingCount = bindings.size();
			createInfo.pBindings = bindings.data();

			if (vkCreateDescriptorSetLayout(
				device, &createInfo, nullptr, &descriptorSetLayouts[1]) != VK_SUCCESS)
				throw std::runtime_error("failed to create descriptorSetLayout");
		}

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
		auto computeShaderCode = MuVk::readFile(MU_SHADER_PATH "/ray_tracing/ray_tracing.comp.spv");
		auto computeShaderModule = createShaderModule(computeShaderCode);
		VkPipelineShaderStageCreateInfo shaderStageCreateInfo = MuVk::pipelineShaderStageCreateInfo();
		shaderStageCreateInfo.module = computeShaderModule;
		shaderStageCreateInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
		shaderStageCreateInfo.pName = "main";

		VkPushConstantRange range{};
		range.offset = 0;
		range.size = sizeof(PushConstantData);
		range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

		VkPipelineLayoutCreateInfo layoutCreateInfo = MuVk::pipelineLayoutCreateInfo();
		layoutCreateInfo.setLayoutCount = descriptorSetLayouts.size();
		layoutCreateInfo.pSetLayouts = descriptorSetLayouts.data();
		layoutCreateInfo.pushConstantRangeCount = 1;
		layoutCreateInfo.pPushConstantRanges = &range;
		if (vkCreatePipelineLayout(device, &layoutCreateInfo, nullptr, &pipelineLayout)
			!= VK_SUCCESS)
			throw std::runtime_error("failed to create pipeline layout!");

		VkComputePipelineCreateInfo createInfo = MuVk::computePipelineCreateInfo();
		createInfo.basePipelineHandle = VK_NULL_HANDLE;
		createInfo.basePipelineIndex = -1;
		createInfo.stage = shaderStageCreateInfo;
		createInfo.layout = pipelineLayout;

		if (vkCreateComputePipelines(device, nullptr, 1, &createInfo, nullptr, &computePipeline)
			!= VK_SUCCESS)
			throw std::runtime_error("failed to create compute pipeline");
		vkDestroyShaderModule(device, computeShaderModule, nullptr);
	}

	VkDescriptorPool descriptorPool;
	void createDescriptorPool()
	{
		std::array<VkDescriptorPoolSize, 2> poolSize;
		poolSize[0].descriptorCount = 1 + 2 + 2;
		poolSize[0].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;

		poolSize[1].descriptorCount = 1;
		poolSize[1].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;


		VkDescriptorPoolCreateInfo createInfo = MuVk::descriptorPoolCreateInfo();
		createInfo.poolSizeCount = poolSize.size();
		createInfo.pPoolSizes = poolSize.data();
		createInfo.maxSets = 2;
		
		if (vkCreateDescriptorPool(device, &createInfo, nullptr, &descriptorPool)
			!= VK_SUCCESS)
			throw std::runtime_error("failed to create descriptor pool!");
	}

	std::array<VkDescriptorSet, 2> descriptorSets;
	void createDescriptorSet()
	{
		VkDescriptorSetAllocateInfo allocInfo = MuVk::descriptorSetAllocateInfo();
		allocInfo.descriptorPool = descriptorPool;
		allocInfo.descriptorSetCount = descriptorSetLayouts.size();
		allocInfo.pSetLayouts = descriptorSetLayouts.data();

		if (vkAllocateDescriptorSets(device, &allocInfo, descriptorSets.data())
			!= VK_SUCCESS)
			throw std::runtime_error("failed to create descriptor set!");

		std::array<VkDescriptorBufferInfo,5> storageBufferInfos;
		for (size_t i = 0; i < storageBufferInfos.size(); i++)
		{
			storageBufferInfos[i].buffer = storageBuffers[i];
			storageBufferInfos[i].offset = 0;
		}
		storageBufferInfos[0].range = target.DumpSize();
		storageBufferInfos[1].range = materials.HeadSize();
		storageBufferInfos[2].range = materials.DumpSize();
		storageBufferInfos[3].range = hittables.HeadSize();
		storageBufferInfos[4].range = hittables.DumpSize();

		VkDescriptorBufferInfo uniformBufferInfo;
		uniformBufferInfo.buffer = uniformBuffer;
		uniformBufferInfo.offset = 0;
		uniformBufferInfo.range = sizeof(camera);

		VkWriteDescriptorSet write = MuVk::writeDescriptorSet();
		std::array<VkWriteDescriptorSet, 6> writes;
		writes.fill(write);
		for (size_t i = 0; i < writes.size() - 1; ++i)
		{
			writes[i].descriptorCount = 1;
			writes[i].dstSet = descriptorSets[0];
			writes[i].dstArrayElement = 0;
			writes[i].dstBinding = i;
			writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
			writes[i].pBufferInfo = &storageBufferInfos[i];
		}
		writes[5].descriptorCount = 1;
		writes[5].dstSet = descriptorSets[1];
		writes[5].dstArrayElement = 0;
		writes[5].dstBinding = 0;
		writes[5].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		writes[5].pBufferInfo = &uniformBufferInfo;
		
		vkUpdateDescriptorSets(device, writes.size(), writes.data(), 0, nullptr);
	}

	VkCommandPool commandPool;
	void createCommandPool()
	{
		VkCommandPoolCreateInfo createInfo = MuVk::commandPoolCreateInfo();
		createInfo.queueFamilyIndex = queueFamilyIndex.value();
		if (vkCreateCommandPool(device, &createInfo, nullptr, &commandPool)
			!= VK_SUCCESS)
			throw std::runtime_error("failed to create command pool!");
	}


	VkCommandBuffer commandBuffer;
	void createCommandBuffer()
	{
		VkCommandBufferAllocateInfo allocInfo = MuVk::commandBufferAllocateInfo();
		allocInfo.commandBufferCount = 1;
		allocInfo.commandPool = commandPool;
		allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
		if (vkAllocateCommandBuffers(device, &allocInfo, &commandBuffer)
			!= VK_SUCCESS)
			throw std::runtime_error("failed to create command buffer!");
	}

	void execute(uint32_t sampleStart, uint32_t samples)
	{
		auto start = std::chrono::high_resolution_clock::now();
		VkCommandBufferBeginInfo beginInfo = MuVk::commandBufferBeginInfo();
		vkBeginCommandBuffer(commandBuffer, &beginInfo);
		vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, computePipeline);
		vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout,
			0, descriptorSets.size(), descriptorSets.data(), 0, nullptr);

		pushConstantData.samples = samples;
		pushConstantData.sampleStart = sampleStart;

		vkCmdPushConstants(commandBuffer, pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT,
			0, sizeof(pushConstantData), &pushConstantData);
		auto sizeX = static_cast<uint32_t>(target.width / computeShaderProcessUnit + 1);
		auto sizeY = static_cast<uint32_t>(target.height / computeShaderProcessUnit + 1);
		std::cout << "WorkX=" << sizeX << ' ' << "WorkY=" << sizeY << std::endl;
		vkCmdDispatch(commandBuffer,
			sizeX, //x
			sizeY, //y
			1  //z
		);
		vkEndCommandBuffer(commandBuffer);
		VkSubmitInfo submitInfo = MuVk::submitInfo();
		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers = &commandBuffer;
		submitInfo.waitSemaphoreCount = 0;
		submitInfo.signalSemaphoreCount = 0;

		if (vkQueueSubmit(universalQueue, 1, &submitInfo, VK_NULL_HANDLE)
			!= VK_SUCCESS)
			throw std::runtime_error("failed to submit command buffer!");

		//wait the calculation to finish
		if (vkQueueWaitIdle(universalQueue) != VK_SUCCESS)
			throw std::runtime_error("failed to wait queue idle!");
		auto end = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double> delta = end - start;
		std::cout << "[" << sampleStart + samples << "/" << pushConstantData.totalSamples << "]"<<"GPU Process Time:" << delta.count() << "s" << std::endl;
	}

	void execute()
	{		
		createCommandBuffer();
		auto restSamples = pushConstantData.totalSamples;
		auto sampleStart = 0;
		while (true)
		{
			if (restSamples >= maxSamplesForSingleShader)
			{
				execute(sampleStart, maxSamplesForSingleShader);
				restSamples -= maxSamplesForSingleShader;
				sampleStart += maxSamplesForSingleShader;
			}
			else if (restSamples > 0)
			{
				execute(sampleStart, restSamples);
				restSamples = 0;
			}
			else 
			{
				std::cout << "total:" << pushConstantData.totalSamples << " Done!" << std::endl;
				break;
			}
		}
		ReadMemory(storageAllocation[0], target.dump.data(), target.DumpSize());
		//void* data;
		//vkMapMemory(device, storageBufferMemorys[0], 0, target.DumpSize(), 0, &data);
		//memcpy(target.dump.data(), data, target.DumpSize());
		//vkUnmapMemory(device, storageBufferMemorys[0]);
		std::filesystem::path out("./RenderingTarget.ppm");
		auto absPath = std::filesystem::absolute(out);
		std::ofstream os(out);
		os << target;
		std::cout << "Output Path:" << absPath << std::endl;
	}

	GLFWwindow* window;
	int WIDTH = 800;
	int HEIGHT = 600;



	void createWindow()
	{
		glfwInit();
		glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
		glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);
		window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan", nullptr, nullptr);
	}

	VkSurfaceKHR surface;
	void createSurface()
	{
		if (glfwCreateWindowSurface(instance, window, nullptr, &surface)
			!= VK_SUCCESS)
			throw std::runtime_error("failed to create surface!");
	}

	VkSwapchainKHR swapchain;
	std::vector<VkImage> swapchainImages;
	void createSwapchain()
	{
		auto support = MuVk::Query::querySwapChainSupport(physicalDevice, surface);
		int width, height;
		glfwGetWindowSize(window, &width, &height);
		auto createInfo = MuVk::Utils::fillSwapchainCreateInfo(support, surface, { (uint32_t)width, (uint32_t)height });
		createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
		createInfo.queueFamilyIndexCount = VK_QUEUE_FAMILY_IGNORED;

		if (vkCreateSwapchainKHR(device, &createInfo, nullptr, &swapchain)
			!= VK_SUCCESS)
			throw std::runtime_error("failed to create swapchain!");

		swapchainImages = MuVk::Utils::getSwapchainImages(device, swapchain);
	}

	void drawFrame()
	{
		uint32_t imageIndex;
		vkAcquireNextImageKHR(device, swapchain, UINT64_MAX, nullptr, nullptr, &imageIndex);
		{
			MuVk::Utils::SingleTimeCommandGuard guard(device, universalQueue, commandPool);
			//vkCmdCopyImage(guard.commandBuffer, storageImage,)
		}
	}

	void mainloop()
	{
		while (!glfwWindowShouldClose(window))
		{
			glfwPollEvents();
			drawFrame();
		}
		vkDeviceWaitIdle(device);
	}

	void Run()
    {
		GreatScene();
		createWindow();
		createInstance();
		createSurface();
		pickPhyscialDevice();
		createLogicalDevice();
		createSwapchain();
		createAllocator();
		//SetupScene();
		
		createBuffers();
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
		vkDestroyCommandPool(device, commandPool, nullptr);
		vkDestroyDescriptorPool(device, descriptorPool, nullptr);
		vkDestroyPipeline(device, computePipeline, nullptr);
		vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
		for (size_t i = 0; i < descriptorSetLayouts.size(); i++)
			vkDestroyDescriptorSetLayout(device, descriptorSetLayouts[i], nullptr);
		
		for (size_t i = 0; i < storageBuffers.size(); i++)
		{
			vmaDestroyBuffer(allocator, storageBuffers[i], nullptr);
			vmaFreeMemory(allocator, storageAllocation[i]);
			//vkDestroyBuffer(device, storageBuffers[i], nullptr);
			//vkFreeMemory(device, storageBufferMemorys[i], nullptr);
		}
		vmaDestroyBuffer(allocator, uniformBuffer, nullptr);
		vmaFreeMemory(allocator, uniformAllocation);
		//vkDestroyBuffer(device, uniformBuffer, nullptr);
		//vkFreeMemory(device, uniformBufferMemory, nullptr);
		vmaDestroyAllocator(allocator);
		
		MuVk::Proxy::destoryDebugUtilsMessengerEXT(instance, debugMessenger, nullptr);
		vkDestroyDevice(device, nullptr);
		vkDestroySurfaceKHR(instance, surface, nullptr);
		vkDestroyInstance(instance, nullptr);
	}

	VmaAllocator allocator;

	void createAllocator()
	{
		VmaAllocatorCreateInfo createInfo = {};
		createInfo.vulkanApiVersion = VK_API_VERSION_1_2;
		createInfo.instance = instance;
		createInfo.device = device;
		createInfo.physicalDevice = physicalDevice;

		if (vmaCreateAllocator(&createInfo, &allocator) != VK_SUCCESS)
		{
			std::runtime_error("failed to create allocator");
		}
	}
};

int main()
{
	ComputeShaderExample program;
	program.Run();
	try
	{
		
	}
	catch (std::runtime_error)
	{
		//program.cleanUp();
	}

}

