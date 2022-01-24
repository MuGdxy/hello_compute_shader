#include <iostream>
#include <optional>
#include "MuVk/MuVK.h"
#include <array>
#include<glm/glm.hpp>

#ifndef MU_SHADER_PATH
#define MU_SHADER_PATH "./shader/"
#endif

class TargetBuffer
{
public:
	int samples_per_pixel;
	glm::ivec2 offset{ 0 };
	size_t width;
	size_t height;
	TargetBuffer() {};
	void RestructureFromDamp()
	{
		for (size_t y = 0; y < height; ++y)
			for (size_t x = 0; x < width; ++x)
			{
				write_color(x, y, glm::vec3(damp[y * width + x]));
				//std::cout << damp[y * width + x].x << "," << damp[y * width + x].y << std::endl;
			}
				
	}
	TargetBuffer(size_t width, size_t height, int samples_per_pixel)
		:width(width), height(height), samples_per_pixel(samples_per_pixel)
	{
		buf.resize(height);
		for (size_t i = 0; i < height; ++i)
			buf[i].resize(width);
		damp.resize(width * height, glm::vec4(0));
	}
	void write_color(size_t x, size_t y, glm::vec3 pixel_color)
	{
		x -= offset.x;
		y -= offset.y;
		auto r = pixel_color.x;
		auto g = pixel_color.y;
		auto b = pixel_color.z;

		// Divide the color by the number of samples.
		auto scale = 1.0 / samples_per_pixel;
		r = std::sqrt(scale * r);
		g = std::sqrt(scale * g);
		b = std::sqrt(scale * b);

		buf[y][x] = glm::ivec3
		{
			256 * glm::clamp(r, 0.0f, 0.999f),
			256 * glm::clamp(g, 0.0f, 0.999f),
			256 * glm::clamp(b, 0.0f, 0.999f)
		};
		//buf[y][x] = pixel_color;
	}
	void write(TargetBuffer& sub_buf)
	{
		if (sub_buf.offset.x + sub_buf.width > width
			|| sub_buf.offset.y + sub_buf.height > height)
			throw std::exception("out of range");
		glm::ivec2 offset_ = sub_buf.offset - offset;
		for (size_t i = 0; i < sub_buf.height; ++i)
			for (size_t j = 0; j < sub_buf.width; ++j)
			{
				buf[i + offset_.y][j + offset_.x] = sub_buf.buf[i][j];
			}
	}
	friend std::ostream& operator << (std::ostream& out, TargetBuffer& buffer)
	{
		out << "P3\n" << buffer.width << ' ' << buffer.height << "\n255\n";
		for (const auto& line : buffer.buf)
			for (const auto& pixel : line)
			{
				out << pixel.r << ' '
					<< pixel.g << ' '
					<< pixel.b << '\n';
			}
		return out;
	}
	VkDeviceSize DampSize() { return damp.size() * sizeof(glm::vec4); }
	std::vector<glm::vec4> damp;
private:
	std::vector<std::vector<glm::ivec3>> buf;
};

struct Camera
{
	//:0
	alignas(16) glm::vec3 origin;
	alignas(16) glm::vec3 horizontal;
	alignas(16) glm::vec3 vertical;
	alignas(16) glm::vec3 lowerLeftCorner;
	//:4
	alignas(4) float viewportHeight;
	alignas(4) float viewportWidth;
	alignas(4) float aspectRatio;
	alignas(4) float focalLength;
};

class ComputeShaderExample
{
	constexpr uint32_t computeShaderProcessUnit() { return 32; }

	TargetBuffer target;
	Camera camera;
	glm::ivec2 screenSize;
public:
	ComputeShaderExample()
	{
		const auto aspectRatio = 16.0 / 9.0;
		const int imageWidth = 800;
		const int imageHeight = static_cast<int>(imageWidth / aspectRatio);

		target = TargetBuffer(imageWidth, imageHeight, 1);
		camera.viewportHeight = 2.0;
		camera.aspectRatio = aspectRatio;
		camera.viewportWidth = aspectRatio * camera.viewportHeight;
		camera.focalLength = 1.0;

		camera.origin = glm::vec3(0, 0, 0);
		camera.horizontal = glm::vec3(camera.viewportWidth, 0, 0);
		camera.vertical = glm::vec3(0, camera.viewportHeight, 0);
		camera.lowerLeftCorner = camera.origin - camera.horizontal / 2.0f - camera.vertical / 2.0f - glm::vec3(0, 0, camera.focalLength);
		
		screenSize.x = imageWidth;
		screenSize.y = imageHeight;
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
		auto p = MuVk::Query::physicalDeviceProperties(physicalDevice);
		std::cout << "maxComputeWorkGroupInvocations:" << p.limits.maxComputeWorkGroupInvocations << std::endl;
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

	VkBuffer uniformBuffer;
	VkDeviceMemory uniformBufferMemory;
	void createBuffer(VkBufferUsageFlags usage, VkBuffer& buffer, VkDeviceMemory& memory)
	{
		VkBufferCreateInfo createInfo = MuVk::bufferCreateInfo();
		createInfo.size = target.DampSize();
		createInfo.usage = usage;
		createInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
		createInfo.queueFamilyIndexCount = 1;
		createInfo.pQueueFamilyIndices = &computeTransferQueueFamilyIndex.value();
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

	void createBuffers()
	{
		createBuffer(VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, storageBuffer, storageBufferMemory);
		createBuffer(VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, uniformBuffer, uniformBufferMemory);
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

	void writeMemoryFromHost()
	{
		WriteMemory(storageBufferMemory, target.damp.data(), target.DampSize());
		WriteMemory(uniformBufferMemory, &camera, sizeof(camera));
	}

	VkDescriptorSetLayout descriptorSetLayout;
	void createDescriptorSetLayout()
	{
		std::array<VkDescriptorSetLayoutBinding,2> bindings;
		bindings[0].binding = 0;
		bindings[0].descriptorCount = 1;
		bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		bindings[0].pImmutableSamplers = nullptr;
		bindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
		bindings[1].binding = 1;
		bindings[1].descriptorCount = 1;
		bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		bindings[1].pImmutableSamplers = nullptr;
		bindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

		VkDescriptorSetLayoutCreateInfo createInfo = MuVk::descriptorSetLayoutCreateInfo();
		createInfo.bindingCount = bindings.size();
		createInfo.pBindings = bindings.data();

		if (vkCreateDescriptorSetLayout(
			device, &createInfo, nullptr, &descriptorSetLayout) != VK_SUCCESS)
			throw std::runtime_error("failed to create descriptorSetLayout");
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
		auto computeShaderCode = MuVk::readFile(MU_SHADER_PATH "ray_tracing.comp.spv");
		auto computeShaderModule = createShaderModule(computeShaderCode);
		VkPipelineShaderStageCreateInfo shaderStageCreateInfo = MuVk::pipelineShaderStageCreateInfo();
		shaderStageCreateInfo.module = computeShaderModule;
		shaderStageCreateInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
		shaderStageCreateInfo.pName = "main";

		VkPushConstantRange range{};
		range.offset = 0;
		range.size = sizeof(glm::ivec2);
		range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

		VkPipelineLayoutCreateInfo layoutCreateInfo = MuVk::pipelineLayoutCreateInfo();
		layoutCreateInfo.setLayoutCount = 1;
		layoutCreateInfo.pSetLayouts = &descriptorSetLayout;
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
		poolSize[0].descriptorCount = 1;
		poolSize[0].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;

		poolSize[1].descriptorCount = 1;
		poolSize[1].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;


		VkDescriptorPoolCreateInfo createInfo = MuVk::descriptorPoolCreateInfo();
		createInfo.poolSizeCount = 2;
		createInfo.pPoolSizes = poolSize.data();
		createInfo.maxSets = 1;
		
		if (vkCreateDescriptorPool(device, &createInfo, nullptr, &descriptorPool)
			!= VK_SUCCESS)
			throw std::runtime_error("failed to create descriptor pool!");
	}

	VkDescriptorSet descriptorSet;
	void createDescriptorSet()
	{
		VkDescriptorSetAllocateInfo allocInfo = MuVk::descriptorSetAllocateInfo();
		allocInfo.descriptorPool = descriptorPool;
		allocInfo.descriptorSetCount = 1;
		allocInfo.pSetLayouts = &descriptorSetLayout;

		if (vkAllocateDescriptorSets(device, &allocInfo, &descriptorSet)
			!= VK_SUCCESS)
			throw std::runtime_error("failed to create descriptor set!");

		VkDescriptorBufferInfo storageBufferInfo;
		storageBufferInfo.buffer = storageBuffer;
		storageBufferInfo.offset = 0;
		storageBufferInfo.range = target.DampSize();

		VkDescriptorBufferInfo uniformBufferInfo;
		uniformBufferInfo.buffer = uniformBuffer;
		uniformBufferInfo.offset = 0;
		uniformBufferInfo.range = sizeof(camera);

		VkWriteDescriptorSet write = MuVk::writeDescriptorSet();
		write.descriptorCount = 1;
		write.dstSet = descriptorSet;
		write.dstArrayElement = 0;
		std::array writes = { write, write };
		writes[0].dstBinding = 0;
		writes[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		writes[0].pBufferInfo = &storageBufferInfo;
		writes[1].dstBinding = 1;
		writes[1].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		writes[1].pBufferInfo = &uniformBufferInfo;
		
		vkUpdateDescriptorSets(device, writes.size(), writes.data(), 0, nullptr);
	}

	VkCommandPool commandPool;
	void createCommandPool()
	{
		VkCommandPoolCreateInfo createInfo = MuVk::commandPoolCreateInfo();
		createInfo.queueFamilyIndex = computeTransferQueueFamilyIndex.value();
		if (vkCreateCommandPool(device, &createInfo, nullptr, &commandPool)
			!= VK_SUCCESS)
			throw std::runtime_error("failed to create command pool!");
	}


	VkCommandBuffer commandBuffer;
	void execute()
	{
		VkCommandBufferAllocateInfo allocInfo = MuVk::commandBufferAllocateInfo();
		allocInfo.commandBufferCount = 1;
		allocInfo.commandPool = commandPool;
		allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
		if (vkAllocateCommandBuffers(device, &allocInfo, &commandBuffer)
			!= VK_SUCCESS)
			throw std::runtime_error("failed to create command buffer!");
		
		VkCommandBufferBeginInfo beginInfo = MuVk::commandBufferBeginInfo();
		vkBeginCommandBuffer(commandBuffer, &beginInfo);
		vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, computePipeline);
		vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout,
			0, 1, &descriptorSet, 0, nullptr);

		vkCmdPushConstants(commandBuffer, pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT,
			0, sizeof(screenSize), &screenSize);
		auto sizeX = static_cast<uint32_t>(target.width / computeShaderProcessUnit() + 1);
		auto sizeY = static_cast<uint32_t>(target.height / computeShaderProcessUnit() + 1);
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
		if (vkQueueSubmit(computeTransferQueue, 1, &submitInfo, VK_NULL_HANDLE)
			!= VK_SUCCESS)
			throw std::runtime_error("failed to submit command buffer!");

		//wait the calculation to finish
		if (vkQueueWaitIdle(computeTransferQueue) != VK_SUCCESS)
			throw std::runtime_error("failed to wait queue idle!");
		void* data;
		vkMapMemory(device, storageBufferMemory, 0, target.DampSize(), 0, &data);
		memcpy(target.damp.data(), data, target.DampSize());
		vkUnmapMemory(device, storageBufferMemory);
		target.RestructureFromDamp();
		std::ofstream os("./RenderingTarget.ppm");
		os << target;
	}

	void Run()
    {
		createInstance();
		pickPhyscialDevice();
		createLogicalDevice();

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
		vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);

		vkDestroyBuffer(device, storageBuffer, nullptr);
		vkFreeMemory(device, storageBufferMemory, nullptr);

		vkDestroyBuffer(device, uniformBuffer, nullptr);
		vkFreeMemory(device, uniformBufferMemory, nullptr);

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

