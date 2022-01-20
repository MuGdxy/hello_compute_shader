#pragma once
#include <vulkan/vulkan.h>
#include <vector>
#include <iostream>
//#include <functional>
namespace MuVk
{
	struct Query
	{
		static std::vector<VkExtensionProperties> instanceExtensionProperties()
		{
			uint32_t extensionCount = 0;
			vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, nullptr);
			std::vector<VkExtensionProperties> ext(extensionCount);
			vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, ext.data());
			return ext;
		}

		static std::vector<VkPhysicalDevice> physicalDevices(VkInstance instance)
		{
			uint32_t deviceCount;
			vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);
			std::vector<VkPhysicalDevice> devices(deviceCount);
			vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());
			return devices;
		}

		static VkPhysicalDeviceProperties physicalDeviceProperties(VkPhysicalDevice device)
		{
			VkPhysicalDeviceProperties properties;
			vkGetPhysicalDeviceProperties(device, &properties);
			return properties;
		}

		static VkPhysicalDeviceFeatures physicalDevicePropertiesfeatures(VkPhysicalDevice device)
		{
			VkPhysicalDeviceFeatures features;
			vkGetPhysicalDeviceFeatures(device, &features);
			return features;
		}

		static std::vector<VkQueueFamilyProperties> queueFamilies(VkPhysicalDevice device)
		{
			uint32_t propertyCount;
			vkGetPhysicalDeviceQueueFamilyProperties(device, &propertyCount, nullptr);
			std::vector<VkQueueFamilyProperties> queueFamilies(propertyCount);
			vkGetPhysicalDeviceQueueFamilyProperties(device, &propertyCount, queueFamilies.data());
			return queueFamilies;
		}

		static VkMemoryRequirements memoryRequirements(VkDevice device, VkBuffer buffer)
		{
			VkMemoryRequirements memoryRequirements;
			vkGetBufferMemoryRequirements(device, buffer, &memoryRequirements);
			return memoryRequirements;
		}

		static VkPhysicalDeviceMemoryProperties physicalDeviceMemoryProperties(VkPhysicalDevice device)
		{
			VkPhysicalDeviceMemoryProperties memoryProperties;
			vkGetPhysicalDeviceMemoryProperties(device, &memoryProperties);
			return memoryProperties;
		}

	};
}

static struct tabs
{
	int t;
	tabs(int i):t(i){}

	friend std::ostream& operator << (std::ostream& o, const tabs& t)
	{
		for (int i = 0; i < t.t; ++i) o << '\t';
		return o;
	}
};

inline std::ostream& operator << (std::ostream& o, const std::vector<VkExtensionProperties>& properties)
{
	o << "available extensions:";
	tabs t(1);
	for (const auto& extension : properties)
	{
		std::cout
			<< "\n"<< t << extension.extensionName
			<< "<Version=" << extension.specVersion << ">";
	}
	return o;
}

inline std::ostream& operator << (std::ostream& o, VkPhysicalDevice device)
{
	auto properties = MuVk::Query::physicalDeviceProperties(device);
	o << properties.deviceName;
	return o;
}

inline std::ostream& operator << (std::ostream& o, const std::vector<VkPhysicalDevice>& devices)
{
	o << "physical devices:";
	int index = 0;
	tabs t(1);
	for (const auto& device : devices)
		std::cout << "\n" << t << "["<< index++ << "] " << device;
	return o;
}

inline std::ostream& operator << (std::ostream& o, const std::vector<VkQueueFamilyProperties>& families)
{
	o << "queue families:";
	auto presentFlags = [](std::ostream& o, VkQueueFlags flags)
	{

		o << tabs(1) << "Flags:";
		if (flags & VkQueueFlagBits::VK_QUEUE_GRAPHICS_BIT) o << "[Graphics]";
		if (flags & VkQueueFlagBits::VK_QUEUE_COMPUTE_BIT) o << "[Compute]";
		if (flags & VkQueueFlagBits::VK_QUEUE_TRANSFER_BIT) o << "[Transfer]";
		if (flags & VkQueueFlagBits::VK_QUEUE_SPARSE_BINDING_BIT) o << "[Sparse Binding]";
	};
	int index = 0;
	tabs t(1);
	for (const auto& family : families)
	{
		o << "\n" << t << "family index=" << index++;
		o << "\n" << t << "Queue Count=" << family.queueCount << "\n";
		presentFlags(o, family.queueFlags);
	}
	return o;
}

inline std::ostream& operator << (std::ostream& o, const VkMemoryRequirements& requirements)
{
	tabs t(1);
	o << "memory requirements:\n";
	o << t << "alignment=" << requirements.alignment << "\n";
	o << t << "size=" << requirements.size << "\n";
	o << t << "type bits=0x" << std::uppercase << std::hex << requirements.memoryTypeBits << std::dec;
	return o;
}

inline std::ostream& operator << (std::ostream& o, const VkPhysicalDeviceMemoryProperties& properties)
{
	auto presentType = [](std::ostream& o, const VkMemoryType& type)
	{
		tabs t(1);
		o << t << "heap index=" << type.heapIndex << "\n";
		o << t << "propertyFlags: ";
		if (type.propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) o << "[Device Local]";
		if (type.propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) o << "[Host Visible]";
		if (type.propertyFlags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) o << "[Host Coherent]";
		if (type.propertyFlags & VK_MEMORY_PROPERTY_HOST_CACHED_BIT) o << "[Host Cached]";
		if (type.propertyFlags & VK_MEMORY_PROPERTY_LAZILY_ALLOCATED_BIT) o << "[Lazily Allocated]";
	};

	auto presentHeap = [](std::ostream& o, const VkMemoryHeap& heap)
	{
		tabs t(1);
		o << t << "heap size=" << heap.size << "\n";
		o << t << "heap flags: ";
		if (heap.flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) o << "[Device Local]";
		if (heap.flags & VK_MEMORY_HEAP_MULTI_INSTANCE_BIT) o << "[Multi Instance]";
	};

	o << "memory type:\n";
	for (size_t i = 0; i < properties.memoryTypeCount; ++i)
	{
		o << "[" << i << "]:\n";
		presentType(o, properties.memoryTypes[i]);
		o << std::endl;
	}
	o << "memory heap:\n";
	for (size_t i = 0; i < properties.memoryHeapCount; ++i)
	{
		o << "[" << i << "]:\n";
		presentHeap(o, properties.memoryHeaps[i]);
		o << std::endl;
	}
	return o;
}