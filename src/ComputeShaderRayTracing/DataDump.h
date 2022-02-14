#pragma once
#include <vector>
#include <glm/glm.hpp>
#include <vulkan/vulkan.h>

using color = glm::vec3;
using point3 = glm::vec3;
using vec3 = glm::vec3;

class TargetBuffer
{
public:
	size_t width;
	size_t height;
	bool gammaCorrectOnOutput = true;
	TargetBuffer() {};

	TargetBuffer(size_t width, size_t height)
		:width(width), height(height)
	{
		dump.resize(width * height, glm::vec4(0));
	}

	friend std::ostream& operator << (std::ostream& out, TargetBuffer& buffer)
	{
		out << "P3\n" << buffer.width << ' ' << buffer.height << "\n255\n";
		for (auto pixel : buffer.dump)
		{
			pixel.w = 1.0f;
			if (buffer.gammaCorrectOnOutput) pixel = glm::sqrt(pixel);
			out << (int)(256 * glm::clamp(pixel.x, 0.0f, 0.999f)) << ' '
				<< (int)(256 * glm::clamp(pixel.y, 0.0f, 0.999f)) << ' '
				<< (int)(256 * glm::clamp(pixel.z, 0.0f, 0.999f)) << '\n';
		}
		return out;
	}

	VkDeviceSize DumpSize() { return dump.size() * sizeof(glm::vec4); }
	std::vector<glm::vec4> dump;
};

struct Camera
{
	//:0
	alignas(16) glm::vec3 origin;
	alignas(16) glm::vec3 horizontal;
	alignas(16) glm::vec3 vertical;
	alignas(16) glm::vec3 lowerLeftCorner;
	//:4
	float viewportHeight;
	float viewportWidth;
	float aspectRatio;
	float focalLength;
	//:5
	alignas(16) glm::vec3 u;
	alignas(16) glm::vec3 v;
	alignas(16) glm::vec3 w;
	float lensRadius;
public:
	Camera() = default;
	Camera(
		point3 lookfrom,
		point3 lookat,
		vec3   vup,
		float vfov, // vertical field-of-view in degrees
		float aspectRatio,
		float aperture,
		float focus_dist)
	{
		const auto offset = offsetof(Camera, viewportHeight);
		float theta = glm::radians(vfov);
		float h = glm::tan(theta / 2);
		viewportHeight = 2.0f * h;
		viewportWidth = aspectRatio * viewportHeight;
		this->aspectRatio = aspectRatio;

		w = glm::normalize(lookfrom - lookat);
		u = glm::normalize(cross(vup, w));
		v = glm::cross(w, u);

		origin = lookfrom;
		horizontal = focus_dist * viewportWidth * u;
		vertical = focus_dist * viewportHeight * v;
		lowerLeftCorner = origin - horizontal / 2.0f - vertical / 2.0f - w * focus_dist;
		focalLength = focus_dist;
		lensRadius = aperture / 2.0f;
	}

};

struct PushConstantData
{
	glm::ivec2 screenSize;
	uint32_t hittableCount;
	uint32_t sampleStart;
	uint32_t samples;
	uint32_t totalSamples;
	uint32_t maxDepth;
};

template<typename Ty, typename GlslTy, typename DumpTy>
class DataDump
{
protected:
	std::vector<Ty*> handles;
public:
	std::vector<GlslTy> heads;
	std::vector<DumpTy> dump;
	DataDump() = default;
	template<typename Detrive, typename ...Args>
	Detrive* Allocate(Args&&... args)
	{
		static_assert(std::is_base_of_v<Ty, Detrive>, "need to derive from Ty");
		auto ret = new Detrive(std::forward<Args>(args)...);
		ret->ptr = handles.size();
		handles.push_back(ret);
		return ret;
	}
	void Clear()
	{
		for (auto handle : handles) delete handle;
		handles.clear();
	}
	~DataDump()
	{
		Clear();
	}
	uint32_t HeadSize()
	{
		return heads.size() * sizeof(GlslTy);
	}
	uint32_t DumpSize()
	{
		return dump.size() * sizeof(DumpTy);
	}

	void WriteMemory(VkDevice device, VkDeviceMemory headMemory, VkDeviceMemory dumpMemory)
	{
		WriteMemory(device, headMemory, heads.data(), HeadSize());
		WriteMemory(device, dumpMemory, dump.data(), DumpSize());
	}
private:
	void WriteMemory(VkDevice device, VkDeviceMemory memory, void* dataBlock, VkDeviceSize size)
	{
		void* data;
		if (vkMapMemory(device, memory, 0, size, 0, &data) != VK_SUCCESS)
			throw std::runtime_error("failed to map memory");
		memcpy(data, dataBlock, size);
		vkUnmapMemory(device, memory);
	}
};

enum class MaterialType : uint32_t
{
	None = 0, Lambertian, Metal, Dielectrics
};

struct GLSLMaterial
{
	MaterialType type = MaterialType::None;
	uint32_t ptr;
};

class Material
{
	friend class MaterialDump;
	friend class HittableDump;
	friend class DataDump<Material, GLSLMaterial, glm::vec4>;
public:
	MaterialType type;
	virtual std::vector<glm::vec4> Dump() = 0;
private:
	uint32_t ptr;
};

class Lambertian : public Material
{
public:
	Lambertian(const glm::vec3 albedo)
		:albedo(albedo)
	{
		this->type = MaterialType::Lambertian;
	}
	glm::vec3 albedo;
	virtual std::vector<glm::vec4> Dump() override
	{
		return { glm::vec4(albedo, 0.0) };
	}
};

class Metal :public Material
{
public:
	Metal(const glm::vec3& a, float fuzz) :albedo(a), fuzz(fuzz < 1 ? fuzz : 1) 
	{
		this->type = MaterialType::Metal;
	};
	virtual std::vector<glm::vec4> Dump() override
	{
		return { glm::vec4(albedo, fuzz) };
	}
	glm::vec3 albedo;
	float fuzz;
};

class Dielectric :public Material
{
public:
	Dielectric(float ir) :ir(ir) 
	{
		this->type = MaterialType::Dielectrics;
	}
	float ir;
	virtual std::vector<glm::vec4> Dump() override
	{
		return { glm::vec4(ir, 0, 0, 0) };
	}
};

class MaterialDump: public DataDump<Material,GLSLMaterial,glm::vec4>
{
public:
	void Dump()
	{
		heads.clear();
		heads.reserve(handles.size());
		for (auto handle : handles)
		{
			heads.push_back({ handle->type,handle->ptr });
			auto vec4s = handle->Dump();
			for (const auto& v : vec4s) dump.push_back(v);
		}
	}
};

enum class HittableType : uint32_t
{
	None = 0, TriangleMesh, Sphere
};

struct GLSLHittable
{
	HittableType type = HittableType::None;
	uint32_t ptr;
	uint32_t mat = 0;
};

class Hittable
{
	friend class HittableDump;
	friend class DataDump<Hittable, GLSLHittable, glm::vec4>;
public:
	HittableType type;
	virtual std::vector<glm::vec4> Dump() = 0;
	Material* mat;
private:
	uint32_t ptr;
};

class Sphere : public Hittable
{
public:
	Sphere(const glm::vec3& center, float radius)
		:center(center), radius(radius)
	{
		this->type = HittableType::Sphere;
	}
	glm::vec3 center;
	float radius;
	virtual std::vector<glm::vec4> Dump() override
	{
		return { glm::vec4(center, radius) };
	}
};

class HittableDump : public DataDump<Hittable, GLSLHittable, glm::vec4>
{
public:
	uint32_t Count() { return handles.size(); }
	void Dump()
	{
		heads.clear();
		heads.reserve(handles.size());
		for (auto handle : handles)
		{
			heads.push_back({ handle->type, handle->ptr, handle->mat->ptr });
			auto vec4s = handle->Dump();
			for (const auto& v : vec4s) dump.push_back(v);
		}
	}
};