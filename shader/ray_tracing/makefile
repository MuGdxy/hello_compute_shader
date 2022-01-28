EntryPoint=ray_tracing.comp
Lib=Random.comp Material.comp Ray.comp Sphere.comp HitRecord.comp Camera.comp
ray_tracing.comp:
	glslangValidator -V ${EntryPoint} ${Lib} -o ${EntryPoint}.spv
clear:
	rm -f ${EntryPoint}.spv