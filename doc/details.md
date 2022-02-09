The steps for rendering to an image from a compute shader:

- Create your compute target image with the VK_IMAGE_USAGE_STORAGE_BIT usage flag (check if the format supports that first)
- Create and assign a descriptor of type VK_DESCRIPTOR_TYPE_STORAGE_IMAGE for that image for the compute pass
- Add a binding for that image in your compute shader that fits the descriptor (e.g. `layout ( binding = 0, rgba8 ) uniform writeonly image2D resultImage`)

- Create and assign a descriptor of type VK_DESCRIPTOR_TYPE_COMBINED_IMAGE (or separate sampler if you want) for that image for the rendering (sampling) pass
- Write to the image in a compute command buffer submitted to the compute queue
- Do proper synchronization (ensure that comptue shader writes are finished before sample)
- Sample from the image in your graphics command buffer



Write Image from compute shader and copy to swapchain image:

- Create your compute target image with the `VK_IMAGE_USAGE_STORAGE_BIT` **usage flag** (check if the format supports that first)
- Create and assign a **descriptor** of type `VK_DESCRIPTOR_TYPE_STORAGE_IMAGE` for that image for the compute pass
- Add a binding for that image in your compute shader that fits the descriptor (e.g. `layout ( binding = 0, rgba8 ) uniform writeonly image2D resultImage`)
- using `vkCmdCopyImage`  to copy image to swapchain image.