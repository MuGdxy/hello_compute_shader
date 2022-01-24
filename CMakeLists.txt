cmake_minimum_required(VERSION 3.1)
project(HelloComputeShader)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED true)

include_directories(${CMAKE_SOURCE_DIR}/include)
link_directories(${CMAKE_SOURCE_DIR}/libs)
file(GLOB MuVkHeaders "${CMAKE_SOURCE_DIR}/include/MuVk/*.h")

message("Headers:")
foreach(MuVkHeader ${MuVkHeaders})
    message(${MuVkHeader})
endforeach()

include_directories(
    "${CMAKE_CURRENT_SOURCE_DIR}/include/"
)

add_definitions(-DMU_SHADER_PATH="${CMAKE_SOURCE_DIR}/shader/")
link_libraries("vulkan-1.lib")
add_executable(
    MultiplyComputeShader
    "src/Multiply.cpp"
    ${MuVkHeaders}
)

add_executable(
    MultiplyComputeShaderHomework
    "homework/Multiply.cpp"
    ${MuVkHeaders}
)

add_executable(
    ViewID
    "src/ViewID.cpp"
    ${MuVkHeaders}
)

add_executable(
    PhysicalDeviceProperties
    "src/PhysicalDeviceProperties.cpp"
    ${MuVkHeaders}
)

add_executable(
    ComputeShaderRayTracing
    "src/ComputeShaderRayTracing/main.cpp"
    ${MuVkHeaders}
)







