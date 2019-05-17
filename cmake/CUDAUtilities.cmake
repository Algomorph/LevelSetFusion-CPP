if (${CMAKE_VERSION} VERSION_LESS 3.8 OR (MSVC_IDE AND ${CMAKE_VERSION} VERSION_LESS 3.9))
    find_package(CUDA QUIET)
else ()
    # Use the new CMake mechanism wich enables full Nsight support
    include(CheckLanguage)
    check_language(CUDA)
    set(CUDA_FOUND ${CMAKE_CUDA_COMPILER})
endif ()