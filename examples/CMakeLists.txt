cmake_minimum_required(VERSION 3.20)
project(hybit_ffi_examples LANGUAGES C CXX Fortran)

set(HYBIT_ROOT "${CMAKE_CURRENT_LIST_DIR}/.." CACHE PATH "HyBIT source root")
set(HYBIT_LIB_DIR "${HYBIT_ROOT}/target/release" CACHE PATH "Directory containing built HyBIT library")
set(HYBIT_DLL "${HYBIT_LIB_DIR}/hybit.dll")

if(WIN32)
    if(NOT EXISTS "${HYBIT_DLL}")
        message(FATAL_ERROR
            "${HYBIT_DLL} was not found. Build the Rust C-ABI DLL first with:\n"
            "  cargo build --release -p hybit-ffi")
    endif()

    # Rust's default Windows toolchain is commonly MSVC, while the examples may
    # be compiled with MinGW GCC/G++/GFortran.  Their import-library formats are
    # different.  Keep the DLL as the single ABI boundary and create the import
    # library expected by the active compiler.
    if(MINGW)
        find_program(HYBIT_DLLTOOL NAMES dlltool x86_64-w64-mingw32-dlltool REQUIRED)
        set(HYBIT_DEF "${HYBIT_ROOT}/include/hybit.def")
        set(HYBIT_IMPORT_LIB "${CMAKE_CURRENT_BINARY_DIR}/libhybit.dll.a")

        execute_process(
            COMMAND "${HYBIT_DLLTOOL}"
                    -d "${HYBIT_DEF}"
                    -D hybit.dll
                    -l "${HYBIT_IMPORT_LIB}"
            RESULT_VARIABLE HYBIT_DLLTOOL_RESULT
            OUTPUT_VARIABLE HYBIT_DLLTOOL_STDOUT
            ERROR_VARIABLE HYBIT_DLLTOOL_STDERR
        )
        if(NOT HYBIT_DLLTOOL_RESULT EQUAL 0)
            message(FATAL_ERROR
                "dlltool failed while creating the MinGW import library.\n"
                "stdout: ${HYBIT_DLLTOOL_STDOUT}\n"
                "stderr: ${HYBIT_DLLTOOL_STDERR}")
        endif()

        add_library(HyBIT::hybit SHARED IMPORTED GLOBAL)
        set_target_properties(HyBIT::hybit PROPERTIES
            IMPORTED_LOCATION "${HYBIT_DLL}"
            IMPORTED_IMPLIB "${HYBIT_IMPORT_LIB}"
        )
        message(STATUS "HyBIT: generated MinGW import library ${HYBIT_IMPORT_LIB}")
    elseif(MSVC)
        # cdylib built with the Rust MSVC target emits hybit.dll.lib.
        set(HYBIT_IMPORT_LIB "${HYBIT_LIB_DIR}/hybit.dll.lib")
        if(NOT EXISTS "${HYBIT_IMPORT_LIB}")
            message(FATAL_ERROR
                "${HYBIT_IMPORT_LIB} was not found. Ensure Rust uses the MSVC Windows target.")
        endif()
        add_library(HyBIT::hybit SHARED IMPORTED GLOBAL)
        set_target_properties(HyBIT::hybit PROPERTIES
            IMPORTED_LOCATION "${HYBIT_DLL}"
            IMPORTED_IMPLIB "${HYBIT_IMPORT_LIB}"
        )
    else()
        message(FATAL_ERROR "Unsupported Windows C/C++ toolchain for the examples")
    endif()
else()
    find_library(HYBIT_LIBRARY NAMES hybit libhybit PATHS "${HYBIT_LIB_DIR}" NO_DEFAULT_PATH REQUIRED)
    add_library(HyBIT::hybit SHARED IMPORTED GLOBAL)
    set_target_properties(HyBIT::hybit PROPERTIES IMPORTED_LOCATION "${HYBIT_LIBRARY}")
endif()

function(hybit_attach_runtime target)
    target_link_libraries(${target} PRIVATE HyBIT::hybit)
    if(WIN32)
        add_custom_command(TARGET ${target} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    "${HYBIT_DLL}"
                    "$<TARGET_FILE_DIR:${target}>/hybit.dll"
            VERBATIM)
    endif()
endfunction()

add_executable(hybit_c c/main.c)
target_include_directories(hybit_c PRIVATE "${HYBIT_ROOT}/include")
hybit_attach_runtime(hybit_c)

add_executable(hybit_cpp cpp/main.cpp)
target_compile_features(hybit_cpp PRIVATE cxx_std_17)
target_include_directories(hybit_cpp PRIVATE "${HYBIT_ROOT}/include")
if(MINGW)
    # Keep the public ABI boundary at hybit.dll while avoiding accidental use
    # of an incompatible libstdc++/libgcc DLL found earlier on PATH.
    target_link_options(hybit_cpp PRIVATE -static-libstdc++ -static-libgcc)
endif()
hybit_attach_runtime(hybit_cpp)

add_executable(hybit_fortran "${HYBIT_ROOT}/fortran/hybit.f90" fortran/main.f90)
if(MINGW)
    # Fortran examples should not depend on a particular dynamic GNU runtime
    # installation.  The HyBIT DLL remains dynamically linked.
    target_link_options(hybit_fortran PRIVATE -static-libgfortran -static-libgcc)
endif()
hybit_attach_runtime(hybit_fortran)
