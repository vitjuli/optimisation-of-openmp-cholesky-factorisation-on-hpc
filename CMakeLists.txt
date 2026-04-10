cmake_minimum_required(VERSION 3.16)

project(mphil_dis_cholesky
    VERSION 1.0
    DESCRIPTION "MPhil DIS C2: Optimisation of OpenMP Cholesky Factorisation on CSD3"
    LANGUAGES CXX
)

# ─── C++ standard ────────────────────────────────────────────────────────────
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ─── Build output directories ────────────────────────────────────────────────
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# ─── Version selection ───────────────────────────────────────────────────────
# CHOLESKY_VERSION controls which implementation is compiled:
#   0 = serial baseline  (default; git tag v0_serial_correct)
#   1 = serial optimised (git tag v1_serial_optimised)
#   2 = OpenMP parallel  (git tag v2_openmp_parallel)
#   3 = OpenMP tuned     (git tag v3_column_scaling)
set(CHOLESKY_VERSION "0" CACHE STRING "Cholesky implementation version (0|1|2|3)")
message(STATUS "Building CHOLESKY_VERSION=${CHOLESKY_VERSION}")

# ─── OpenMP (versions 2 and 3 only) ──────────────────────────────────────────
# Baseline and serial-optimised builds (0/1) skip OpenMP entirely, so the build
# works on any machine without libomp installed.
# On macOS with Apple Clang, OpenMP requires libomp from Homebrew.
# On Linux (CSD3 with GCC), -fopenmp is sufficient and find_package works directly.
if("${CHOLESKY_VERSION}" STREQUAL "2" OR "${CHOLESKY_VERSION}" STREQUAL "3")
    if(APPLE)
        execute_process(COMMAND brew --prefix libomp
            OUTPUT_VARIABLE LIBOMP_PREFIX OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET)
        if(LIBOMP_PREFIX)
            message(STATUS "macOS: found libomp at ${LIBOMP_PREFIX}")
            set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp")
            set(OpenMP_CXX_INCLUDE_DIRS "${LIBOMP_PREFIX}/include")
            set(OpenMP_CXX_LIB_NAMES "omp")
            set(OpenMP_omp_LIBRARY "${LIBOMP_PREFIX}/lib/libomp.dylib")
        endif()
    endif()
    find_package(OpenMP REQUIRED)
    message(STATUS "OpenMP version: ${OpenMP_CXX_VERSION}")
    message(STATUS "OpenMP flags:   ${OpenMP_CXX_FLAGS}")
else()
    message(STATUS "OpenMP: skipped (only needed for CHOLESKY_VERSION=2 or 3)")
endif()

# ─── Compiler flags ──────────────────────────────────────────────────────────
# -march=native is OFF by default for portability and reproducibility on CSD3.
# Enable with: cmake -DCHOLESKY_MARCH_NATIVE=ON ...
option(CHOLESKY_MARCH_NATIVE "Tune code for the current CPU's ISA (-march=native)" OFF)

set(CHOLESKY_COMPILE_OPTIONS -O3 -Wall -Wextra)

if(CHOLESKY_MARCH_NATIVE)
    list(APPEND CHOLESKY_COMPILE_OPTIONS -march=native)
    message(STATUS "march=native: enabled")
else()
    message(STATUS "march=native: disabled (use -DCHOLESKY_MARCH_NATIVE=ON to enable)")
endif()

# ─── Production library: cholesky_lib ────────────────────────────────────────
# Contains ONLY src/mphil_dis_cholesky.cpp - no test instrumentation.
# cholesky_example and cholesky_benchmark link against this target only and
# therefore contain no test symbols whatsoever.
add_library(cholesky_lib STATIC
    src/mphil_dis_cholesky.cpp
)

target_include_directories(cholesky_lib
    PUBLIC
        ${PROJECT_SOURCE_DIR}/include
)

# CHOLESKY_VERSION is PUBLIC so consumers (testing hooks, example, benchmark)
# all see the same version definition without having to set it themselves.
# No MPHIL_CHOLESKY_TESTING here - the production library is clean.
target_compile_definitions(cholesky_lib
    PUBLIC
        CHOLESKY_VERSION=${CHOLESKY_VERSION}
)

target_compile_options(cholesky_lib
    PRIVATE
        ${CHOLESKY_COMPILE_OPTIONS}
)

# OpenMP for V2/V3: PUBLIC so the compile flags and link options propagate to
# all consumers (testing hooks, example, benchmark - they all need -fopenmp
# to link a library that was compiled with it, and the testing hooks need the
# compile flags to use OpenMP pragmas in their own TU).
if("${CHOLESKY_VERSION}" STREQUAL "2" OR "${CHOLESKY_VERSION}" STREQUAL "3")
    target_link_libraries(cholesky_lib
        PUBLIC
            OpenMP::OpenMP_CXX
    )
    # On macOS, OpenMP::OpenMP_CXX may not carry the include path automatically.
    if(APPLE AND LIBOMP_PREFIX)
        set_target_properties(OpenMP::OpenMP_CXX PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES "${LIBOMP_PREFIX}/include"
        )
    endif()
endif()

# ─── Test hooks library: cholesky_testing_hooks ─────────────────────────
# Compiles src/mphil_dis_cholesky_testing.cpp - the single-step oracle,
# V2/V3 column-scaling TID recorder, and V1 OPT-1 write-stride probe.
#
# Linked ONLY to cholesky_test and cholesky_test_n2.
# NOT linked to cholesky_example or cholesky_benchmark.
#
# MPHIL_CHOLESKY_TESTING=1 is set PUBLIC so it propagates to the test
# executables, enabling the #ifdef-guarded Group G/I/I2/H2/J blocks in
# test/test_cholesky.cpp.
#
# CHOLESKY_VERSION and OpenMP flags/includes are inherited via PUBLIC linkage
# against cholesky_lib - no duplication needed.
add_library(cholesky_testing_hooks STATIC
    src/mphil_dis_cholesky_testing.cpp
)

target_compile_definitions(cholesky_testing_hooks
    PUBLIC
        # Propagates to test executables, enabling #ifdef MPHIL_CHOLESKY_TESTING
        # guards in test_cholesky.cpp (Groups G, I, I2, H2, J).
        MPHIL_CHOLESKY_TESTING=1
)

target_compile_options(cholesky_testing_hooks
    PRIVATE
        ${CHOLESKY_COMPILE_OPTIONS}
)

# PUBLIC linkage: test executables that link cholesky_testing_hooks also
# transitively link cholesky_lib (and OpenMP for V2/V3).
target_link_libraries(cholesky_testing_hooks
    PUBLIC
        cholesky_lib
)

# ─── Example program ─────────────────────────────────────────────────────────
# Links ONLY cholesky_lib - no hooks, no MPHIL_CHOLESKY_TESTING.
add_executable(cholesky_example
    example/main.cpp
)
target_link_libraries(cholesky_example PRIVATE cholesky_lib)
target_compile_options(cholesky_example PRIVATE ${CHOLESKY_COMPILE_OPTIONS})

# ─── Tests ───────────────────────────────────────────────────────────────────
enable_testing()

# cholesky_test_n2: smoke test (spec 2x2 example).
# Links cholesky_testing_hooks for consistency (brings MPHIL_CHOLESKY_TESTING
# into scope, though test_n2.cpp does not use any testing symbols itself).
add_executable(cholesky_test_n2
    test/test_n2.cpp
)
target_link_libraries(cholesky_test_n2 PRIVATE cholesky_testing_hooks)
target_compile_options(cholesky_test_n2 PRIVATE ${CHOLESKY_COMPILE_OPTIONS})

add_test(NAME CholeskyN2 COMMAND cholesky_test_n2)
set_tests_properties(CholeskyN2 PROPERTIES ENVIRONMENT "OMP_NUM_THREADS=1")

# cholesky_test: full suite (Groups A–K).
# Links cholesky_testing_hooks, which brings:
#   - MPHIL_CHOLESKY_TESTING=1 (enables Groups G, I, I2, H2, J in source)
#   - cholesky_lib (and OpenMP for V2/V3) transitively
# The thread-invariance (Group H) and V3 parallelism (H2) tests call
# omp_set_num_threads() internally to override OMP_NUM_THREADS=1 from ctest.
add_executable(cholesky_test
    test/test_cholesky.cpp
)
target_link_libraries(cholesky_test PRIVATE cholesky_testing_hooks)
target_compile_options(cholesky_test PRIVATE ${CHOLESKY_COMPILE_OPTIONS})

add_test(NAME CholeskyTestSuite COMMAND cholesky_test)
set_tests_properties(CholeskyTestSuite PROPERTIES ENVIRONMENT "OMP_NUM_THREADS=1")

# ─── Benchmark ────────────────────────────────────────────────────────────────
# Links ONLY cholesky_lib - no hooks, no MPHIL_CHOLESKY_TESTING.
# cholesky_benchmark does not contain any test symbols.
# Verify: nm -g build/bin/cholesky_benchmark | grep mphil_dis_cholesky_step
# should return nothing.
add_executable(cholesky_benchmark
    benchmark/benchmark.cpp
)
target_link_libraries(cholesky_benchmark PRIVATE cholesky_lib)
target_compile_options(cholesky_benchmark PRIVATE ${CHOLESKY_COMPILE_OPTIONS})

# Doxygen documentation ──────────────────────────────────────────
find_package(Doxygen QUIET)
if(DOXYGEN_FOUND)
    set(DOXYFILE "${PROJECT_SOURCE_DIR}/docs/Doxyfile")
    add_custom_target(docs
        COMMAND ${DOXYGEN_EXECUTABLE} "${DOXYFILE}"
        WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
        COMMENT "Generating HTML documentation with Doxygen → docs/html/index.html"
        VERBATIM
    )
    message(STATUS "Doxygen ${DOXYGEN_VERSION} found - build target 'docs' available")
else()
    message(STATUS "Doxygen not found - 'docs' target unavailable (install doxygen to enable)")
endif()
