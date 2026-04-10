# CMake generated Testfile for 
# Source directory: /Users/julia/Desktop/courses/c2_claude/gitlab
# Build directory: /Users/julia/Desktop/courses/c2_claude/gitlab/build_v1
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(CholeskyN2 "/Users/julia/Desktop/courses/c2_claude/gitlab/build_v1/bin/cholesky_test_n2")
set_tests_properties(CholeskyN2 PROPERTIES  ENVIRONMENT "OMP_NUM_THREADS=1" _BACKTRACE_TRIPLES "/Users/julia/Desktop/courses/c2_claude/gitlab/CMakeLists.txt;166;add_test;/Users/julia/Desktop/courses/c2_claude/gitlab/CMakeLists.txt;0;")
add_test(CholeskyTestSuite "/Users/julia/Desktop/courses/c2_claude/gitlab/build_v1/bin/cholesky_test")
set_tests_properties(CholeskyTestSuite PROPERTIES  ENVIRONMENT "OMP_NUM_THREADS=1" _BACKTRACE_TRIPLES "/Users/julia/Desktop/courses/c2_claude/gitlab/CMakeLists.txt;181;add_test;/Users/julia/Desktop/courses/c2_claude/gitlab/CMakeLists.txt;0;")
