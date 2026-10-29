set(cumlrkt_gtest_timeout_seconds 6)

function(cumlrkt_enable_warnings target)
  target_compile_options(${target}
    PRIVATE
      $<$<COMPILE_LANGUAGE:C,CXX>:-Wall -Wextra -Wpedantic -Werror>
      $<$<COMPILE_LANGUAGE:CUDA>:-Werror=all-warnings -Xcompiler=-Wall,-Wextra,-Werror>
  )
endfunction()

# Nix's cc-wrapper passes search paths through NIX_CFLAGS_COMPILE, which
# compile_commands.json never sees; mirror them so clang-tidy finds headers.
function(cumlrkt_apply_nix_cflags target)
  if(DEFINED ENV{NIX_CFLAGS_COMPILE})
    separate_arguments(cumlrkt_nix_flags UNIX_COMMAND "$ENV{NIX_CFLAGS_COMPILE}")
    set(expect_dir OFF)
    foreach(flag IN LISTS cumlrkt_nix_flags)
      if(expect_dir)
        target_compile_options(${target} PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:-isystem${flag}>")
        set(expect_dir OFF)
      elseif(flag STREQUAL "-isystem")
        set(expect_dir ON)
      endif()
    endforeach()
  endif()
  foreach(dir IN LISTS CMAKE_CXX_IMPLICIT_INCLUDE_DIRECTORIES)
    if(EXISTS "${dir}")
      target_compile_options(${target} PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:-isystem${dir}>")
    endif()
  endforeach()
endfunction()

function(cumlrkt_family family)
  cmake_parse_arguments(PARSE_ARGV 1 arg "TEST_ONLY" "" "SOURCES;TESTS")
  if(arg_TEST_ONLY AND NOT BUILD_TESTING)
    return()
  endif()
  set(lib cumlrkt_${family})
  set(common_sources)
  if(NOT arg_TEST_ONLY)
    set(common_sources common/src/rows.cu)
  endif()
  add_library(${lib} SHARED ${arg_SOURCES} ${common_sources})
  set_target_properties(${lib} PROPERTIES
    LINKER_LANGUAGE CXX
    CXX_VISIBILITY_PRESET hidden
    CUDA_VISIBILITY_PRESET hidden
    POSITION_INDEPENDENT_CODE ON
  )
  target_include_directories(${lib}
    PUBLIC
      ${CMAKE_CURRENT_SOURCE_DIR}/common/include
      ${CMAKE_CURRENT_SOURCE_DIR}/${family}/include
    PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/${family}/src
  )
  target_link_libraries(${lib} PRIVATE raftrkt::headers cuml_runtime)
  target_link_options(${lib} PRIVATE "LINKER:--exclude-libs,ALL"
    "$<$<NOT:$<BOOL:${CUMLRKT_SANITIZE}>>:LINKER:--no-undefined>")
  cumlrkt_enable_warnings(${lib})
  cumlrkt_apply_nix_cflags(${lib})
  if(NOT arg_TEST_ONLY)
    install(TARGETS ${lib} LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
  elseif(DEFINED CUMLRKT_TESTS_PREFIX)
    install(TARGETS ${lib} LIBRARY DESTINATION ${CUMLRKT_TESTS_PREFIX}/lib)
  endif()
  set_property(GLOBAL APPEND PROPERTY cumlrkt_libraries ${lib})
  set(tidy ${arg_SOURCES})

  if(BUILD_TESTING AND arg_TESTS)
    set(tests ${lib}_tests)
    add_executable(${tests} ${arg_TESTS})
    target_include_directories(${tests} PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/${family}/tests
      ${CMAKE_CURRENT_SOURCE_DIR}/${family}/src)
    target_link_libraries(${tests}
      PRIVATE ${lib} raftrkt::headers "${RAFTRKT_LIBRARY}" GTest::gtest_main)
    cumlrkt_enable_warnings(${tests})
    cumlrkt_apply_nix_cflags(${tests})
    gtest_discover_tests(${tests} DISCOVERY_MODE PRE_TEST
      PROPERTIES TIMEOUT ${cumlrkt_gtest_timeout_seconds})
    if(DEFINED CUMLRKT_TESTS_PREFIX)
      if(arg_TEST_ONLY)
        set_property(TARGET ${tests} APPEND PROPERTY INSTALL_RPATH "${CUMLRKT_TESTS_PREFIX}/lib")
      endif()
      install(TARGETS ${tests} RUNTIME DESTINATION ${CUMLRKT_TESTS_PREFIX}/bin)
    endif()
    list(APPEND tidy ${arg_TESTS})
    set_property(GLOBAL APPEND PROPERTY cumlrkt_libraries ${tests})
  endif()

  list(TRANSFORM tidy PREPEND "${CMAKE_CURRENT_SOURCE_DIR}/")
  set_property(GLOBAL APPEND PROPERTY cumlrkt_tidy_sources ${tidy})
endfunction()

function(cumlrkt_add_tidy_target target)
  find_program(CLANG_TIDY_EXECUTABLE NAMES clang-tidy)
  get_property(sources GLOBAL PROPERTY cumlrkt_tidy_sources)
  get_property(builds GLOBAL PROPERTY cumlrkt_libraries)
  list(FILTER sources INCLUDE REGEX "\\.cpp$")
  if(CLANG_TIDY_EXECUTABLE)
    add_custom_target(${target}
      COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target ${builds}
      COMMAND ${CLANG_TIDY_EXECUTABLE} --quiet --warnings-as-errors=*
              -p ${CMAKE_BINARY_DIR} ${sources}
      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
      COMMENT "Running clang-tidy over the cumlrkt sources"
      VERBATIM
    )
  else()
    add_custom_target(${target}
      COMMAND ${CMAKE_COMMAND} -E false
      COMMENT "clang-tidy was not found"
      VERBATIM
    )
  endif()
endfunction()
