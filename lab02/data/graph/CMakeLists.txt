# The CMake setting of 'data/'

# Set objects
file(GLOB_RECURSE objfiles RELATIVE "${CMAKE_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/*.obj")

set(SCSC_OBJ_INPUT "data/Square.obj" CACHE STRING "The input object.")
set_property(CACHE SCSC_OBJ_INPUT PROPERTY STRINGS "${objfiles}")

set(SCSC_OBJ_OUTPUT "output.obj" CACHE STRING "The output object.")

set(SCSC_METHOD "0" CACHE STRING "The method.")
set_property(CACHE SCSC_METHOD PROPERTY STRINGS "0;1")

# Create symbolic links
foreach(objfile ${objfiles})
  string(REGEX REPLACE "\\.[^.]*$" "" file ${objfile})
  execute_process(
    COMMAND ${CMAKE_COMMAND} -E create_symlink
    ${CMAKE_SOURCE_DIR}/${objfile}
    ${CMAKE_BINARY_DIR}/${objfile}
  )
endforeach()
