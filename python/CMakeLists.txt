add_subdirectory(pybind11)

set(DBOW_DIRS ../src/DBoW3/include)
set(DBOW_LIBS DBoW3)
set(VLAD_DIRS ../src/VLAD/include)
set(VLAD_LIBS VLAD)
set(PYBIND_DIRS pybind11/include)

include_directories(${DBOW_DIRS} ${VLAD_DIRS} ${PYBIND_DIRS})

pybind11_add_module(loopclosuretoolbox loopclosuretoolbox.cpp)
target_link_libraries(loopclosuretoolbox PRIVATE ${DBOW_LIBS} ${VLAD_LIBS})