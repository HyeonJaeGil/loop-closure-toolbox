set(DBOW_DIRS ../src/DBoW3/include)
set(DBOW_LIBS DBoW3)
set(VLAD_DIRS ../src/VLAD/include)
set(VLAD_LIBS VLAD)

include_directories(${DBOW_DIRS} ${VLAD_DIRS})

set(LIBRARIES ${DBOW_LIBS} ${VLAD_LIBS} stdc++fs)

add_executable(test_dbow_database test_dbow_database.cpp)
target_link_libraries(test_dbow_database ${LIBRARIES})

add_executable(test_dbow_transform test_dbow_transform.cpp)
target_link_libraries(test_dbow_transform ${LIBRARIES})

add_executable(test_vlad_database test_vlad_database.cpp)
target_link_libraries(test_vlad_database ${LIBRARIES})

add_executable(test_vlad_transform test_vlad_transform.cpp)
target_link_libraries(test_vlad_transform ${LIBRARIES})