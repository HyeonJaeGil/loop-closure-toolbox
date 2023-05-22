set(DBOW_DIRS ../src/DBoW3/include)
set(DBOW_LIBS DBoW3)
set(VLAD_DIRS ../src/VLAD/include)
set(VLAD_LIBS VLAD)

include_directories(${DBOW_DIRS} ${VLAD_DIRS})

set(LIBRARIES ${DBOW_LIBS} ${VLAD_LIBS} stdc++fs)

add_executable(test_vocabulary test_vocabulary.cpp)
target_link_libraries(test_vocabulary ${LIBRARIES})

add_executable(test_database test_database.cpp)
target_link_libraries(test_database ${LIBRARIES})