# C++ test

*For simplicity and modularity of package, we did not include gtest or other test utils.*
Instead, we added some per-function test functions.
More tests will be added later.


Test | Description | 
--- | --- |
[`test_dbow_database`](test_dbow_database.cpp)  | test `add` and `query` method of `DBoW3::Database` |
[`test_vlad_database`](test_vlad_database.cpp) | test `add` and `query` method of `VLAD::Database` |
[`test_vlad_transform`](test_vlad_transform.cpp) | test `transform` and `score` method of `VLAD::Vocabulary` |
