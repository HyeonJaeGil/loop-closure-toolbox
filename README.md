# DBoW3 + VLAD
## C++ and Python interface for Loop closure Detection

This repository provides well-known two loop closure detection modules, 
*DBoW3* and *VLAD* (Vector of Locally Aggregated Descriptors), in both C++ and Python interface.
There are no Feature Extraction or Matching utilities. We provide:
- Feature Aggregation (BoW or VLAD)
- Database Management (add + query)
  
So users are recommended to include this repository in their own SLAM project ***as a module***. 


### How to use (C++)
Please refer to [test folder](./test/) to check the usage of Vocabulary and Database.
**TODO**

### How to use (Python)
Please refer to [pydbow example](./python/test_pydbow.py) and [pyvlad example](./python/test_pyvlad.py) to check the usage of Vocabulary and Database.
**TODO**



### Citing

Codebase imported from the following:
- **Original DBoW3 repository [link](https://github.com/rmsalinas/DBow3)** : slightly modified to divide declarations and definitions.

```@online{DBoW3, author = {Rafael Muñoz-Salinas}, 
   title = {{DBoW3} DBoW3}, 
  year = 2017, 
  url = {https://github.com/rmsalinas/DBow3}, 
  urldate = {2017-02-17} 
 } 
```
- **JPL X library [link](https://github.com/jpl-x/x_multi_agent)** : import VLAD implementation and architecture.

```bibtex
@ARTICLE{Polizzi22RAL,
  author={Polizzi, Vincenzo and Hewitt, Robert and Hidalgo-Carrió, Javier and Delaune, Jeff and Scaramuzza, Davide},
  journal={IEEE Robotics and Automation Letters},   
  title={Data-Efficient Collaborative Decentralized Thermal-Inertial Odometry},   
  year={2022},  
  volume={7},  
  number={4},  
  pages={10681-10688},  
  doi={10.1109/LRA.2022.3194675}
}
```
- **pytlsd [link](https://github.com/iago-suarez/pytlsd)** : import setup.py implementation for pybind.