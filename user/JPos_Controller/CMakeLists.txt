include_directories(${PROJECT_SOURCE_DIR})
include_directories("../../robot/include")
include_directories("../../common/include/")
include_directories("../../lcm-types/cpp")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
set(Torch_DIR /home/user/raisim_workspace/libtorch/share/cmake/Torch)
find_package(Torch REQUIRED)

add_executable(jpos_ctrl main.cpp JPos_Controller.cpp)
target_link_libraries(jpos_ctrl robot biomimetics dynacore_param_handler qpOASES "${TORCH_LIBRARIES}")

