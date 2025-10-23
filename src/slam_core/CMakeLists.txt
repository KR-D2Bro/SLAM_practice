cmake_minimum_required(VERSION 3.8)
project(slam_core)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

add_compile_options(-march=native)
add_compile_definitions(EIGEN_MAX_ALIGN_BYTES=32)

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(OpenCV REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(Sophus REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(g2o REQUIRED COMPONENTS core stuff types_sba solver_dense solver_eigen)
find_package(sensor_msgs REQUIRED)

# uncomment the following section in order to fill in
# further dependencies manually.
# find_package(<dependency> REQUIRED)
add_executable(slam_system src/slam_system.cpp src/frontend/feature_tracker.cpp src/frontend/visual_odometry.cpp src/frontend/frontend.cpp)

ament_target_dependencies( slam_system
  "rclcpp"
  #"image_/ransport"
  "cv_bridge"
  "sensor_msgs"
  "nav_msgs"
  #"std_msgs"
  # "OpenCV"
  #"PCL"
  "Sophus"
)

target_include_directories(slam_system PUBLIC ${OpenCV_INCLUDE_DIRS} $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include> )
target_link_libraries(slam_system
  ${OpenCV_LIBRARIES}
  Sophus::Sophus
  g2o::core
  g2o::stuff
  g2o::types_sba
  g2o::solver_dense
  g2o::solver_eigen
)

install(TARGETS 
slam_system
DESTINATION lib/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
