cmake_minimum_required (VERSION 2.8)
project(octomap_maplab_plugin)

find_package(catkin_simple REQUIRED)
# octomap 
find_package( octomap REQUIRED )

catkin_simple(ALL_DEPS_REQUIRED)

add_definitions(-fPIC -shared)

cs_add_library(${PROJECT_NAME} src/octomap_maplab_plugin.cc)
target_link_libraries(${PROJECT_NAME} ${OCTOMAP_LIBRARIES})

create_console_plugin(${PROJECT_NAME})

cs_install()
cs_export()
