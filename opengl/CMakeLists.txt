cmake_minimum_required(VERSION 3.1.0)

project(main)



#find_package (PkgConfig REQUIRED)
find_package (OpenGL REQUIRED)
find_package (GLUT REQUIRED)
find_package (GLEW REQUIRED)



if (WIN32)
include_directories(${OPENGL_INCLUDE_DIR}
                    ${GLUT_INCLUDE_DIRS}
                     ${GLEW_INCLUDE_DIR}
                      src_intro
                      src_shader
                      include
                      )
else ()
include_directories(${OPENGL_INCLUDE_DIR}
                    ${GLUT_INCLUDE_DIRS}
                      src_intro
                      src_shader
			src_adv
                      INCLUDE_LINUX
                      )
set (CMAKE_CXX_FLAGS "-Wall")

link_libraries(${GLUT_LIBRARIES})
endif (WIN32)


add_executable(main examples/reflectdino.c) 
target_link_libraries(main -lm -lglut -lGLU -lGL -lX11 -lXi -lXrandr 
                  -lXxf86vm -lXinerama -lXcursor -pthread -ldl)









