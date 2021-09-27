#Specify the version being used as well as the language
cmake_minimum_required(VERSION 2.8.11)

#Name your project here
project(AFAN) #Accurate Flow in Augmented Network

#set the module directory
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}")

#build the executable in the binary directory on MS Visual Studio
if ( MSVC )
	SET( CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG "${OUTPUT_DIRECTORY}")
	SET( CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE "${OUTPUT_DIRECTORY}")
endif ( MSVC )


#find packages-----------------------------------
#find the STIM library
find_package(STIM)

#if the STIM library isn't found, download it
if(NOT STIM_INCLUDE_DIRS)
	file(REMOVE_RECURSE ${CMAKE_BINARY_DIR}/stimlib)	#remove the stimlib directory if it exists
	set(STIM_GIT "https://git.stim.ee.uh.edu/codebase/stimlib.git")
	execute_process(COMMAND git clone --depth 1 ${STIM_GIT} WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
	set(STIM_INCLUDE_DIRS "${CMAKE_BINARY_DIR}/stimlib" CACHE TYPE PATH)
endif(NOT STIM_INCLUDE_DIRS)

#find BOOST
find_package(Boost REQUIRED)

#find cuda
find_package(CUDA REQUIRED)

#find the GLUT library for visualization
find_package(OpenGL REQUIRED)

find_package(GLUT REQUIRED)

#find OpenCV
find_package(OpenCV REQUIRED)
add_definitions(-DUSING_OPENCV)

#find the pthreads package
find_package(Threads)

#find the X11 package
find_package(X11)

include_directories(
					${OPENGL_INCLUDE_DIRS}
					${GLUT_INCLUDE_DIR}
					${STIM_INCLUDE_DIRS}
					${Boost_INCLUDE_DIR}
					${CUDA_INCLUDE_DIRS}
					${OpenCV_INCLUDE_DIRS}
)

#Assign source files to the appropriate variables
file(GLOB MODULE_FLOW2_SRC "${CMAKE_SOURCE_DIR}/source/flow2/main.cu")
file(GLOB MODULE_FLOW3_SRC "${CMAKE_SOURCE_DIR}/source/flow3/main.cu")


#create the flow2 executable----------------------------------------------
cuda_add_executable(flow2
					 ${MODULE_FLOW2_SRC}
)

target_link_libraries(flow2
#						${OpenGL_LIBRARIES}
						${OPENGL_gl_LIBRARY}
						${OPENGL_glu_LIBRARY}
						${GLUT_LIBRARIES}
						${CMAKE_THREAD_LIBS_INIT}
						${X11_LIBRARIES}
						${CUDA_cublas_LIBRARY}
						${OpenCV_LIBS}
)

#create the flow3 executable----------------------------------------------

cuda_add_executable(flow3
					 ${MODULE_FLOW3_SRC}
)

target_link_libraries(flow3
#						${OpenGL_LIBRARIES}
						${OPENGL_gl_LIBRARY}
						${OPENGL_glu_LIBRARY}
						${GLUT_LIBRARIES}
						${CMAKE_THREAD_LIBS_INIT}
						${X11_LIBRARIES}
						${CUDA_cublas_LIBRARY}
						${OpenCV_LIBS}
)


#set up copying data files to build directory
configure_file(sample_networks/3D_real_seg.obj ${CMAKE_CURRENT_BINARY_DIR}/3D_real_seg.obj @ONLY)
configure_file(sample_networks/3D_synthetic.obj ${CMAKE_CURRENT_BINARY_DIR}/3D_synthetic.obj @ONLY)