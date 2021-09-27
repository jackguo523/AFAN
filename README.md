AFAN can be built using CMake (https://cmake.org/) and a C/C++ compiler.

The STIM codebase is required, but will be cloned automatically if Git (https://git-scm.com/) is installed. The codebase can be downloaded manually here: https://git.stim.ee.uh.edu/codebase/stimlib

Required libraries: OpenCV: http://opencv.org/, OpenGL: http://www.opengl.org/


Step-by-step instructions:

1) Download and install CMake

2) Download and install OpenCV

3) Download and install OpenGL

4) Download and install Git

5) Set the CMake source directory to the directory containing this file

6) Specify the CMake build directory where you want the executable built

7) Use CMake to Configure and Generate the build environment

8) Build the software (ex. in Visual Studio you will open the generated solution and compile)
Note that AFAN has two module (flow2 and flow3), there will be two executables.