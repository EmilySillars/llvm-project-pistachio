# Spanish notes lol
based on ravi and https://mlir.llvm.org/getting_started/ :
1) get ninja here: https://ninja-build.org/
on ubuntu, do
```
apt-get install ninja-build
```
2) make build directory and go into it:
```
cd build-riscv
```
build using following options:
```
cmake -G Ninja ../llvm \
   -DLLVM_ENABLE_PROJECTS="mlir;clang" \
   -DLLVM_BUILD_EXAMPLES=ON \
   -DLLVM_TARGETS_TO_BUILD="Native;NVPTX;AMDGPU;RISCV" \
   -DCMAKE_BUILD_TYPE=Debug \
   -DLLVM_USE_LINKER=lld \
   -DCMAKE_CXX_COMPILER=/usr/bin/clang++ \
   -DLLVM_ENABLE_ASSERTIONS=ON
```
then
```
ninja -j 8
```

# Cloning/Building Notes

1. `git clone --depth 1 https://github.com/EmilySillars/llvm-project-pistachio.git`
2. `cd llvm-project-pistachio`
3. `cmake -S llvm -B build -G <generator> [options]`
where
- for `<generator>`, I picked `Unix Makefiles` even  though I use  visual  studio; hopefully this is okay.
- for `[options]`, I picked
```
-DLLVM_ENABLE_PROJECTS="clang;lld" # use clang front end and lld linker

-DCMAKE_BUILD_TYPE=Debug # use a debug build

-DLLVM_USE_LINKER=lld # use the lld linker provided it's installed

-DLLVM_PARALLEL_LINK_JOBS=2 # limit to 2 parallel linking jobs
```

- Chose N=2 for parallel link option based on [this post](https://reviews.llvm.org/D72402).

So full command looks like:

~~cmake -S llvm -B build -G "Unix Makefiles" -DLLVM_ENABLE_PROJECTS="clang;lld" -DCMAKE_BUILD_TYPE=Debug -DLLVM_USE_LINKER=lld -DCMAKE_CXX_COMPILER=/usr/bin/c++~~
```cmake
cmake -S llvm -B build -G "Unix Makefiles" -DLLVM_ENABLE_PROJECTS="clang;lld" -DCMAKE_BUILD_TYPE=Debug -DLLVM_USE_LINKER=lld -DCMAKE_CXX_COMPILER=/usr/bin/clang++ -DLLVM_PARALLEL_LINK_JOBS=2
```


4. `make -j16 -C build check-llvm`
My machine has 16 cores...

5. To build opt, do `make -j16 -C build opt`





# Resolving some errors

1.  `/usr/bin/ld: cannot find -lstdc++: No such file or directory`
```
CMake Error at /usr/share/cmake-3.22/Modules/CMakeTestCXXCompiler.cmake:62 (message):
  The C++ compiler

    "/usr/bin/c++"

  is not able to compile a simple test program.

  It fails with the following output:

    Change Dir: /home/spinel/llvm-project-pistachio/build/CMakeFiles/CMakeTmp
    
    Run Build Command(s):/usr/bin/gmake -f Makefile cmTC_c9928/fast && /usr/bin/gmake  -f CMakeFiles/cmTC_c9928.dir/build.make CMakeFiles/cmTC_c9928.dir/build
    gmake[1]: Entering directory '/home/spinel/llvm-project-pistachio/build/CMakeFiles/CMakeTmp'
    Building CXX object CMakeFiles/cmTC_c9928.dir/testCXXCompiler.cxx.o
    /usr/bin/c++    -MD -MT CMakeFiles/cmTC_c9928.dir/testCXXCompiler.cxx.o -MF CMakeFiles/cmTC_c9928.dir/testCXXCompiler.cxx.o.d -o CMakeFiles/cmTC_c9928.dir/testCXXCompiler.cxx.o -c /home/spinel/llvm-project-pistachio/build/CMakeFiles/CMakeTmp/testCXXCompiler.cxx
    Linking CXX executable cmTC_c9928
    /usr/bin/cmake -E cmake_link_script CMakeFiles/cmTC_c9928.dir/link.txt --verbose=1
    /usr/bin/c++ CMakeFiles/cmTC_c9928.dir/testCXXCompiler.cxx.o -o cmTC_c9928 
    /usr/bin/ld: cannot find -lstdc++: No such file or directory
    clang: error: linker command failed with exit code 1 (use -v to see invocation)
    gmake[1]: *** [CMakeFiles/cmTC_c9928.dir/build.make:100: cmTC_c9928] Error 1
    gmake[1]: Leaving directory '/home/spinel/llvm-project-pistachio/build/CMakeFiles/CMakeTmp'
    gmake: *** [Makefile:127: cmTC_c9928/fast] Error 2
    
    
  CMake will not be able to correctly generate this project.
Call Stack (most recent call first):
  CMakeLists.txt:47 (project)

```
Solution: upgrade to latest GCC toolchain:

```
sudo apt-get update -y && \
sudo apt-get upgrade -y && \
sudo apt-get dist-upgrade -y && \
sudo apt-get install build-essential software-properties-common -y && \
sudo add-apt-repository ppa:ubuntu-toolchain-r/test -y && \
sudo apt-get update -y && \
sudo apt-get install gcc-13 g++-13 -y && \
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-13 60 --slave /usr/bin/g++ g++ /usr/bin/g++-13 && \
sudo update-alternatives --config gcc
```

2.
```
CMake Error at cmake/modules/HandleLLVMOptions.cmake:330 (message):
  Host compiler does not support '-fuse-ld=lld'
Call Stack (most recent call first):
  CMakeLists.txt:907 (include)
```
Maybe Solution:
```
sudo apt install lld
ln -s /usr/bin/ld.lld /usr/bin/ld 
```
New problem: `ln: failed to create symbolic link '/usr/bin/ld': File exists`
Solution to New problem: `sudo rm /usr/bin/ld`
Old problem remained, so then tried:

```
ln -s /usr/bin/lld /usr/bin/ld 
```
Old problem remained, so then tried:
```
cmake -S llvm -B build -G "Unix Makefiles" -DLLVM_ENABLE_PROJECTS="clang;lld" -DCMAKE_BUILD_TYPE=Debug -DLLVM_USE_LINKER=lld -DCMAKE_CXX_COMPILER=/usr/bin/clang++
```

^ and this worked!

3.

```
CMake Error at CMakeLists.txt:47 (project):
  The CMAKE_CXX_COMPILER:

    /usr/bin/clang++

  is not a full path to an existing compiler tool.

  Tell CMake where to find the compiler by setting either the environment
  variable "CXX" or the CMake cache entry CMAKE_CXX_COMPILER to the full path
  to the compiler, or to the compiler name if it is in the PATH.
```

Solution: `sudo apt install clang`

# Pistachio Notes

Testing on emily-notes directory's hello2.c test file:

```
make -j16 -C build opt;
clang -O1 -Xclang -disable-llvm-passes -emit-llvm emily-notes/hello2.c -c -o emily-notes/out/hello2.bc;
build/bin/opt -disable-output emily-notes/out/hello2.bc -passes=pistachio
```

Example of running a single regression test: `build/bin/llvm-lit llvm/test/Transforms/Pistachio`

[This](http://www.cs.cornell.edu/~asampson/blog/llvm.html) would have been helpful earlier on xD

## Cholesky Notes

`cd emily-notes/polybench-c-3.2`

Compile Cholesky:

`gcc -I utilities -I linear-algebra/kernels/cholesky utilities/polybench.c linear-algebra/kernels/cholesky/cholesky.c -o cholesky_base -lm`

Run it:

`./cholesky_base 2>cholesky_ref.out`

```
/* Main computational kernel. The whole function will be timed,
   including the call and return. */
static
void kernel_cholesky(int n,
		     DATA_TYPE POLYBENCH_1D(p,N,n),
		     DATA_TYPE POLYBENCH_2D(A,N,N,n,n))
{
  int i, j, k;

  DATA_TYPE x;

#pragma scop
for (i = 0; i < _PB_N; ++i)
  {
    x = A[i][i];
    for (j = 0; j <= i - 1; ++j)
      x = x - A[i][j] * A[i][j];
    p[i] = 1.0 / sqrt(x);
    for (j = i + 1; j < _PB_N; ++j)
      {
	x = A[i][j];
	for (k = 0; k <= i - 1; ++k)
	  x = x - A[j][k] * A[i][k];
	A[j][i] = x * p[i];
      }
  }
#pragma endscop

}
```

