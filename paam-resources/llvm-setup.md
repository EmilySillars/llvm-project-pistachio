# PAAM Resources: LLVM Setup

[[Return to Homepage]](./README.md)

*Based on [https://llvm.org/docs/GettingStarted.html](https://llvm.org/docs/GettingStarted.html#getting-the-source-code-and-building-llvm)*

## 1. Install ninja and lld

1. Get ninja, instructions here: [https://ninja-build.org/](https://ninja-build.org/)
   On ubuntu, get ninja by doing:

   ```
   apt-get install ninja-build
   ```

   On fedora, do:
   ```
   dnf install ninja-build
   ```

2. Install the linker `lld`
   On fedora, do: `sudo dnf install lld`

   On Ubuntu, do: `sudo apt install lld`

## 2. Fork, Clone, and build LLVM for Default Targets

1. make a fork of the repo https://github.com/llvm/llvm-project.git on github.

2. clone the fork (replace the URL in the clone command with the URL to your own forked repo):
   `git clone https://github.com/EmilySillars/llvm-project-pistachio.git`

3. Enter the top-level directory of your cloned repo:

   ```
   cd llvm-project-pistachio
   ```


   Create a build directory and go inside it:

   ```
   mkdir myBuild; \
   cd myBuild
   ```

4. Configure your build with the following options:

   ```
   cmake -G Ninja ../llvm \
   -DLLVM_ENABLE_PROJECTS="clang;lld" \
   -DLLVM_BUILD_EXAMPLES=ON \
   -DLLVM_TARGETS_TO_BUILD="Native;NVPTX;AMDGPU" \
   -DCMAKE_BUILD_TYPE=Release \
   -DLLVM_USE_LINKER=lld \
   -DCMAKE_C_COMPILER=/usr/bin/clang \
   -DCMAKE_CXX_COMPILER=/usr/bin/clang++ \
   -DLLVM_ENABLE_ASSERTIONS=ON \
   -DLLVM_ENABLE_RTTI=ON
   ```

   ### Wait, what do all those cmake flags mean?

   - Full the list of definitions [here](https://llvm.org/docs/CMake.html#frequently-used-llvm-related-variables)
   - Note that `-DLLVM_TARGETS_TO_BUILD` specifies which hardware targets want to compile for. To add riscv as an additional target, for example, append to the end of the string "`;RISCV`"
   - Note that `-DLLVM_ENABLE_PROJECTS` specifies which LLVM projects you want to build. You want to build the clang project (LLVM's compiler), and LLVM's linker lld. MLIR is an example of another project within LLVM. If you want to build the MLIR project, as well, you add to the end of the string "`;mlir`".
   - `-DCMAKE_BUILD_TYPE=Release` we recommend release because it will take less time to build.

5. Build using ninja, passing the `-j` option the number of cores on your computer. (you can use `lscpu` to find out how many CPUs you have)

   ```
   ninja -j 20
   ```

   ^^ replace `20` with the number of CPUs your computer has.

## 4. Running out of space? 

*Based on: https://www.techotopia.com/index.php/Adding_and_Managing_Fedora_Swap_Space, also https://wiki.archlinux.org/title/Btrfs#Swap_file*

When I run `cmake --build .` , my computer runs out of space and kills the process. **I tried temporarily increasing my swap space...** 

Let's make a temporary swap space of 8 GB:

7812500 * 1024 = 8000000000 MB = 8 GB, so do

```
btrfs filesystem mkswapfile --size 8g --uuid clear /newswap
swapon /newswap
```

verify everything works properly with

```
swapon -s
```

**Now try re-running your build command!**

Deactivate your swap using

```
swapoff /newswap
```

## 5. Build LLVM with MLIR for Default + RISC-V Targets

*Based on Ravi's notes and [https://mlir.llvm.org/getting_started/](https://urldefense.com/v3/__https://mlir.llvm.org/getting_started/__;!!D9dNQwwGXtA!SEOGnQ6kNZqQqZPcpFmf8I3xb2_IWywnYuC-onD9gFeQlo2vOJYwE8HHNPkxP2PYzkO6OKhJlpcbI77vcrKwuA$)*

1. Clone LLVM (which we already did) and then make a build directory and go inside:

```
cd llvm-project-pistachio; \
mkdir build-riscv; \
cd build-riscv
```

2. Configure your build using the following options:

```
cmake -G Ninja ../llvm \
-DLLVM_ENABLE_PROJECTS="mlir;clang;lld" \
-DLLVM_BUILD_EXAMPLES=ON \
-DLLVM_TARGETS_TO_BUILD="Native;NVPTX;AMDGPU;RISCV" \
-DCMAKE_BUILD_TYPE=Release \
-DLLVM_USE_LINKER=lld \
-DCMAKE_C_COMPILER=/usr/bin/clang \
-DCMAKE_CXX_COMPILER=/usr/bin/clang++ \
-DLLVM_ENABLE_ASSERTIONS=ON \
-DLLVM_ENABLE_RTTI=ON
```

3. Build with:

```
ninja -j 20
```

^^ replace `20` with the number of CPUs your computer has.

## Troubleshooting (Fedora)

1. Error: `bash: cmake: command not found...`
   Solution: `dnf install cmake`

2. Error:

   ```CMake Error at cmake/modules/HandleLLVMOptions.cmake:330 (message):
   Host compiler does not support '-fuse-ld=lld'
   Call Stack (most recent call first):
   CMakeLists.txt:907 (include)```
   ```

   Temporary Solution: Edited this line `-DLLVM_ENABLE_PROJECTS="mlir;clang" \` to include lld: `-DLLVM_ENABLE_PROJECTS="mlir;clang;lld" \`
   Error came back after solving #3...

   ```
   CMake Error at cmake/modules/HandleLLVMOptions.cmake:330 (message):
     Host compiler does not support '-fuse-ld=lld'
   Call Stack (most recent call first):
     CMakeLists.txt:907 (include)
   ```

   Solution:

   ```
   sudo dnf install lld
   ln -s /usr/bin/ld.lld /usr/bin/ld
   ```

   AND add following flags to cmake command: ` -DCMAKE_C_COMPILER=/usr/bin/clang \ `

   **NOTE:** If `fuse-ld`error persists even after doing all these things, try removing everything from your build directory and the running the cmake command again.

3. Error:

   ``````
   CMake Error at CMakeLists.txt:47 (project):
     The CMAKE_CXX_COMPILER:
   
       /usr/bin/clang++
   
     is not a full path to an existing compiler tool.
   
     Tell CMake where to find the compiler by setting either the environment
     variable "CXX" or the CMake cache entry CMAKE_CXX_COMPILER to the full path
     to the compiler, or to the compiler name if it is in the PATH.
   ``````

   Solution:
   Install clang using terminal prompts:

   ``` 
   [hoppip@inf-205-234 build-riscv]$ clang
   bash: clang: command not found...
   Install package 'clang' to provide command 'clang'? [N/y] y 
   y
   ```


4. Error:

   ```
   CMake Error at MLIR.cmake:13 (find_package):
     Could not find a package configuration file provided by "MLIR" with any of
     the following names:
   
       MLIRConfig.cmake
       mlir-config.cmake
   
     Add the installation prefix of "MLIR" to CMAKE_PREFIX_PATH or set
     "MLIR_DIR" to a directory containing one of the above files.  If "MLIR"
     provides a separate development package or SDK, be sure it has been
     installed.
   Call Stack (most recent call first):
     CMakeLists.txt:79 (include)
   ```

   Solution: 
   Make sure you set the MLIR_DIR environment variable correctly! My fork of llvm-project is called llvm-project-*pistachio*, and my build folder name is *build-riscv*, not build.

   `MLIR_DIR=$(pwd)/llvm-project-pistachio/build-riscv/lib/cmake/mlir`

5. Error:

   ```
   CMake Error at /usr/share/cmake/Modules/FindPackageHandleStandardArgs.cmake:230 (message):
     Could NOT find Python3 (missing: Python3_INCLUDE_DIRS Python3_LIBRARIES
     Development Development.Module Development.Embed) (found version "3.11.6")
   Call Stack (most recent call first):
     /usr/share/cmake/Modules/FindPackageHandleStandardArgs.cmake:600 (_FPHSA_FAILURE_MESSAGE)
     /usr/share/cmake/Modules/FindPython/Support.cmake:3824 (find_package_handle_standard_args)
     /usr/share/cmake/Modules/FindPython3.cmake:545 (include)
     CMakeLists.txt:89 (find_package)
   ```

   Tried: `pythonLocation=/usr/bin/python3.11` but did not help :(

   Tried: `pythonLocation=/usr/bin/python`

   Tried: `pythonLocation=/usr/bin/python3`
   Tried: adding cmake the command line flag `-DPython3_EXECUTABLE=/usr/bin/python3.11` to previously entered command:

   ```
   if [[ -z "$pythonLocation" ]]; then
     cmake -G Ninja \
           -DCMAKE_CXX_COMPILER=/usr/bin/c++ \
           -DMLIR_DIR=${MLIR_DIR} \
           -DPython3_EXECUTABLE=/usr/bin/python3.11 \
           ..
   else
     cmake -G Ninja \
           -DCMAKE_CXX_COMPILER=/usr/bin/c++ \
           -DPython3_ROOT_DIR=$pythonLocation \
           -DMLIR_DIR=${MLIR_DIR} \
           -DPython3_EXECUTABLE=/usr/bin/python3.11 \
           ..
   fi
   ```

   And got new error! #6

   **Solution:** `sudo dnf install python3-devel`

   6. Error:

      ```
      CMake Error: Error: generator : Ninja
      Does not match the generator used previously: Unix Makefiles
      Either remove the CMakeCache.txt file and CMakeFiles directory or choose a different binary directory.
      ```

      Tried:

      In `/onnx-mlir/CMakeCache.txt` tried changing

      ```
      //Name of generator.
      CMAKE_GENERATOR:INTERNAL=Unix Makefiles
      ```

      to

      ```
      //Name of generator.
      CMAKE_GENERATOR:INTERNAL=Ninja
      ```

      and got new error! #7

      

      7. Error:

      ```
      CMake Error:
        The detected version of Ninja (GNU Make 4.3
      
        Built for x86_64-redhat-linux-gnu
      
        Copyright (C) 1988-2020 Free Software Foundation, Inc.
      
        License GPLv3+: GNU GPL version 3 or later
        <http://gnu.org/licenses/gpl.html>
      
        This is free software: you are free to change and redistribute it.
      
        There is NO WARRANTY, to the extent permitted by law.) is less than the
        version of Ninja required by CMake (1.3).
      
      
      CMake Error at /usr/share/cmake/Modules/Internal/CheckSourceCompiles.cmake:101 (try_compile):
        Failed to generate test project build system.
      Call Stack (most recent call first):
        /usr/share/cmake/Modules/Internal/CheckCompilerFlag.cmake:18 (cmake_check_source_compiles)
        /usr/share/cmake/Modules/CheckCCompilerFlag.cmake:51 (cmake_check_compiler_flag)
        /home/hoppip/llvm-project-pistachio/build-riscv/lib/cmake/llvm/HandleLLVMOptions.cmake:277 (check_c_compiler_flag)
        /home/hoppip/llvm-project-pistachio/build-riscv/lib/cmake/llvm/HandleLLVMOptions.cmake:342 (add_flag_or_print_warning)
        MLIR.cmake:25 (include)
        CMakeLists.txt:79 (include)
      
      ```

      Tried:

      In `/onnx-mlir/CMakeCache.txt` tried changing

      ```
      //Path to a program.
      CMAKE_MAKE_PROGRAM:FILEPATH=/usr/bin/gmake
      ```

      to

      ```
      //Path to a program.
      CMAKE_MAKE_PROGRAM:FILEPATH=/usr/bin/ninja
      ```

      and error #5 returned!!

   7. Error:

   ```
   [80/578] Building CXX object src/Dialect...ntsAttr.dir/DisposableElementsAttr.cpp.o
   FAILED: src/Dialect/ONNX/ElementsAttr/CMakeFiles/OMONNXElementsAttr.dir/DisposableElementsAttr.cpp.o 
   /usr/bin/c++ -DONNX_MLIR_DECOMP_ONNX_CONVTRANSPOSE -DONNX_MLIR_ENABLE_STABLEHLO -D_DEBUG -D_GLIBCXX_ASSERTIONS -D_LIBCPP_ENABLE_HARDENED_MODE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -D__STDC_LIMIT_MACROS -I/home/hoppip/llvm-project-pistachio/llvm/include -I/home/hoppip/llvm-project-pistachio/build-riscv/include -I/home/hoppip/llvm-project-pistachio/mlir/include -I/home/hoppip/llvm-project-pistachio/build-riscv/tools/mlir/include -I/home/hoppip/onnx-mlir/include -I/home/hoppip/onnx-mlir -fno-semantic-interposition -fvisibility-inlines-hidden -Werror=date-time -fno-lifetime-dse -Wall -Wextra -Wno-unused-parameter -Wwrite-strings -Wcast-qual -Wno-missing-field-initializers -Wimplicit-fallthrough -Wno-nonnull -Wno-class-memaccess -Wno-redundant-move -Wno-pessimizing-move -Wno-noexcept-type -Wdelete-non-virtual-dtor -Wsuggest-override -Wno-comment -Wno-misleading-indentation -Wctad-maybe-unsupported -fdiagnostics-color -DSUPPRESS_THIRD_PARTY_WARNINGS -g -std=gnu++17   -D_DEBUG -D_GLIBCXX_ASSERTIONS -D_LIBCPP_ENABLE_HARDENED_MODE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -D__STDC_LIMIT_MACROS -MD -MT src/Dialect/ONNX/ElementsAttr/CMakeFiles/OMONNXElementsAttr.dir/DisposableElementsAttr.cpp.o -MF src/Dialect/ONNX/ElementsAttr/CMakeFiles/OMONNXElementsAttr.dir/DisposableElementsAttr.cpp.o.d -o src/Dialect/ONNX/ElementsAttr/CMakeFiles/OMONNXElementsAttr.dir/DisposableElementsAttr.cpp.o -c /home/hoppip/onnx-mlir/src/Dialect/ONNX/ElementsAttr/DisposableElementsAttr.cpp
   /home/hoppip/onnx-mlir/src/Dialect/ONNX/ElementsAttr/DisposableElementsAttr.cpp: In function ‘bool mlir::{anonymous}::shouldSwapLEBytes(unsigned int)’:
   /home/hoppip/onnx-mlir/src/Dialect/ONNX/ElementsAttr/DisposableElementsAttr.cpp:163:16: error: ‘llvm::endianness’ has not been declared
     163 |          llvm::endianness::native != llvm::endianness::little;
   ```

   Tried: Make sure you checked out the correct branch in your cloned fork of LLVM!!! (I hadn't) I need `b2cdf3cc4c08729d0ff582d55e40793a20bbcdcc` from here: https://github.com/llvm/llvm-project/commit/b2cdf3cc4c08729d0ff582d55e40793a20bbcdcc

   ```
   cd ~/llvm-project-pistachio
   git pull
   git config pull.rebase true
   git pull
   git checkout b2cdf3cc4c08729d0ff582d55e40793a20bbcdcc && cd ..
   ```

   9. Error:

   ```
   In file included from /home/hoppip/llvm-project-pistachio/mlir/include/mlir/Dialect/SparseTensor/IR/SparseTensor.h:14,
                    from /home/hoppip/onnx-mlir/third_party/stablehlo/stablehlo/dialect/Register.cpp:21:
   /home/hoppip/llvm-project-pistachio/mlir/include/mlir/Dialect/SparseTensor/IR/SparseTensorInterfaces.h:29:10: fatal error: mlir/Dialect/SparseTensor/IR/SparseTensorInterfaces.h.inc: No such file or directory
      29 | #include "mlir/Dialect/SparseTensor/IR/SparseTensorInterfaces.h.inc"
   
   ```

   What did I do to fix this? Reset the MLIR_DIR flag?

   10. Error:

       ```
       ld: error: undefined symbol: typeinfo for llvm::cl::Option
       >>> referenced by BinaryDecoder.cpp
       ```

       Tried: Recompiling MLIR with `-DLLVM_ENABLE_RTTI=ON` flag set.

   11. Error:

       ```
       In file included from /home/hoppip/onnx-mlir/third_party/rapidcheck/include/rapidcheck/detail/ShowType.h:21,
                        from /home/hoppip/onnx-mlir/third_party/rapidcheck/include/rapidcheck/detail/Any.hpp:8,
                        from /home/hoppip/onnx-mlir/third_party/rapidcheck/include/rapidcheck/detail/Any.h:60,
                        from /home/hoppip/onnx-mlir/third_party/rapidcheck/src/detail/Any.cpp:1:
       /home/hoppip/onnx-mlir/third_party/rapidcheck/include/rapidcheck/detail/ShowType.hpp: In static member function ‘static void rc::ShowType<T>::showType(std::ostream&)’:
       /home/hoppip/onnx-mlir/third_party/rapidcheck/include/rapidcheck/detail/ShowType.hpp:48:36: error: cannot use ‘typeid’ with ‘-fno-rtti’
          48 |     os << detail::demangle(typeid(T).name());
             |                                    ^
       
       ```

   12. Error:

   ```
   Traceback (most recent call last):
     File "/home/hoppip/onnx-mlir/build/docs/doc_example/gen_add_onnx.py", line 1, in <module>
       import onnx
   ModuleNotFoundError: No module named 'onnx'
   ```

   Solution: `pip install onnx`

## Troubleshooting (Ubuntu):

1. Error: 

   ```sudo ./configure --prefix=/opt/riscv --with-arch=rv64g --with-abi=lp64d
   [sudo] password for emily: 
   checking for gcc... gcc
   checking whether the C compiler works... yes
   checking for C compiler default output file name... a.out
   checking for suffix of executables... 
   checking whether we are cross compiling... no
   checking for suffix of object files... o
   checking whether we are using the GNU C compiler... yes
   checking whether gcc accepts -g... yes
   checking for gcc option to accept ISO C89... none needed
   checking for grep that handles long lines and -e... /usr/bin/grep
   checking for fgrep... /usr/bin/grep -F
   checking for grep that handles long lines and -e... (cached) /usr/bin/grep
   checking for bash... /bin/bash
   configure: error: GNU Awk not found
   ```

   Solution:
   ```sudo apt-get install gawk```

2. Error:

   ```
   /home/emily/riscv-gnu-toolchain/binutils/missing: 81: makeinfo: not found
   WARNING: 'makeinfo' is missing on your system.
   ```

   Solution: `sudo apt-get install texinfo`

3. Error:

   ```
   make[3]: *** [Makefile:1244: arparse.c] Error 127
   /home/emily/riscv-gnu-toolchain/binutils/missing: 81: bison: not found
   WARNING: 'bison' is missing on your system.
   ```

   Solution: 

   ```
   sudo apt-get install bison
   sudo apt-get install flex  
   ```


4. `/usr/bin/ld: cannot find -lstdc++: No such file or directory`

```
CMake Error at /usr/share/cmake-3.22/Modules/CMakeTestCXXCompiler.cmake:62 (message):

The C++ compiler

  

"/usr/bin/c++"

  

is not able to compile a simple test program.

  

It fails with the following output:

  

Change Dir: /home/spinel/llvm-project-pistachio/build/CMakeFiles/CMakeTmp

Run Build Command(s):/usr/bin/gmake -f Makefile cmTC_c9928/fast && /usr/bin/gmake -f CMakeFiles/cmTC_c9928.dir/build.make CMakeFiles/cmTC_c9928.dir/build

gmake[1]: Entering directory '/home/spinel/llvm-project-pistachio/build/CMakeFiles/CMakeTmp'

Building CXX object CMakeFiles/cmTC_c9928.dir/testCXXCompiler.cxx.o

/usr/bin/c++ -MD -MT CMakeFiles/cmTC_c9928.dir/testCXXCompiler.cxx.o -MF CMakeFiles/cmTC_c9928.dir/testCXXCompiler.cxx.o.d -o CMakeFiles/cmTC_c9928.dir/testCXXCompiler.cxx.o -c /home/spinel/llvm-project-pistachio/build/CMakeFiles/CMakeTmp/testCXXCompiler.cxx

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

  

5.

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

6.  

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

7. ```
   cc: error: unrecognized command-line option ‘-Wcovered-switch-default’; did you mean ‘-Wno-switch-default’?
   cc: error: unrecognized command-line option ‘-Wstring-conversion’; did you mean ‘-Wsign-conversion’?
   ```

Solution: remove the flag `DCMAKE_CXX_COMPILER=/usr/bin/clang++` from your cmake command!