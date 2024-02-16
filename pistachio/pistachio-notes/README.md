# Pistachio Pass

This forked repo of LLVM adds a compiler pass called pistachio. Pistachio takes in LLVM IR, examines the source program function by function, and then prints every store instruction with any loads with which it aliases. 

Relevant files changed:

- [pistachio.cpp](https://github.com/EmilySillars/llvm-project-pistachio/blob/main/llvm/lib/Transforms/Utils/Pistachio.cpp) 

- [pistachio.h](https://github.com/EmilySillars/llvm-project-pistachio/blob/main/llvm/include/llvm/Transforms/Utils/Pistachio.h)

Output from running on kernel_cholesky function: 

- [cholesky-analyzed.txt](cholesky-analyzed.txt) 

### Clone + Build

Clone with:

```
git clone --depth 1 git@github.com:EmilySillars/llvm-project-pistachio.git
```

Build with:

```
make -j$(nproc) -C build opt
```

### Testing

To run a few basic tests, including [kernel_cholesky](hacky-cholesky/hacky-cholesky.c): 

```
sh pistachio-notes/test-pistachio
```

To run this pass manually on  any LLVM bitcode file:

```
build/bin/opt -disable-output -passes=pistachio <filename.bc>
```

### Worked Example

See [slides](worked_example.pdf).

