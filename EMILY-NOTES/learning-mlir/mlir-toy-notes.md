# notes on MLIR toy tutorial
## Ch 1
To take toy source code and emit as MLIR AST, do
```
build-riscv/bin/toyc-ch1 mlir/test/Examples/Toy/Ch1/ast.toy -emit=ast

```
Example program written in toy:
```
# User defined generic function that operates on unknown shaped arguments.
def multiply_transpose(a, b) {
  return transpose(a) * transpose(b);
}

def main() {
  # Define a variable `a` with shape <2, 3>, initialized with the literal value.
  var a = [[1, 2, 3], [4, 5, 6]];
  var b<2, 3> = [1, 2, 3, 4, 5, 6];

  # This call will specialize `multiply_transpose` with <2, 3> for both
  # arguments and deduce a return type of <3, 2> in initialization of `c`.
  var c = multiply_transpose(a, b);

  # A second call to `multiply_transpose` with <2, 3> for both arguments will
  # reuse the previously specialized and inferred version and return <3, 2>.
  var d = multiply_transpose(b, a);

  # A new call with <3, 2> (instead of <2, 3>) for both dimensions will
  # trigger another specialization of `multiply_transpose`.
  var e = multiply_transpose(c, d);

  # Finally, calling into `multiply_transpose` with incompatible shapes
  # (<2, 3> and <3, 2>) will trigger a shape inference error.
  var f = multiply_transpose(a, c);
}
```
Parsed AST: 
```
Module:
  Function 
    Proto 'multiply_transpose' @test/Examples/Toy/Ch1/ast.toy:4:1
    Params: [a, b]
    Block {
      Return
        BinOp: * @test/Examples/Toy/Ch1/ast.toy:5:25
          Call 'transpose' [ @test/Examples/Toy/Ch1/ast.toy:5:10
            var: a @test/Examples/Toy/Ch1/ast.toy:5:20
          ]
          Call 'transpose' [ @test/Examples/Toy/Ch1/ast.toy:5:25
            var: b @test/Examples/Toy/Ch1/ast.toy:5:35
          ]
    } // Block
  Function 
    Proto 'main' @test/Examples/Toy/Ch1/ast.toy:8:1
    Params: []
    Block {
      VarDecl a<> @test/Examples/Toy/Ch1/ast.toy:11:3
        Literal: <2, 3>[ <3>[ 1.000000e+00, 2.000000e+00, 3.000000e+00], <3>[ 4.000000e+00, 5.000000e+00, 6.000000e+00]] @test/Examples/Toy/Ch1/ast.toy:11:11
      VarDecl b<2, 3> @test/Examples/Toy/Ch1/ast.toy:15:3
        Literal: <6>[ 1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00] @test/Examples/Toy/Ch1/ast.toy:15:17
      VarDecl c<> @test/Examples/Toy/Ch1/ast.toy:19:3
        Call 'multiply_transpose' [ @test/Examples/Toy/Ch1/ast.toy:19:11
          var: a @test/Examples/Toy/Ch1/ast.toy:19:30
          var: b @test/Examples/Toy/Ch1/ast.toy:19:33
        ]
      VarDecl d<> @test/Examples/Toy/Ch1/ast.toy:22:3
        Call 'multiply_transpose' [ @test/Examples/Toy/Ch1/ast.toy:22:11
          var: b @test/Examples/Toy/Ch1/ast.toy:22:30
          var: a @test/Examples/Toy/Ch1/ast.toy:22:33
        ]
      VarDecl e<> @test/Examples/Toy/Ch1/ast.toy:25:3
        Call 'multiply_transpose' [ @test/Examples/Toy/Ch1/ast.toy:25:11
          var: c @test/Examples/Toy/Ch1/ast.toy:25:30
          var: d @test/Examples/Toy/Ch1/ast.toy:25:33
        ]
      VarDecl f<> @test/Examples/Toy/Ch1/ast.toy:28:3
        Call 'multiply_transpose' [ @test/Examples/Toy/Ch1/ast.toy:28:11
          var: a @test/Examples/Toy/Ch1/ast.toy:28:30
          var: c @test/Examples/Toy/Ch1/ast.toy:28:33
        ]
    } // Block
```
The module contains two functions, one called `multiplty_transpose`, and one called `main`. `multiply_transpose` returns something, while `main` does not. Each function contains a single a block. `multiply_transpose`'s block contains a `return` containing a `binop`. `main`'s block contains a list of `VarDecl`s.

*We will gloss over the frontend lexer and parser for now :P*

## Ch 2

"At the same time, IR elements can always be reduced to the above fundamental concepts."
Which concepts? These ones!

- A name for the operation.
- A list of SSA operand values.
- A list of attributes.
- A list of types for result values.
- A source location for debugging purposes.
- A list of successors blocks (for branches, mostly).
- A list of regions (for structural operations like functions).

Let's break down a function in MLIR assembly using the above concepts...
```
%t_tensor = "toy.transpose"(%tensor) {inplace = true} : (tensor<2x3xf64>) -> tensor<3x2xf64> loc("example/file/path":12:1)
              ^ name of operand ^     ^ attribute            ^ input type      ^ output type  ^ source location (mandatory!)
                                |
                              one SSA operand value
```
Set `build_root` to and `/home/hoppip/llvm-project-pistachio/build-riscv` and `mlir_src_root` to `/home/hoppip/llvm-project-pistachio/mlir`, then do
```
export build_root="/home/hoppip/llvm-project-pistachio/build-riscv"
export mlir_src_root="/home/hoppip/llvm-project-pistachio/mlir"

${build_root}/bin/mlir-tblgen -gen-dialect-decls ${mlir_src_root}/examples/toy/Ch2/include/toy/Ops.td -I ${mlir_src_root}/include/
```
to see what tablegen generates! (tablegen uses an ODS file to generate its cpp code)

- "**Traits** are a mechanism with which we can *inject additional behavior into an Operation*, such as additional accessors, verification, and more."

# Transform Dialect Ch 0

```
%2 = arith.addf %0, %1 : f32
```

I think this is saying, "assign %2 the result of performing arith dialect's addf function which takes two operands, %0, and %1, and returns a value of type f32" Which is probably a 32 bit float???

TODO: take in a tiny C program and produce linalg. DO THIS FIRST THING TOMORROW MORNING!!!!
