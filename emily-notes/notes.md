# Bonus Notes

[loop-utils] `top Loop` before any messing around is 

```
affine.for %arg3 = 0 to 104 {
  affine.for %arg4 = 0 to 104 {
    affine.for %arg5 = 0 to 104 {
      %0 = affine.load %arg0[%arg3, %arg5] : memref<104x104xi8, strided<[?, ?], offset: ?>>
      %1 = affine.load %arg1[%arg5, %arg4] : memref<104x104xi8, strided<[?, ?], offset: ?>>
      %2 = affine.load %arg2[%arg3, %arg4] : memref<104x104xi32, strided<[?, ?], offset: ?>>
      %3 = arith.extsi %0 : i8 to i32
      %4 = arith.extsi %1 : i8 to i32
      %5 = arith.muli %3, %4 : i32
      %6 = arith.addi %2, %5 : i32
      affine.store %6, %arg2[%arg3, %arg4] : memref<104x104xi32, strided<[?, ?], offset: ?>>
    }
  }
} 
```

## Hoodle.

[loop-utils] After **point loop** stuff, `top Loop` looks like

```
affine.for %arg3 = 0 to 0 {
  affine.for %arg4 = 0 to 0 {
    affine.for %arg5 = 0 to 0 {
      affine.for %arg6 = 0 to 104 {
        affine.for %arg7 = 0 to 104 {
          affine.for %arg8 = 0 to 104 {
            %0 = affine.load %arg0[%arg6, %arg8] : memref<104x104xi8, strided<[?, ?], offset: ?>>
            %1 = affine.load %arg1[%arg8, %arg7] : memref<104x104xi8, strided<[?, ?], offset: ?>>
            %2 = affine.load %arg2[%arg6, %arg7] : memref<104x104xi32, strided<[?, ?], offset: ?>>
            %3 = arith.extsi %0 : i8 to i32
            %4 = arith.extsi %1 : i8 to i32
            %5 = arith.muli %3, %4 : i32
            %6 = arith.addi %2, %5 : i32
            affine.store %6, %arg2[%arg6, %arg7] : memref<104x104xi32, strided<[?, ?], offset: ?>>
          }
        }
      }
    }
  }
} 
```

[loop-utils] After **space loop** stuff, `top Loop` looks like

```
affine.for %arg3 = 0 to 0 {
  affine.for %arg4 = 0 to 0 {
    affine.for %arg5 = 0 to 0 {
      affine.for %arg6 = 0 to 0 {
        affine.for %arg7 = 0 to 0 {
          affine.for %arg8 = 0 to 0 {
            affine.for %arg9 = 0 to 104 {
              affine.for %arg10 = 0 to 104 {
                affine.for %arg11 = 0 to 104 {
                  %0 = affine.load %arg0[%arg9, %arg11] : memref<104x104xi8, strided<[?, ?], offset: ?>>
                  %1 = affine.load %arg1[%arg11, %arg10] : memref<104x104xi8, strided<[?, ?], offset: ?>>
                  %2 = affine.load %arg2[%arg9, %arg10] : memref<104x104xi32, strided<[?, ?], offset: ?>>
                  %3 = arith.extsi %0 : i8 to i32
                  %4 = arith.extsi %1 : i8 to i32
                  %5 = arith.muli %3, %4 : i32
                  %6 = arith.addi %2, %5 : i32
                  affine.store %6, %arg2[%arg9, %arg10] : memref<104x104xi32, strided<[?, ?], offset: ?>>
                }
              }
            }
          }
        }
      }
    }
  }
} 
```

[loop-utils] `origLoops.back()` is 

```
affine.for %arg11 = 0 to 104 {
  %0 = affine.load %arg0[%arg9, %arg11] : memref<104x104xi8, strided<[?, ?], offset: ?>>
  %1 = affine.load %arg1[%arg11, %arg10] : memref<104x104xi8, strided<[?, ?], offset: ?>>
  %2 = affine.load %arg2[%arg9, %arg10] : memref<104x104xi32, strided<[?, ?], offset: ?>>
  %3 = arith.extsi %0 : i8 to i32
  %4 = arith.extsi %1 : i8 to i32
  %5 = arith.muli %3, %4 : i32
  %6 = arith.addi %2, %5 : i32
  affine.store %6, %arg2[%arg9, %arg10] : memref<104x104xi32, strided<[?, ?], offset: ?>>
} 
```

## Yodel

After `extractForInductionVars` and `constructTiledIndexSetHyperRect`,

tiledLoops holds 6 AffineForOps?

The first element is

```
affine.for %arg3 = 0 to 104 step 13 {
  affine.for %arg4 = 0 to 104 step 13 {
    affine.for %arg5 = 0 to 104 step 13 {
      affine.for %arg6 = affine_map<(d0) -> (d0)>(%arg3) to affine_map<(d0) -> (d0 + 13)>(%arg3) {
        affine.for %arg7 = affine_map<(d0) -> (d0)>(%arg4) to affine_map<(d0) -> (d0 + 13)>(%arg4) {
          affine.for %arg8 = affine_map<(d0) -> (d0)>(%arg5) to affine_map<(d0) -> (d0 + 13)>(%arg5) {
            affine.for %arg9 = 0 to 104 {
              affine.for %arg10 = 0 to 104 {
                affine.for %arg11 = 0 to 104 {
                  %0 = affine.load %arg0[%arg9, %arg11] : memref<104x104xi8, strided<[?, ?], offset: ?>>
                  %1 = affine.load %arg1[%arg11, %arg10] : memref<104x104xi8, strided<[?, ?], offset: ?>>
                  %2 = affine.load %arg2[%arg9, %arg10] : memref<104x104xi32, strided<[?, ?], offset: ?>>
                  %3 = arith.extsi %0 : i8 to i32
                  %4 = arith.extsi %1 : i8 to i32
                  %5 = arith.muli %3, %4 : i32
                  %6 = arith.addi %2, %5 : i32
                  affine.store %6, %arg2[%arg9, %arg10] : memref<104x104xi32, strided<[?, ?], offset: ?>>
                }
              }
            }
          }
        }
      }
    }
  }
} 
```

The last element is

```
affine.for %arg8 = affine_map<(d0) -> (d0)>(%arg5) to affine_map<(d0) -> (d0 + 13)>(%arg5) {
  affine.for %arg9 = 0 to 104 {
    affine.for %arg10 = 0 to 104 {
      affine.for %arg11 = 0 to 104 {
        %0 = affine.load %arg0[%arg9, %arg11] : memref<104x104xi8, strided<[?, ?], offset: ?>>
        %1 = affine.load %arg1[%arg11, %arg10] : memref<104x104xi8, strided<[?, ?], offset: ?>>
        %2 = affine.load %arg2[%arg9, %arg10] : memref<104x104xi32, strided<[?, ?], offset: ?>>
        %3 = arith.extsi %0 : i8 to i32
        %4 = arith.extsi %1 : i8 to i32
        %5 = arith.muli %3, %4 : i32
        %6 = arith.addi %2, %5 : i32
        affine.store %6, %arg2[%arg9, %arg10] : memref<104x104xi32, strided<[?, ?], offset: ?>>
      }
    }
  }
} 
```

`origLoops.size()` is 3 and `origLoops.back()` is

```
affine.for %arg11 = 0 to 104 {
  %0 = affine.load %arg0[%arg9, %arg11] : memref<104x104xi8, strided<[?, ?], offset: ?>>
  %1 = affine.load %arg1[%arg11, %arg10] : memref<104x104xi8, strided<[?, ?], offset: ?>>
  %2 = affine.load %arg2[%arg9, %arg10] : memref<104x104xi32, strided<[?, ?], offset: ?>>
  %3 = arith.extsi %0 : i8 to i32
  %4 = arith.extsi %1 : i8 to i32
  %5 = arith.muli %3, %4 : i32
  %6 = arith.addi %2, %5 : i32
  affine.store %6, %arg2[%arg9, %arg10] : memref<104x104xi32, strided<[?, ?], offset: ?>>
} 

```

## Replaced some vars

[affine-ad-hoc-loop-tile] `tiledLoops[0]` is

```
affine.for %arg3 = 0 to 104 step 13 {
  affine.for %arg4 = 0 to 104 step 13 {
    affine.for %arg5 = 0 to 104 step 13 {
      affine.for %arg6 = affine_map<(d0) -> (d0)>(%arg3) to affine_map<(d0) -> (d0 + 13)>(%arg3) {
        affine.for %arg7 = affine_map<(d0) -> (d0)>(%arg4) to affine_map<(d0) -> (d0 + 13)>(%arg4) {
          affine.for %arg8 = affine_map<(d0) -> (d0)>(%arg5) to affine_map<(d0) -> (d0 + 13)>(%arg5) {
            affine.for %arg9 = 0 to 104 {
              affine.for %arg10 = 0 to 104 {
                affine.for %arg11 = 0 to 104 {
                  %0 = affine.load %arg0[%arg6, %arg8] : memref<104x104xi8, strided<[?, ?], offset: ?>>
                  %1 = affine.load %arg1[%arg8, %arg7] : memref<104x104xi8, strided<[?, ?], offset: ?>>
                  %2 = affine.load %arg2[%arg6, %arg7] : memref<104x104xi32, strided<[?, ?], offset: ?>>
                  %3 = arith.extsi %0 : i8 to i32
                  %4 = arith.extsi %1 : i8 to i32
                  %5 = arith.muli %3, %4 : i32
                  %6 = arith.addi %2, %5 : i32
                  affine.store %6, %arg2[%arg6, %arg7] : memref<104x104xi32, strided<[?, ?], offset: ?>>
                }
              }
            }
          }
        }
      }
    }
  }
} 
```

[affine-ad-hoc-loop-tile] `tiledLoops[tiledLoops.size()-1]` is 

``` 
affine.for %arg8 = affine_map<(d0) -> (d0)>(%arg5) to affine_map<(d0) -> (d0 + 13)>(%arg5) {
  affine.for %arg9 = 0 to 104 {
    affine.for %arg10 = 0 to 104 {
      affine.for %arg11 = 0 to 104 {
        %0 = affine.load %arg0[%arg6, %arg8] : memref<104x104xi8, strided<[?, ?], offset: ?>>
        %1 = affine.load %arg1[%arg8, %arg7] : memref<104x104xi8, strided<[?, ?], offset: ?>>
        %2 = affine.load %arg2[%arg6, %arg7] : memref<104x104xi32, strided<[?, ?], offset: ?>>
        %3 = arith.extsi %0 : i8 to i32
        %4 = arith.extsi %1 : i8 to i32
        %5 = arith.muli %3, %4 : i32
        %6 = arith.addi %2, %5 : i32
        affine.store %6, %arg2[%arg6, %arg7] : memref<104x104xi32, strided<[?, ?], offset: ?>>
      }
    }
  }
}
```

[affine-ad-hoc-loop-tile] `origLoops.back()` is 

```
affine.for %arg11 = 0 to 104 {
  %0 = affine.load %arg0[%arg6, %arg8] : memref<104x104xi8, strided<[?, ?], offset: ?>>
  %1 = affine.load %arg1[%arg8, %arg7] : memref<104x104xi8, strided<[?, ?], offset: ?>>
  %2 = affine.load %arg2[%arg6, %arg7] : memref<104x104xi32, strided<[?, ?], offset: ?>>
  %3 = arith.extsi %0 : i8 to i32
  %4 = arith.extsi %1 : i8 to i32
  %5 = arith.muli %3, %4 : i32
  %6 = arith.addi %2, %5 : i32
  affine.store %6, %arg2[%arg6, %arg7] : memref<104x104xi32, strided<[?, ?], offset: ?>>
} 
```

So I think the only thing that changes was the innermost loop's body's ***indexing variables***.

## What finally comes out

```
func.func @matmul104x104(
%arg0: memref<104x104xi8, strided<[?, ?], offset: ?>>, 
%arg1: memref<104x104xi8, strided<[?, ?], offset: ?>>, 
%arg2: memref<104x104xi32, strided<[?, ?], offset: ?>>) -> memref<104x104xi32, strided<[?, ?], offset: ?>> {
    affine.for %arg3 = 0 to 104 step 13 {
      affine.for %arg4 = 0 to 104 step 13 {
        affine.for %arg5 = 0 to 104 step 13 {
          affine.for %arg6 = #map(%arg3) to #map1(%arg3) {
            affine.for %arg7 = #map(%arg4) to #map1(%arg4) {
              affine.for %arg8 = #map(%arg5) to #map1(%arg5) {
              }
            }
          }
        }
      }
    }
    return %arg2 : memref<104x104xi32, strided<[?, ?], offset: ?>>
  }

```

Where is the body of the for nested for loop???!!! This must be fixed.

FIXED IT. I NEEDED TO UNCOMMENT OUT THE FOLLOWING LINE INSIDE CONSTRUCTDUMMYLOOPNEST:

```
AdHocLoopTile::moveLoopBody(origLoops.back(), innermostPointLoop);
```

Actual Final Output

```
func.func @matmul104x104(
%arg0: memref<104x104xi8, strided<[?, ?], offset: ?>>, 
%arg1: memref<104x104xi8, strided<[?, ?], offset: ?>>, 
%arg2: memref<104x104xi32, strided<[?, ?], offset: ?>>) -> memref<104x104xi32, strided<[?, ?], offset: ?>> {
    affine.for %arg3 = 0 to 104 step 13 {
      affine.for %arg4 = 0 to 104 step 13 {
        affine.for %arg5 = 0 to 104 step 13 {
          affine.for %arg6 = #map(%arg3) to #map1(%arg3) {
            affine.for %arg7 = #map(%arg4) to #map1(%arg4) {
              affine.for %arg8 = #map(%arg5) to #map1(%arg5) {
                %0 = affine.load %arg0[%arg6, %arg8] : memref<104x104xi8, strided<[?, ?], offset: ?>>
                %1 = affine.load %arg1[%arg8, %arg7] : memref<104x104xi8, strided<[?, ?], offset: ?>>
                %2 = affine.load %arg2[%arg6, %arg7] : memref<104x104xi32, strided<[?, ?], offset: ?>>
                %3 = arith.extsi %0 : i8 to i32
                %4 = arith.extsi %1 : i8 to i32
                %5 = arith.muli %3, %4 : i32
                %6 = arith.addi %2, %5 : i32
                affine.store %6, %arg2[%arg6, %arg7] : memref<104x104xi32, strided<[?, ?], offset: ?>>
              }
            }
          }
        }
      }
    }
    return %arg2 : memref<104x104xi32, strided<[?, ?], offset: ?>>
}
```

## How to create the affine maps and put them in the for-loops???

Either `extractForInductionVars` or `constructTiledIndexSetHyperRect` does it.