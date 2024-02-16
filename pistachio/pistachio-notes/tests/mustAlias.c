#include <stdio.h>

int foo(){
    int a = 4;
    int b = a;
    return 76;                 
}

/* LLVM
define dso_local i32 @foo() #0 {
  %1 = alloca i32, align 4
  %2 = alloca i32, align 4
  %3 = bitcast i32* %1 to i8*
  call void @llvm.lifetime.start.p0i8(i64 4, i8* %3) #2
  store i32 4, i32* %1, align 4, !tbaa !5       <--------------------.
  %4 = bitcast i32* %2 to i8*                                        |-------- MustAlias; share SAME ptr as target address
  call void @llvm.lifetime.start.p0i8(i64 4, i8* %4) #2              |
  %5 = load i32, i32* %1, align 4, !tbaa !5     <--------------------'
  store i32 %5, i32* %2, align 4, !tbaa !5      <--------------------- Does not alias with any loads.
  %6 = bitcast i32* %2 to i8*
  call void @llvm.lifetime.end.p0i8(i64 4, i8* %6) #2
  %7 = bitcast i32* %1 to i8*
  call void @llvm.lifetime.end.p0i8(i64 4, i8* %7) #2
  ret i32 76
}
*/

/* Results
Function foo:

  store i32 4, ptr %1, align 4, !tbaa !5 ALIASES WITH...
	  %5 = load i32, ptr %1, align 4, !tbaa !5 ( MustAlias )

  store i32 %5, ptr %2, align 4, !tbaa !5 DOES NOT ALIAS WITH ANY LOADS

Alias Set Tracker: 2 alias sets for 2 pointer values.
  AliasSet[0x561916918eb0, 1] must alias, Mod/Ref   Pointers: (ptr %1, LocationSize::precise(4))
  AliasSet[0x561916918f50, 1] must alias, Mod       Pointers: (ptr %2, LocationSize::precise(4))
*/

int main() {
  return foo();
}