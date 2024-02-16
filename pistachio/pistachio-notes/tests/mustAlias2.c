#include <stdio.h>

int foo(){
    int a = 4;
    int *ptrToA = &a; 
    int b = a;
    int c = *ptrToA;
    return 76;                 
}

/* LLVM
define dso_local i32 @foo() #0 {
  %1 = alloca i32, align 4
  %2 = alloca i32*, align 8
  %3 = alloca i32, align 4
  %4 = alloca i32, align 4
  %5 = bitcast i32* %1 to i8*
  call void @llvm.lifetime.start.p0i8(i64 4, i8* %5) #2
  store i32 4, i32* %1, align 4, !tbaa !5         <-------------------------------------------.---.
  %6 = bitcast i32** %2 to i8*                                                                |   |
  call void @llvm.lifetime.start.p0i8(i64 8, i8* %6) #2                                       |   |
  store i32* %1, i32** %2, align 8, !tbaa !9    // store address of %1 at %2                  |   |
  %7 = bitcast i32* %3 to i8*                                                                 |   |----- MustAlias; %1
  call void @llvm.lifetime.start.p0i8(i64 4, i8* %7) #2                                       |   |      
  %8 = load i32, i32* %1, align 4, !tbaa !5       <-----------------------------------------------'      
  store i32 %8, i32* %3, align 4, !tbaa !5      // b = a;                                     |   
  %9 = bitcast i32* %4 to i8*                                                                 |   
  call void @llvm.lifetime.start.p0i8(i64 4, i8* %9) #2                                       |--------- MayAlias; %10 is BASED ON %1   
  %10 = load i32*, i32** %2, align 8, !tbaa !9  // %10 gets dereference of %2, which is %1    |   
  %11 = load i32, i32* %10, align 4, !tbaa !5     <-------------------------------------------'
  store i32 %11, i32* %4, align 4, !tbaa !5     // c = *ptrToA;
  %12 = bitcast i32* %4 to i8*
  call void @llvm.lifetime.end.p0i8(i64 4, i8* %12) #2
  %13 = bitcast i32* %3 to i8*
  call void @llvm.lifetime.end.p0i8(i64 4, i8* %13) #2
  %14 = bitcast i32** %2 to i8*
  call void @llvm.lifetime.end.p0i8(i64 8, i8* %14) #2
  %15 = bitcast i32* %1 to i8*
  call void @llvm.lifetime.end.p0i8(i64 4, i8* %15) #2
  ret i32 76
}
*/

/* Results
Function foo:

  store i32 4, ptr %1, align 4, !tbaa !5 ALIASES WITH...
	  %8 = load i32, ptr %1, align 4, !tbaa !5 ( MustAlias )
	  %11 = load i32, ptr %10, align 4, !tbaa !5 ( MayAlias )

  store ptr %1, ptr %2, align 8, !tbaa !9 ALIASES WITH...
	  %10 = load ptr, ptr %2, align 8, !tbaa !9 ( MustAlias )

  store i32 %8, ptr %3, align 4, !tbaa !5 DOES NOT ALIAS WITH ANY LOADS

  store i32 %11, ptr %4, align 4, !tbaa !5 DOES NOT ALIAS WITH ANY LOADS

Alias Set Tracker: 4 alias sets for 5 pointer values.
  AliasSet[0x56291d10e660, 2] may alias, Mod/Ref   Pointers: (ptr %1, LocationSize::precise(4)), (ptr %10, LocationSize::precise(4))
  AliasSet[0x56291d10e700, 1] must alias, Mod/Ref   Pointers: (ptr %2, LocationSize::precise(8))
  AliasSet[0x56291d10e7a0, 1] must alias, Mod       Pointers: (ptr %3, LocationSize::precise(4))
  AliasSet[0x56291d10ebf0, 1] must alias, Mod       Pointers: (ptr %4, LocationSize::precise(4))
*/

int main() {
  return foo();
}