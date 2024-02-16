#include <stdio.h>
#include <stdint.h>

int foo(int32_t a){
    int8_t *p = (int8_t *) &a;
    p[0] = 77;
    int8_t b = p[1];
    int8_t c = p[2];
    printf("hoodle\n");
    return 76;                 
}

/* LLVM
define dso_local i32 @foo(i32 noundef %0) #0 {
  %2 = alloca i32, align 4  // a
  %3 = alloca i8*, align 8  // p
  %4 = alloca i8, align 1   // b
  %5 = alloca i8, align 1   // c
  store i32 %0, i32* %2, align 4, !tbaa !5           // store parameter a at %2
  %6 = bitcast i8** %3 to i8*                        
  call void @llvm.lifetime.start.p0i8(i64 8, i8* %6) #2
  %7 = bitcast i32* %2 to i8*                        // %7 is the address of the first byte of a, a[0]
  store i8* %7, i8** %3, align 8, !tbaa !9           // store %7 at %3
  %8 = load i8*, i8** %3, align 8, !tbaa !9          // %8 gets dereference of %3, which is %7
  %9 = getelementptr inbounds i8, i8* %8, i64 0      // %9 gets address of a[0]
  store i8 77, i8* %9, align 1, !tbaa !11            // store 77 to a[0] (or to %9)
  call void @llvm.lifetime.start.p0i8(i64 1, i8* %4) #2
  %10 = load i8*, i8** %3, align 8, !tbaa !9         // %10 gets dereference of %3, which is %7   
  %11 = getelementptr inbounds i8, i8* %10, i64 1    // %11 gets address of a[1]
  %12 = load i8, i8* %11, align 1, !tbaa !11         // %12 gets dereference of %11, which is a[1]; a[1] is NOT a[0]
  store i8 %12, i8* %4, align 1, !tbaa !11           // store this a[1] to %4
  call void @llvm.lifetime.start.p0i8(i64 1, i8* %5) #2
  %13 = load i8*, i8** %3, align 8, !tbaa !9         // %13 gets dereference of %3, which is %7
  %14 = getelementptr inbounds i8, i8* %13, i64 2    // %14 gets address of a[2]
  %15 = load i8, i8* %14, align 1, !tbaa !11         // %15 gets dereference of %14, which is a[2]; a[2] is NOT a[0]
  store i8 %15, i8* %5, align 1, !tbaa !11
  call void @llvm.lifetime.end.p0i8(i64 1, i8* %5) #2
  call void @llvm.lifetime.end.p0i8(i64 1, i8* %4) #2
  %16 = bitcast i8** %3 to i8*
  call void @llvm.lifetime.end.p0i8(i64 8, i8* %16) #2
  ret i32 76
}
*/

/* Results
Function foo:

  store i32 %0, ptr %2, align 4, !tbaa !5 ALIASES WITH...
	  %12 = load i8, ptr %11, align 1, !tbaa !11 ( MayAlias )
	  %15 = load i8, ptr %14, align 1, !tbaa !11 ( MayAlias )

  store ptr %7, ptr %3, align 8, !tbaa !9 ALIASES WITH...
	  %8 = load ptr, ptr %3, align 8, !tbaa !9 ( MustAlias )
	  %10 = load ptr, ptr %3, align 8, !tbaa !9 ( MustAlias )
	  %13 = load ptr, ptr %3, align 8, !tbaa !9 ( MustAlias )

  store i8 77, ptr %9, align 1, !tbaa !11 ALIASES WITH...
	  %12 = load i8, ptr %11, align 1, !tbaa !11 ( MayAlias )
	  %15 = load i8, ptr %14, align 1, !tbaa !11 ( MayAlias )

  store i8 %12, ptr %4, align 1, !tbaa !11 DOES NOT ALIAS WITH ANY LOADS

  store i8 %15, ptr %5, align 1, !tbaa !11 DOES NOT ALIAS WITH ANY LOADS

Alias Set Tracker: 4 alias sets for 7 pointer values.
  AliasSet[0x55db0d0647a0, 4] may alias, Mod/Ref   Pointers: (ptr %2, LocationSize::precise(4)), (ptr %9, LocationSize::precise(1)), (ptr %11, LocationSize::precise(1)), (ptr %14, LocationSize::precise(1))
  AliasSet[0x55db0d064840, 1] must alias, Mod/Ref   Pointers: (ptr %3, LocationSize::precise(8))
  AliasSet[0x55db0d064ce0, 1] must alias, Mod       Pointers: (ptr %4, LocationSize::precise(1))
  AliasSet[0x55db0d064e20, 1] must alias, Mod       Pointers: (ptr %5, LocationSize::precise(1))
*/

int main() {
  return foo(42);
}