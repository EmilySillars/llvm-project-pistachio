#include <stdio.h>
#include <stdint.h>

struct Bag {
  int ball;
  int bat;
};

int foo(){             
    int32_t a = 9;              // a: [ 0][ 0][ 0][ 9]
    int8_t *p = (int8_t *) &a;  // byte pointer
    int16_t*p2 = (int16_t *)&a; // 2-byte pointer
    p2[0] = 85;                 // a: [ 0][85][ 0][ 9]
    p[0] = 77;                  // a: [77][85][ 0][ 9]
    int8_t b = p[0];            // b: [77]
    int8_t c = p[1];            // c: [85]
    return 76;                  
}

/* LLVM
; Function Attrs: nounwind uwtable
define dso_local i32 @foo() #0 {
  %1 = alloca i32, align 4
  %2 = alloca i8*, align 8
  %3 = alloca i16*, align 8
  %4 = alloca i8, align 1
  %5 = alloca i8, align 1
  %6 = bitcast i32* %1 to i8*
  call void @llvm.lifetime.start.p0i8(i64 4, i8* %6) #2
  store i32 9, i32* %1, align 4, !tbaa !5
  %7 = bitcast i8** %2 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* %7) #2
  %8 = bitcast i32* %1 to i8*
  store i8* %8, i8** %2, align 8, !tbaa !9
  %9 = bitcast i16** %3 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* %9) #2
  %10 = bitcast i32* %1 to i16*
  store i16* %10, i16** %3, align 8, !tbaa !9
  %11 = load i16*, i16** %3, align 8, !tbaa !9
  %12 = getelementptr inbounds i16, i16* %11, i64 0
  store i16 85, i16* %12, align 2, !tbaa !11
  %13 = load i8*, i8** %2, align 8, !tbaa !9
  %14 = getelementptr inbounds i8, i8* %13, i64 0
  store i8 77, i8* %14, align 1, !tbaa !13
  call void @llvm.lifetime.start.p0i8(i64 1, i8* %4) #2
  %15 = load i8*, i8** %2, align 8, !tbaa !9
  %16 = getelementptr inbounds i8, i8* %15, i64 0
  %17 = load i8, i8* %16, align 1, !tbaa !13
  store i8 %17, i8* %4, align 1, !tbaa !13
  call void @llvm.lifetime.start.p0i8(i64 1, i8* %5) #2
  %18 = load i8*, i8** %2, align 8, !tbaa !9
  %19 = getelementptr inbounds i8, i8* %18, i64 1
  %20 = load i8, i8* %19, align 1, !tbaa !13
  store i8 %20, i8* %5, align 1, !tbaa !13
  call void @llvm.lifetime.end.p0i8(i64 1, i8* %5) #2
  call void @llvm.lifetime.end.p0i8(i64 1, i8* %4) #2
  %21 = bitcast i16** %3 to i8*
  call void @llvm.lifetime.end.p0i8(i64 8, i8* %21) #2
  %22 = bitcast i8** %2 to i8*
  call void @llvm.lifetime.end.p0i8(i64 8, i8* %22) #2
  %23 = bitcast i32* %1 to i8*
  call void @llvm.lifetime.end.p0i8(i64 4, i8* %23) #2
  ret i32 76
}
*/

/* Results
Function foo:

  store i32 9, ptr %1, align 4, !tbaa !5 ALIASES WITH...
	  %17 = load i8, ptr %16, align 1, !tbaa !13 ( MayAlias )
	  %20 = load i8, ptr %19, align 1, !tbaa !13 ( MayAlias )

  store ptr %8, ptr %2, align 8, !tbaa !9 ALIASES WITH...
	  %13 = load ptr, ptr %2, align 8, !tbaa !9 ( MustAlias )
	  %15 = load ptr, ptr %2, align 8, !tbaa !9 ( MustAlias )
	  %18 = load ptr, ptr %2, align 8, !tbaa !9 ( MustAlias )

  store ptr %10, ptr %3, align 8, !tbaa !9 ALIASES WITH...
	  %11 = load ptr, ptr %3, align 8, !tbaa !9 ( MustAlias )

  store i16 85, ptr %12, align 2, !tbaa !11 ALIASES WITH...
	  %17 = load i8, ptr %16, align 1, !tbaa !13 ( MayAlias )
	  %20 = load i8, ptr %19, align 1, !tbaa !13 ( MayAlias )

  store i8 77, ptr %14, align 1, !tbaa !13 ALIASES WITH...
	  %17 = load i8, ptr %16, align 1, !tbaa !13 ( MayAlias )
	  %20 = load i8, ptr %19, align 1, !tbaa !13 ( MayAlias )

  store i8 %17, ptr %4, align 1, !tbaa !13 DOES NOT ALIAS WITH ANY LOADS

  store i8 %20, ptr %5, align 1, !tbaa !13 DOES NOT ALIAS WITH ANY LOADS

Total Alias Queries: 42.00
MustAlias: 9.52% May Alias: 14.29% Partial Alias: 0.00% NoAlias: 76.19%

Alias Set Tracker: 5 alias sets for 15 pointer values.
  AliasSet[0x55b2b8087c40, 7] may alias, Mod/Ref   Pointers: (ptr %6, LocationSize::precise(4)), (ptr %1, LocationSize::precise(4)), (ptr %12, LocationSize::precise(2)), (ptr %14, LocationSize::precise(1)), (ptr %16, LocationSize::precise(1)), (ptr %19, LocationSize::precise(1)), (ptr %23, LocationSize::precise(4))
  AliasSet[0x55b2b8087d30, 3] must alias, Mod/Ref   Pointers: (ptr %7, LocationSize::precise(8)), (ptr %2, LocationSize::precise(8)), (ptr %22, LocationSize::precise(8))
  AliasSet[0x55b2b8087e20, 3] must alias, Mod/Ref   Pointers: (ptr %9, LocationSize::precise(8)), (ptr %3, LocationSize::precise(8)), (ptr %21, LocationSize::precise(8))
  AliasSet[0x55b2b8088350, 1] must alias, Mod/Ref   Pointers: (ptr %4, LocationSize::precise(1))
  AliasSet[0x55b2b80884e0, 1] must alias, Mod/Ref   Pointers: (ptr %5, LocationSize::precise(1))
*/

int main() {
  return foo();
}