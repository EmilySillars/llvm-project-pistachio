; ModuleID = '<stdin>'
source_filename = "pistachio-notes/tests/z-example3.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-linux-gnu"

@__const.foo.arr = private unnamed_addr constant [3 x i32] [i32 867, i32 777, i32 835], align 4

; Function Attrs: nounwind uwtable
define dso_local i32 @foo() #0 {
  %1 = alloca [3 x i32], align 4
  %2 = alloca i32, align 4
  %3 = alloca i32, align 4
  %4 = bitcast [3 x i32]* %1 to i8*
  call void @llvm.lifetime.start.p0i8(i64 12, i8* %4) #3
  %5 = bitcast [3 x i32]* %1 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %5, i8* align 4 bitcast ([3 x i32]* @__const.foo.arr to i8*), i64 12, i1 false)
  %6 = getelementptr inbounds [3 x i32], [3 x i32]* %1, i64 0, i64 1
  store i32 444, i32* %6, align 4, !tbaa !5
  %7 = bitcast i32* %2 to i8*
  call void @llvm.lifetime.start.p0i8(i64 4, i8* %7) #3
  %8 = getelementptr inbounds [3 x i32], [3 x i32]* %1, i64 0, i64 1
  %9 = load i32, i32* %8, align 4, !tbaa !5
  store i32 %9, i32* %2, align 4, !tbaa !5
  %10 = bitcast i32* %3 to i8*
  call void @llvm.lifetime.start.p0i8(i64 4, i8* %10) #3
  %11 = getelementptr inbounds [3 x i32], [3 x i32]* %1, i64 0, i64 1
  %12 = load i32, i32* %11, align 4, !tbaa !5
  store i32 %12, i32* %3, align 4, !tbaa !5
  %13 = load i32, i32* %2, align 4, !tbaa !5
  %14 = load i32, i32* %3, align 4, !tbaa !5
  %15 = add nsw i32 %13, %14
  %16 = bitcast i32* %3 to i8*
  call void @llvm.lifetime.end.p0i8(i64 4, i8* %16) #3
  %17 = bitcast i32* %2 to i8*
  call void @llvm.lifetime.end.p0i8(i64 4, i8* %17) #3
  %18 = bitcast [3 x i32]* %1 to i8*
  call void @llvm.lifetime.end.p0i8(i64 12, i8* %18) #3
  ret i32 %15
}

; Function Attrs: argmemonly nofree nosync nounwind willreturn
declare void @llvm.lifetime.start.p0i8(i64 immarg, i8* nocapture) #1

; Function Attrs: argmemonly nofree nounwind willreturn
declare void @llvm.memcpy.p0i8.p0i8.i64(i8* noalias nocapture writeonly, i8* noalias nocapture readonly, i64, i1 immarg) #2

; Function Attrs: argmemonly nofree nosync nounwind willreturn
declare void @llvm.lifetime.end.p0i8(i64 immarg, i8* nocapture) #1

; Function Attrs: nounwind uwtable
define dso_local i32 @main() #0 {
  %1 = alloca i32, align 4
  store i32 0, i32* %1, align 4
  %2 = call i32 @foo()
  ret i32 %2
}

attributes #0 = { nounwind uwtable "frame-pointer"="none" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #1 = { argmemonly nofree nosync nounwind willreturn }
attributes #2 = { argmemonly nofree nounwind willreturn }
attributes #3 = { nounwind }

!llvm.module.flags = !{!0, !1, !2, !3}
!llvm.ident = !{!4}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 7, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 1}
!4 = !{!"Ubuntu clang version 14.0.0-1ubuntu1.1"}
!5 = !{!6, !6, i64 0}
!6 = !{!"int", !7, i64 0}
!7 = !{!"omnipotent char", !8, i64 0}
!8 = !{!"Simple C/C++ TBAA"}
