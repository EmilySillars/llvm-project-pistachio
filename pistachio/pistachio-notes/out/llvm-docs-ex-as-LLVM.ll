; ModuleID = '<stdin>'
source_filename = "pistachio-notes/tests/llvm-docs-ex.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-linux-gnu"

; Function Attrs: nounwind uwtable
define dso_local void @foo() #0 {
  %1 = alloca i32, align 4
  %2 = alloca [2 x i8], align 1
  %3 = alloca [10 x i8], align 1
  %4 = bitcast i32* %1 to i8*
  call void @llvm.lifetime.start.p0i8(i64 4, i8* %4) #2
  %5 = bitcast [2 x i8]* %2 to i8*
  call void @llvm.lifetime.start.p0i8(i64 2, i8* %5) #2
  %6 = bitcast [10 x i8]* %3 to i8*
  call void @llvm.lifetime.start.p0i8(i64 10, i8* %6) #2
  store i32 0, i32* %1, align 4, !tbaa !5
  br label %7

7:                                                ; preds = %22, %0
  %8 = load i32, i32* %1, align 4, !tbaa !5
  %9 = icmp ne i32 %8, 10
  br i1 %9, label %10, label %25

10:                                               ; preds = %7
  %11 = load i32, i32* %1, align 4, !tbaa !5
  %12 = sext i32 %11 to i64
  %13 = getelementptr inbounds [10 x i8], [10 x i8]* %3, i64 0, i64 %12
  %14 = load i8, i8* %13, align 1, !tbaa !9
  %15 = getelementptr inbounds [2 x i8], [2 x i8]* %2, i64 0, i64 0
  store i8 %14, i8* %15, align 1, !tbaa !9
  %16 = load i32, i32* %1, align 4, !tbaa !5
  %17 = sub nsw i32 9, %16
  %18 = sext i32 %17 to i64
  %19 = getelementptr inbounds [10 x i8], [10 x i8]* %3, i64 0, i64 %18
  %20 = load i8, i8* %19, align 1, !tbaa !9
  %21 = getelementptr inbounds [2 x i8], [2 x i8]* %2, i64 0, i64 1
  store i8 %20, i8* %21, align 1, !tbaa !9
  br label %22

22:                                               ; preds = %10
  %23 = load i32, i32* %1, align 4, !tbaa !5
  %24 = add nsw i32 %23, 1
  store i32 %24, i32* %1, align 4, !tbaa !5
  br label %7, !llvm.loop !10

25:                                               ; preds = %7
  %26 = bitcast [10 x i8]* %3 to i8*
  call void @llvm.lifetime.end.p0i8(i64 10, i8* %26) #2
  %27 = bitcast [2 x i8]* %2 to i8*
  call void @llvm.lifetime.end.p0i8(i64 2, i8* %27) #2
  %28 = bitcast i32* %1 to i8*
  call void @llvm.lifetime.end.p0i8(i64 4, i8* %28) #2
  ret void
}

; Function Attrs: argmemonly nofree nosync nounwind willreturn
declare void @llvm.lifetime.start.p0i8(i64 immarg, i8* nocapture) #1

; Function Attrs: argmemonly nofree nosync nounwind willreturn
declare void @llvm.lifetime.end.p0i8(i64 immarg, i8* nocapture) #1

; Function Attrs: nounwind uwtable
define dso_local void @bar() #0 {
  %1 = alloca i32, align 4
  %2 = alloca [2 x i8], align 1
  %3 = alloca [10 x i8], align 1
  %4 = bitcast i32* %1 to i8*
  call void @llvm.lifetime.start.p0i8(i64 4, i8* %4) #2
  %5 = bitcast [2 x i8]* %2 to i8*
  call void @llvm.lifetime.start.p0i8(i64 2, i8* %5) #2
  %6 = bitcast [10 x i8]* %3 to i8*
  call void @llvm.lifetime.start.p0i8(i64 10, i8* %6) #2
  store i32 0, i32* %1, align 4, !tbaa !5
  br label %7

7:                                                ; preds = %25, %0
  %8 = load i32, i32* %1, align 4, !tbaa !5
  %9 = icmp ne i32 %8, 10
  br i1 %9, label %10, label %28

10:                                               ; preds = %7
  %11 = load i32, i32* %1, align 4, !tbaa !5
  %12 = sext i32 %11 to i64
  %13 = getelementptr inbounds [10 x i8], [10 x i8]* %3, i64 0, i64 %12
  %14 = load i8, i8* %13, align 1, !tbaa !9
  %15 = sext i8 %14 to i16
  %16 = getelementptr inbounds [2 x i8], [2 x i8]* %2, i64 0, i64 0
  %17 = bitcast i8* %16 to i16*
  %18 = getelementptr inbounds i16, i16* %17, i64 0
  store i16 %15, i16* %18, align 1, !tbaa !13
  %19 = load i32, i32* %1, align 4, !tbaa !5
  %20 = sub nsw i32 9, %19
  %21 = sext i32 %20 to i64
  %22 = getelementptr inbounds [10 x i8], [10 x i8]* %3, i64 0, i64 %21
  %23 = load i8, i8* %22, align 1, !tbaa !9
  %24 = getelementptr inbounds [2 x i8], [2 x i8]* %2, i64 0, i64 1
  store i8 %23, i8* %24, align 1, !tbaa !9
  br label %25

25:                                               ; preds = %10
  %26 = load i32, i32* %1, align 4, !tbaa !5
  %27 = add nsw i32 %26, 1
  store i32 %27, i32* %1, align 4, !tbaa !5
  br label %7, !llvm.loop !15

28:                                               ; preds = %7
  %29 = bitcast [10 x i8]* %3 to i8*
  call void @llvm.lifetime.end.p0i8(i64 10, i8* %29) #2
  %30 = bitcast [2 x i8]* %2 to i8*
  call void @llvm.lifetime.end.p0i8(i64 2, i8* %30) #2
  %31 = bitcast i32* %1 to i8*
  call void @llvm.lifetime.end.p0i8(i64 4, i8* %31) #2
  ret void
}

; Function Attrs: nounwind uwtable
define dso_local i32 @main() #0 {
  %1 = alloca i32, align 4
  store i32 0, i32* %1, align 4
  call void @foo()
  call void @bar()
  ret i32 5
}

attributes #0 = { nounwind uwtable "frame-pointer"="none" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #1 = { argmemonly nofree nosync nounwind willreturn }
attributes #2 = { nounwind }

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
!9 = !{!7, !7, i64 0}
!10 = distinct !{!10, !11, !12}
!11 = !{!"llvm.loop.mustprogress"}
!12 = !{!"llvm.loop.unroll.disable"}
!13 = !{!14, !14, i64 0}
!14 = !{!"short", !7, i64 0}
!15 = distinct !{!15, !11, !12}
