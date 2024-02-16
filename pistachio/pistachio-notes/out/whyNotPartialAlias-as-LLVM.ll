; ModuleID = '<stdin>'
source_filename = "pistachio-notes/tests/whyNotPartialAlias.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-linux-gnu"

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

; Function Attrs: argmemonly nofree nosync nounwind willreturn
declare void @llvm.lifetime.start.p0i8(i64 immarg, i8* nocapture) #1

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
!9 = !{!10, !10, i64 0}
!10 = !{!"any pointer", !7, i64 0}
!11 = !{!12, !12, i64 0}
!12 = !{!"short", !7, i64 0}
!13 = !{!7, !7, i64 0}
