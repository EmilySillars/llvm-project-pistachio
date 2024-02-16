; ModuleID = '<stdin>'
source_filename = "pistachio-notes/tests/noAlias.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-linux-gnu"

; Function Attrs: nounwind uwtable
define dso_local i32 @foo(i32 noundef %0) #0 {
  %2 = alloca i32, align 4
  %3 = alloca i8*, align 8
  %4 = alloca i8, align 1
  %5 = alloca i8, align 1
  store i32 %0, i32* %2, align 4, !tbaa !5
  %6 = bitcast i8** %3 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* %6) #2
  %7 = bitcast i32* %2 to i8*
  store i8* %7, i8** %3, align 8, !tbaa !9
  %8 = load i8*, i8** %3, align 8, !tbaa !9
  %9 = getelementptr inbounds i8, i8* %8, i64 0
  store i8 77, i8* %9, align 1, !tbaa !11
  call void @llvm.lifetime.start.p0i8(i64 1, i8* %4) #2
  %10 = load i8*, i8** %3, align 8, !tbaa !9
  %11 = getelementptr inbounds i8, i8* %10, i64 1
  %12 = load i8, i8* %11, align 1, !tbaa !11
  store i8 %12, i8* %4, align 1, !tbaa !11
  call void @llvm.lifetime.start.p0i8(i64 1, i8* %5) #2
  %13 = load i8*, i8** %3, align 8, !tbaa !9
  %14 = getelementptr inbounds i8, i8* %13, i64 2
  %15 = load i8, i8* %14, align 1, !tbaa !11
  store i8 %15, i8* %5, align 1, !tbaa !11
  call void @llvm.lifetime.end.p0i8(i64 1, i8* %5) #2
  call void @llvm.lifetime.end.p0i8(i64 1, i8* %4) #2
  %16 = bitcast i8** %3 to i8*
  call void @llvm.lifetime.end.p0i8(i64 8, i8* %16) #2
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
  %2 = call i32 @foo(i32 noundef 42)
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
!11 = !{!7, !7, i64 0}
