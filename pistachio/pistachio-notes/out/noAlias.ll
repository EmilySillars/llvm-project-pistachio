; ModuleID = 'out/noAlias.bc'
source_filename = "pistachio-notes/tests/noAlias.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-linux-gnu"

; Function Attrs: nounwind uwtable
define dso_local i32 @foo(i32 noundef %0) #0 {
  %2 = alloca i32, align 4
  %3 = alloca ptr, align 8
  %4 = alloca i8, align 1
  %5 = alloca i8, align 1
  store i32 %0, ptr %2, align 4, !tbaa !5
  %6 = bitcast ptr %3 to ptr
  call void @llvm.lifetime.start.p0(i64 8, ptr %6) #2
  %7 = bitcast ptr %2 to ptr
  store ptr %7, ptr %3, align 8, !tbaa !9
  %8 = load ptr, ptr %3, align 8, !tbaa !9
  %9 = getelementptr inbounds i8, ptr %8, i64 0
  store i8 77, ptr %9, align 1, !tbaa !11
  call void @llvm.lifetime.start.p0(i64 1, ptr %4) #2
  %10 = load ptr, ptr %3, align 8, !tbaa !9
  %11 = getelementptr inbounds i8, ptr %10, i64 1
  %12 = load i8, ptr %11, align 1, !tbaa !11
  store i8 %12, ptr %4, align 1, !tbaa !11
  call void @llvm.lifetime.start.p0(i64 1, ptr %5) #2
  %13 = load ptr, ptr %3, align 8, !tbaa !9
  %14 = getelementptr inbounds i8, ptr %13, i64 2
  %15 = load i8, ptr %14, align 1, !tbaa !11
  store i8 %15, ptr %5, align 1, !tbaa !11
  call void @llvm.lifetime.end.p0(i64 1, ptr %5) #2
  call void @llvm.lifetime.end.p0(i64 1, ptr %4) #2
  %16 = bitcast ptr %3 to ptr
  call void @llvm.lifetime.end.p0(i64 8, ptr %16) #2
  ret i32 76
}

; Function Attrs: nounwind uwtable
define dso_local i32 @main() #0 {
  %1 = alloca i32, align 4
  store i32 0, ptr %1, align 4
  %2 = call i32 @foo(i32 noundef 42)
  ret i32 %2
}

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0(i64 immarg, ptr nocapture) #1

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.end.p0(i64 immarg, ptr nocapture) #1

attributes #0 = { nounwind uwtable "frame-pointer"="none" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
attributes #2 = { nounwind }

!llvm.module.flags = !{!0, !1, !2, !3}
!llvm.ident = !{!4}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
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
