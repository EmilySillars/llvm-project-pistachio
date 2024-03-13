; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
; vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv copied from simple-c.ll ---------------------------|
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@.str.1 = private unnamed_addr constant [4 x i8] c"%d\0A\00", align 1
@str = private unnamed_addr constant [19 x i8] c"yodelaheehoooo~~~!\00", align 1

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local i32 @yodel(i32 noundef %a) local_unnamed_addr #0 {
entry:
  %add = shl nsw i32 %a, 1
  ret i32 %add
}

; Function Attrs: nofree nounwind uwtable
define dso_local i32 @main() local_unnamed_addr #1 {
entry:
  %puts = tail call i32 @puts(ptr nonnull dereferenceable(1) @str)
  %call2 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, i32 noundef 154)
  ret i32 0
}

; Function Attrs: nofree nounwind
declare noundef i32 @printf(ptr nocapture noundef readonly, ...) local_unnamed_addr #2

; Function Attrs: nofree nounwind
declare noundef i32 @puts(ptr nocapture noundef readonly) local_unnamed_addr #3
; ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ copied from simple-c.ll ---------------------------|
declare ptr @malloc(i64)

define { ptr, ptr, i64, [1 x i64], [1 x i64] } @foo(ptr %0, ptr %1, i64 %2, i64 %3, i64 %4, ptr %5, ptr %6, i64 %7, i64 %8, i64 %9) {
  %11 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } undef, ptr %0, 0
  %12 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %11, ptr %1, 1
  %13 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %12, i64 %2, 2
  %14 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %13, i64 %3, 3, 0
  %15 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %14, i64 %4, 4, 0
  %16 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } undef, ptr %5, 0
  %17 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %16, ptr %6, 1
  %18 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %17, i64 %7, 2
  %19 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %18, i64 %8, 3, 0
  %20 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %19, i64 %9, 4, 0
  %21 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 10) to i64), i64 64))
  %22 = ptrtoint ptr %21 to i64
  %23 = add i64 %22, 63
  %24 = urem i64 %23, 64
  %25 = sub i64 %23, %24
  %26 = inttoptr i64 %25 to ptr
  %27 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } undef, ptr %21, 0
  %28 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %27, ptr %26, 1
  %29 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %28, i64 0, 2
  %30 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %29, i64 10, 3, 0
  %31 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %30, i64 1, 4, 0
  br label %32

32:                                               ; preds = %35, %10
  %33 = phi i64 [ %46, %35 ], [ 0, %10 ]
  %34 = icmp slt i64 %33, 10
  br i1 %34, label %35, label %47

35:                                               ; preds = %32
  %36 = getelementptr float, ptr %1, i64 %2
  %37 = mul i64 %33, %4
  %38 = getelementptr float, ptr %36, i64 %37
  %39 = load float, ptr %38, align 4
  %40 = getelementptr float, ptr %6, i64 %7
  %41 = mul i64 %33, %9
  %42 = getelementptr float, ptr %40, i64 %41
  %43 = load float, ptr %42, align 4
  %44 = fadd float %39, %43
  %45 = getelementptr float, ptr %26, i64 %33
  store float %44, ptr %45, align 4
  %46 = add i64 %33, 1
  br label %32

47:                                               ; preds = %32
  ret { ptr, ptr, i64, [1 x i64], [1 x i64] } %31
}

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
