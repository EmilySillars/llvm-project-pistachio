; ModuleID = '<stdin>'
source_filename = "print-fake-tensors.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.fakeTensor1D = type { ptr, ptr, i64, [1 x i64], [1 x i64] }

@.str = private unnamed_addr constant [32 x i8] c"Trying to print fake tensor...\0A\00", align 1
@.str.1 = private unnamed_addr constant [12 x i8] c"ptr0 is %p\0A\00", align 1
@.str.2 = private unnamed_addr constant [12 x i8] c"ptr1 is %p\0A\00", align 1
@.str.3 = private unnamed_addr constant [18 x i8] c"i64 field is %ld\0A\00", align 1
@.str.4 = private unnamed_addr constant [15 x i8] c"ptr0[0] is %f\0A\00", align 1
@.str.5 = private unnamed_addr constant [15 x i8] c"ptr0[1] is %f\0A\00", align 1
@.str.6 = private unnamed_addr constant [16 x i8] c"arr0[0] is %ld\0A\00", align 1
@.str.7 = private unnamed_addr constant [16 x i8] c"arr1[0] is %ld\0A\00", align 1
@.str.8 = private unnamed_addr constant [18 x i8] c"ptr1 points to [ \00", align 1
@.str.9 = private unnamed_addr constant [5 x i8] c"%f, \00", align 1
@.str.10 = private unnamed_addr constant [3 x i8] c"]\0A\00", align 1
@.str.11 = private unnamed_addr constant [20 x i8] c"yodelaheehoooo~~~!\0A\00", align 1
@__const.main.ptr0 = private unnamed_addr constant [3 x float] [float 8.880000e+02, float 9.700000e+01, float 3.300000e+01], align 4
@__const.main.ptr1 = private unnamed_addr constant [3 x float] [float 5.500000e+01, float 6.600000e+01, float 7.900000e+01], align 4

;PINEAPPLE
declare ptr @malloc(i64)

define { ptr, ptr, i64, [1 x i64], [1 x i64] } @simp(ptr %0, ptr %1, i64 %2, i64 %3, i64 %4) {
  %6 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 2) to i64), i64 64))
  %7 = ptrtoint ptr %6 to i64
  %8 = add i64 %7, 63
  %9 = urem i64 %8, 64
  %10 = sub i64 %8, %9
  %11 = inttoptr i64 %10 to ptr
  %12 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } undef, ptr %6, 0
  %13 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %12, ptr %11, 1
  %14 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %13, i64 0, 2
  %15 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %14, i64 2, 3, 0
  %16 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %15, i64 1, 4, 0
  store float 7.700000e+01, ptr %11, align 4
  %17 = getelementptr float, ptr %11, i32 1
  store float 7.800000e+01, ptr %17, align 4
  ret { ptr, ptr, i64, [1 x i64], [1 x i64] } %16
}

define { ptr, ptr, i64, [2 x i64], [2 x i64] } @simp2(ptr %0, ptr %1, i64 %2, i64 %3, i64 %4, i64 %5, i64 %6) {
  %8 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 27) to i64), i64 64))
  %9 = ptrtoint ptr %8 to i64
  %10 = add i64 %9, 63
  %11 = urem i64 %10, 64
  %12 = sub i64 %10, %11
  %13 = inttoptr i64 %12 to ptr
  %14 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %8, 0
  %15 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %14, ptr %13, 1
  %16 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %15, i64 0, 2
  %17 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %16, i64 3, 3, 0
  %18 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %17, i64 9, 3, 1
  %19 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %18, i64 9, 4, 0
  %20 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %19, i64 1, 4, 1
  %21 = getelementptr float, ptr %13, i64 0
  store float 7.700000e+01, ptr %21, align 4
  %22 = getelementptr float, ptr %13, i64 1
  store float 7.800000e+01, ptr %22, align 4
  ret { ptr, ptr, i64, [2 x i64], [2 x i64] } %20
}

define { ptr, ptr, i64, [3 x i64], [3 x i64] } @simp3(ptr %0, ptr %1, i64 %2, i64 %3, i64 %4, i64 %5, i64 %6, i64 %7, i64 %8) {
  %10 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 54) to i64), i64 64))
  %11 = ptrtoint ptr %10 to i64
  %12 = add i64 %11, 63
  %13 = urem i64 %12, 64
  %14 = sub i64 %12, %13
  %15 = inttoptr i64 %14 to ptr
  %16 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } undef, ptr %10, 0
  %17 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %16, ptr %15, 1
  %18 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %17, i64 0, 2
  %19 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %18, i64 2, 3, 0
  %20 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %19, i64 3, 3, 1
  %21 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %20, i64 9, 3, 2
  %22 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %21, i64 27, 4, 0
  %23 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %22, i64 9, 4, 1
  %24 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %23, i64 1, 4, 2
  %25 = getelementptr float, ptr %15, i64 0
  store float 7.700000e+01, ptr %25, align 4
  %26 = getelementptr float, ptr %15, i64 9
  store float 7.800000e+01, ptr %26, align 4
  ret { ptr, ptr, i64, [3 x i64], [3 x i64] } %24
}

define { ptr, ptr, i64, [1 x i64], [1 x i64] } @foo(ptr %0, ptr %1, i64 %2, i64 %3, i64 %4, ptr %5, ptr %6, i64 %7, i64 %8, i64 %9) {
  %11 = call ptr @malloc(i64 add (i64 ptrtoint (ptr getelementptr (float, ptr null, i32 3) to i64), i64 64))
  %12 = ptrtoint ptr %11 to i64
  %13 = add i64 %12, 63
  %14 = urem i64 %13, 64
  %15 = sub i64 %13, %14
  %16 = inttoptr i64 %15 to ptr
  %17 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } undef, ptr %11, 0
  %18 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %17, ptr %16, 1
  %19 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %18, i64 0, 2
  %20 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %19, i64 3, 3, 0
  %21 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %20, i64 1, 4, 0
  store float 5.000000e+00, ptr %16, align 4
  %22 = getelementptr float, ptr %16, i32 1
  store float 6.000000e+00, ptr %22, align 4
  %23 = getelementptr float, ptr %16, i32 2
  store float 7.000000e+00, ptr %23, align 4
  ret { ptr, ptr, i64, [1 x i64], [1 x i64] } %21
}
;PINEAPPLE


; Function Attrs: noinline nounwind optnone uwtable
define dso_local void @modifyTensor(ptr noalias sret(%struct.fakeTensor1D) align 8 %agg.result, ptr noundef byval(%struct.fakeTensor1D) align 8 %hoodle) #0 {
entry:
  %ptr0 = getelementptr inbounds %struct.fakeTensor1D, ptr %hoodle, i32 0, i32 0
  %0 = load ptr, ptr %ptr0, align 8
  %arrayidx = getelementptr inbounds float, ptr %0, i64 0
  store float 9.800000e+01, ptr %arrayidx, align 4
  call void @llvm.memcpy.p0.p0.i64(ptr align 8 %agg.result, ptr align 8 %hoodle, i64 40, i1 false)
  ret void
}

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: readwrite)
declare void @llvm.memcpy.p0.p0.i64(ptr noalias nocapture writeonly, ptr noalias nocapture readonly, i64, i1 immarg) #1

; Function Attrs: noinline nounwind optnone uwtable
define dso_local void @printTensor(ptr noalias sret(%struct.fakeTensor1D) align 8 %agg.result, ptr noundef byval(%struct.fakeTensor1D) align 8 %hoodle) #0 {
entry:
  %len = alloca i64, align 8
  %i = alloca i64, align 8
  %call = call i32 (ptr, ...) @printf(ptr noundef @.str)
  %ptr0 = getelementptr inbounds %struct.fakeTensor1D, ptr %hoodle, i32 0, i32 0
  %0 = load ptr, ptr %ptr0, align 8
  %call1 = call i32 (ptr, ...) @printf(ptr noundef @.str.1, ptr noundef %0)
  %ptr1 = getelementptr inbounds %struct.fakeTensor1D, ptr %hoodle, i32 0, i32 1
  %1 = load ptr, ptr %ptr1, align 8
  %call2 = call i32 (ptr, ...) @printf(ptr noundef @.str.2, ptr noundef %1)
  %i64 = getelementptr inbounds %struct.fakeTensor1D, ptr %hoodle, i32 0, i32 2
  %2 = load i64, ptr %i64, align 8
  %call3 = call i32 (ptr, ...) @printf(ptr noundef @.str.3, i64 noundef %2)
  %ptr04 = getelementptr inbounds %struct.fakeTensor1D, ptr %hoodle, i32 0, i32 0
  %3 = load ptr, ptr %ptr04, align 8
  %arrayidx = getelementptr inbounds float, ptr %3, i64 0
  %4 = load float, ptr %arrayidx, align 4
  %conv = fpext float %4 to double
  %call5 = call i32 (ptr, ...) @printf(ptr noundef @.str.4, double noundef %conv)
  %ptr06 = getelementptr inbounds %struct.fakeTensor1D, ptr %hoodle, i32 0, i32 0
  %5 = load ptr, ptr %ptr06, align 8
  %arrayidx7 = getelementptr inbounds float, ptr %5, i64 1
  %6 = load float, ptr %arrayidx7, align 4
  %conv8 = fpext float %6 to double
  %call9 = call i32 (ptr, ...) @printf(ptr noundef @.str.5, double noundef %conv8)
  %arr0 = getelementptr inbounds %struct.fakeTensor1D, ptr %hoodle, i32 0, i32 3
  %arrayidx10 = getelementptr inbounds [1 x i64], ptr %arr0, i64 0, i64 0
  %7 = load i64, ptr %arrayidx10, align 8
  %call11 = call i32 (ptr, ...) @printf(ptr noundef @.str.6, i64 noundef %7)
  %arr1 = getelementptr inbounds %struct.fakeTensor1D, ptr %hoodle, i32 0, i32 4
  %arrayidx12 = getelementptr inbounds [1 x i64], ptr %arr1, i64 0, i64 0
  %8 = load i64, ptr %arrayidx12, align 8
  %call13 = call i32 (ptr, ...) @printf(ptr noundef @.str.7, i64 noundef %8)
  %arr014 = getelementptr inbounds %struct.fakeTensor1D, ptr %hoodle, i32 0, i32 3
  %arrayidx15 = getelementptr inbounds [1 x i64], ptr %arr014, i64 0, i64 0
  %9 = load i64, ptr %arrayidx15, align 8
  store i64 %9, ptr %len, align 8
  %call16 = call i32 (ptr, ...) @printf(ptr noundef @.str.8)
  store i64 0, ptr %i, align 8
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %10 = load i64, ptr %i, align 8
  %11 = load i64, ptr %len, align 8
  %cmp = icmp ult i64 %10, %11
  br i1 %cmp, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %ptr118 = getelementptr inbounds %struct.fakeTensor1D, ptr %hoodle, i32 0, i32 1
  %12 = load ptr, ptr %ptr118, align 8
  %13 = load i64, ptr %i, align 8
  %arrayidx19 = getelementptr inbounds float, ptr %12, i64 %13
  %14 = load float, ptr %arrayidx19, align 4
  %conv20 = fpext float %14 to double
  %call21 = call i32 (ptr, ...) @printf(ptr noundef @.str.9, double noundef %conv20)
  br label %for.inc

for.inc:                                          ; preds = %for.body
  %15 = load i64, ptr %i, align 8
  %inc = add i64 %15, 1
  store i64 %inc, ptr %i, align 8
  br label %for.cond, !llvm.loop !6

for.end:                                          ; preds = %for.cond
  %call22 = call i32 (ptr, ...) @printf(ptr noundef @.str.10)
  call void @llvm.memcpy.p0.p0.i64(ptr align 8 %agg.result, ptr align 8 %hoodle, i64 40, i1 false)
  ret void
}

declare i32 @printf(ptr noundef, ...) #2

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @main() #0 {
entry:
  %retval = alloca i32, align 4
  %ptr0 = alloca [3 x float], align 4
  %ptr1 = alloca [3 x float], align 4
  %ft = alloca %struct.fakeTensor1D, align 8
  %retValueJill = alloca %struct.fakeTensor1D, align 8
  %tmp = alloca %struct.fakeTensor1D, align 8
  store i32 0, ptr %retval, align 4
  %call = call i32 (ptr, ...) @printf(ptr noundef @.str.11)
  call void @llvm.memcpy.p0.p0.i64(ptr align 4 %ptr0, ptr align 4 @__const.main.ptr0, i64 12, i1 false)
  call void @llvm.memcpy.p0.p0.i64(ptr align 4 %ptr1, ptr align 4 @__const.main.ptr1, i64 12, i1 false)
  %arraydecay = getelementptr inbounds [3 x float], ptr %ptr0, i64 0, i64 0
  %ptr01 = getelementptr inbounds %struct.fakeTensor1D, ptr %ft, i32 0, i32 0
  store ptr %arraydecay, ptr %ptr01, align 8
  %arraydecay2 = getelementptr inbounds [3 x float], ptr %ptr1, i64 0, i64 0
  %ptr13 = getelementptr inbounds %struct.fakeTensor1D, ptr %ft, i32 0, i32 1
  store ptr %arraydecay2, ptr %ptr13, align 8
  call void @simp(ptr sret(%struct.fakeTensor1D) align 8 %retValueJill, ptr noundef byval(%struct.fakeTensor1D) align 8 %ft)
  call void @printTensor(ptr sret(%struct.fakeTensor1D) align 8 %tmp, ptr noundef byval(%struct.fakeTensor1D) align 8 %retValueJill)
  ret i32 0
}

attributes #0 = { noinline nounwind optnone uwtable "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #1 = { nocallback nofree nounwind willreturn memory(argmem: readwrite) }
attributes #2 = { "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.ident = !{!5}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{i32 7, !"frame-pointer", i32 2}
!5 = !{!"clang version 18.0.0git (https://github.com/EmilySillars/llvm-project-pistachio.git 81abde31868d77f2f9e905de62ff895376032655)"}
!6 = distinct !{!6, !7}
!7 = !{!"llvm.loop.mustprogress"}
