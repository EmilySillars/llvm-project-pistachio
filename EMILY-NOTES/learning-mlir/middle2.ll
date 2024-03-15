; vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv copied from simple-matrix.ll vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv-|
; vv                                                                                                             vv-|
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.fakeTensor = type { ptr, ptr, i64, [1 x i64], [1 x i64] }

@.str = private unnamed_addr constant [32 x i8] c"Trying to print fake tensor...\0A\00", align 1
@.str.1 = private unnamed_addr constant [12 x i8] c"dim1 is %p\0A\00", align 1
@.str.2 = private unnamed_addr constant [12 x i8] c"dim2 is %p\0A\00", align 1
@.str.3 = private unnamed_addr constant [11 x i8] c"len is %d\0A\00", align 1
@.str.4 = private unnamed_addr constant [15 x i8] c"dim1[0] is %f\0A\00", align 1
@.str.5 = private unnamed_addr constant [15 x i8] c"dim1[1] is %f\0A\00", align 1
@.str.6 = private unnamed_addr constant [15 x i8] c"dim2[0] is %f\0A\00", align 1
@.str.7 = private unnamed_addr constant [15 x i8] c"dim2[1] is %f\0A\00", align 1
@.str.8 = private unnamed_addr constant [12 x i8] c"a[0] is %d\0A\00", align 1
@.str.9 = private unnamed_addr constant [12 x i8] c"b[0] is %d\0A\00", align 1
@.str.10 = private unnamed_addr constant [20 x i8] c"yodelaheehoooo~~~!\0A\00", align 1
@__const.main.dim1 = private unnamed_addr constant [3 x float] [float 8.880000e+02, float 9.700000e+01, float 3.300000e+01], align 4
@__const.main.dim2 = private unnamed_addr constant [3 x float] [float 5.500000e+01, float 6.600000e+01, float 7.900000e+01], align 4

; Function Attrs: noinline nounwind optnone uwtable
define dso_local void @modifyTensor(ptr noalias sret(%struct.fakeTensor) align 8 %agg.result, ptr noundef byval(%struct.fakeTensor) align 8 %hoodle) #0 {
entry:
  %dim1 = getelementptr inbounds %struct.fakeTensor, ptr %hoodle, i32 0, i32 0
  %0 = load ptr, ptr %dim1, align 8
  %arrayidx = getelementptr inbounds float, ptr %0, i64 0
  store float 9.800000e+01, ptr %arrayidx, align 4
  call void @llvm.memcpy.p0.p0.i64(ptr align 8 %agg.result, ptr align 8 %hoodle, i64 40, i1 false)
  ret void
}

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: readwrite)
declare void @llvm.memcpy.p0.p0.i64(ptr noalias nocapture writeonly, ptr noalias nocapture readonly, i64, i1 immarg) #1

; Function Attrs: noinline nounwind optnone uwtable
define dso_local void @printTensor(ptr noalias sret(%struct.fakeTensor) align 8 %agg.result, ptr noundef byval(%struct.fakeTensor) align 8 %hoodle) #0 {
entry:
  %call = call i32 (ptr, ...) @printf(ptr noundef @.str)
  %dim1 = getelementptr inbounds %struct.fakeTensor, ptr %hoodle, i32 0, i32 0
  %0 = load ptr, ptr %dim1, align 8
  %call1 = call i32 (ptr, ...) @printf(ptr noundef @.str.1, ptr noundef %0)
  %dim12 = getelementptr inbounds %struct.fakeTensor, ptr %hoodle, i32 0, i32 0
  %1 = load ptr, ptr %dim12, align 8
  %call3 = call i32 (ptr, ...) @printf(ptr noundef @.str.2, ptr noundef %1)
  %len = getelementptr inbounds %struct.fakeTensor, ptr %hoodle, i32 0, i32 2
  %2 = load i64, ptr %len, align 8
  %call4 = call i32 (ptr, ...) @printf(ptr noundef @.str.3, i64 noundef %2)
  %dim15 = getelementptr inbounds %struct.fakeTensor, ptr %hoodle, i32 0, i32 0
  %3 = load ptr, ptr %dim15, align 8
  %arrayidx = getelementptr inbounds float, ptr %3, i64 0
  %4 = load float, ptr %arrayidx, align 4
  %conv = fpext float %4 to double
  %call6 = call i32 (ptr, ...) @printf(ptr noundef @.str.4, double noundef %conv)
  %dim17 = getelementptr inbounds %struct.fakeTensor, ptr %hoodle, i32 0, i32 0
  %5 = load ptr, ptr %dim17, align 8
  %arrayidx8 = getelementptr inbounds float, ptr %5, i64 1
  %6 = load float, ptr %arrayidx8, align 4
  %conv9 = fpext float %6 to double
  %call10 = call i32 (ptr, ...) @printf(ptr noundef @.str.5, double noundef %conv9)
  %dim2 = getelementptr inbounds %struct.fakeTensor, ptr %hoodle, i32 0, i32 1
  %7 = load ptr, ptr %dim2, align 8
  %arrayidx11 = getelementptr inbounds float, ptr %7, i64 0
  %8 = load float, ptr %arrayidx11, align 4
  %conv12 = fpext float %8 to double
  %call13 = call i32 (ptr, ...) @printf(ptr noundef @.str.6, double noundef %conv12)
  %dim214 = getelementptr inbounds %struct.fakeTensor, ptr %hoodle, i32 0, i32 1
  %9 = load ptr, ptr %dim214, align 8
  %arrayidx15 = getelementptr inbounds float, ptr %9, i64 1
  %10 = load float, ptr %arrayidx15, align 4
  %conv16 = fpext float %10 to double
  %call17 = call i32 (ptr, ...) @printf(ptr noundef @.str.7, double noundef %conv16)
  %a = getelementptr inbounds %struct.fakeTensor, ptr %hoodle, i32 0, i32 3
  %arrayidx18 = getelementptr inbounds [1 x i64], ptr %a, i64 0, i64 0
  %11 = load i64, ptr %arrayidx18, align 8
  %call19 = call i32 (ptr, ...) @printf(ptr noundef @.str.8, i64 noundef %11)
  %b = getelementptr inbounds %struct.fakeTensor, ptr %hoodle, i32 0, i32 4
  %arrayidx20 = getelementptr inbounds [1 x i64], ptr %b, i64 0, i64 0
  %12 = load i64, ptr %arrayidx20, align 8
  %call21 = call i32 (ptr, ...) @printf(ptr noundef @.str.9, i64 noundef %12)
  call void @llvm.memcpy.p0.p0.i64(ptr align 8 %agg.result, ptr align 8 %hoodle, i64 40, i1 false)
  ret void
}

declare i32 @printf(ptr noundef, ...) #2

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @main() #0 {
entry:
  %retval = alloca i32, align 4
  %dim1 = alloca [3 x float], align 4
  %dim2 = alloca [3 x float], align 4
  %ft = alloca %struct.fakeTensor, align 8
  %retValueJill = alloca %struct.fakeTensor, align 8
  %tmp = alloca %struct.fakeTensor, align 8
  store i32 0, ptr %retval, align 4
  %call = call i32 (ptr, ...) @printf(ptr noundef @.str.10)
  call void @llvm.memcpy.p0.p0.i64(ptr align 4 %dim1, ptr align 4 @__const.main.dim1, i64 12, i1 false)
  call void @llvm.memcpy.p0.p0.i64(ptr align 4 %dim2, ptr align 4 @__const.main.dim2, i64 12, i1 false)
  %arraydecay = getelementptr inbounds [3 x float], ptr %dim1, i64 0, i64 0
  %dim11 = getelementptr inbounds %struct.fakeTensor, ptr %ft, i32 0, i32 0
  store ptr %arraydecay, ptr %dim11, align 8
  %arraydecay2 = getelementptr inbounds [3 x float], ptr %dim2, i64 0, i64 0
  %dim23 = getelementptr inbounds %struct.fakeTensor, ptr %ft, i32 0, i32 1
  store ptr %arraydecay2, ptr %dim23, align 8
  call void @simp(ptr sret(%struct.fakeTensor) align 8 %retValueJill, ptr noundef byval(%struct.fakeTensor) align 8 %ft)
  call void @printTensor(ptr sret(%struct.fakeTensor) align 8 %tmp, ptr noundef byval(%struct.fakeTensor) align 8 %retValueJill)
  ret i32 0
}

attributes #0 = { noinline nounwind optnone uwtable "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #1 = { nocallback nofree nounwind willreturn memory(argmem: readwrite) }
attributes #2 = { "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }

; ^^                                                                                                               -|
; ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ copied from simple-matrix.ll -------------------------------------|