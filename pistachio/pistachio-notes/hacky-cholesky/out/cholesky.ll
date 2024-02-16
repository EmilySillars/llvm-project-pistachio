; ModuleID = '<stdin>'
source_filename = "pistachio-notes/hacky-cholesky/hacky-cholesky.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-linux-gnu"

%struct._IO_FILE = type { i32, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, %struct._IO_marker*, %struct._IO_FILE*, i32, i32, i64, i16, i8, [1 x i8], i8*, i64, %struct._IO_codecvt*, %struct._IO_wide_data*, %struct._IO_FILE*, i8*, i64, i32, [20 x i8] }
%struct._IO_marker = type opaque
%struct._IO_codecvt = type opaque
%struct._IO_wide_data = type opaque

@.str = private unnamed_addr constant [1 x i8] zeroinitializer, align 1
@stderr = external global %struct._IO_FILE*, align 8
@.str.1 = private unnamed_addr constant [8 x i8] c"%0.2lf \00", align 1
@.str.2 = private unnamed_addr constant [2 x i8] c"\0A\00", align 1

; Function Attrs: nounwind uwtable
define dso_local i32 @main(i32 noundef %0, i8** noundef %1) #0 {
  %3 = alloca i32, align 4
  %4 = alloca i32, align 4
  %5 = alloca i8**, align 8
  %6 = alloca i32, align 4
  %7 = alloca [1024 x [1024 x double]]*, align 8
  %8 = alloca [1024 x double]*, align 8
  store i32 0, i32* %3, align 4
  store i32 %0, i32* %4, align 4, !tbaa !5
  store i8** %1, i8*** %5, align 8, !tbaa !9
  %9 = bitcast i32* %6 to i8*
  call void @llvm.lifetime.start.p0i8(i64 4, i8* %9) #6
  store i32 1024, i32* %6, align 4, !tbaa !5
  %10 = bitcast [1024 x [1024 x double]]** %7 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* %10) #6
  %11 = call i8* @polybench_alloc_data(i64 noundef 1048576, i32 noundef 8)
  %12 = bitcast i8* %11 to [1024 x [1024 x double]]*
  store [1024 x [1024 x double]]* %12, [1024 x [1024 x double]]** %7, align 8, !tbaa !9
  %13 = bitcast [1024 x double]** %8 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* %13) #6
  %14 = call i8* @polybench_alloc_data(i64 noundef 1024, i32 noundef 8)
  %15 = bitcast i8* %14 to [1024 x double]*
  store [1024 x double]* %15, [1024 x double]** %8, align 8, !tbaa !9
  %16 = load i32, i32* %6, align 4, !tbaa !5
  %17 = load [1024 x double]*, [1024 x double]** %8, align 8, !tbaa !9
  %18 = getelementptr inbounds [1024 x double], [1024 x double]* %17, i64 0, i64 0
  %19 = load [1024 x [1024 x double]]*, [1024 x [1024 x double]]** %7, align 8, !tbaa !9
  %20 = getelementptr inbounds [1024 x [1024 x double]], [1024 x [1024 x double]]* %19, i64 0, i64 0
  call void @init_array(i32 noundef %16, double* noundef %18, [1024 x double]* noundef %20)
  %21 = load i32, i32* %6, align 4, !tbaa !5
  %22 = load [1024 x double]*, [1024 x double]** %8, align 8, !tbaa !9
  %23 = getelementptr inbounds [1024 x double], [1024 x double]* %22, i64 0, i64 0
  %24 = load [1024 x [1024 x double]]*, [1024 x [1024 x double]]** %7, align 8, !tbaa !9
  %25 = getelementptr inbounds [1024 x [1024 x double]], [1024 x [1024 x double]]* %24, i64 0, i64 0
  call void @kernel_cholesky(i32 noundef %21, double* noundef %23, [1024 x double]* noundef %25)
  %26 = load i32, i32* %4, align 4, !tbaa !5
  %27 = icmp sgt i32 %26, 42
  br i1 %27, label %28, label %38

28:                                               ; preds = %2
  %29 = load i8**, i8*** %5, align 8, !tbaa !9
  %30 = getelementptr inbounds i8*, i8** %29, i64 0
  %31 = load i8*, i8** %30, align 8, !tbaa !9
  %32 = call i32 @strcmp(i8* noundef %31, i8* noundef getelementptr inbounds ([1 x i8], [1 x i8]* @.str, i64 0, i64 0)) #7
  %33 = icmp ne i32 %32, 0
  br i1 %33, label %38, label %34

34:                                               ; preds = %28
  %35 = load i32, i32* %6, align 4, !tbaa !5
  %36 = load [1024 x [1024 x double]]*, [1024 x [1024 x double]]** %7, align 8, !tbaa !9
  %37 = getelementptr inbounds [1024 x [1024 x double]], [1024 x [1024 x double]]* %36, i64 0, i64 0
  call void @print_array(i32 noundef %35, [1024 x double]* noundef %37)
  br label %38

38:                                               ; preds = %34, %28, %2
  %39 = load [1024 x [1024 x double]]*, [1024 x [1024 x double]]** %7, align 8, !tbaa !9
  %40 = bitcast [1024 x [1024 x double]]* %39 to i8*
  call void @free(i8* noundef %40) #6
  %41 = load [1024 x double]*, [1024 x double]** %8, align 8, !tbaa !9
  %42 = bitcast [1024 x double]* %41 to i8*
  call void @free(i8* noundef %42) #6
  %43 = bitcast [1024 x double]** %8 to i8*
  call void @llvm.lifetime.end.p0i8(i64 8, i8* %43) #6
  %44 = bitcast [1024 x [1024 x double]]** %7 to i8*
  call void @llvm.lifetime.end.p0i8(i64 8, i8* %44) #6
  %45 = bitcast i32* %6 to i8*
  call void @llvm.lifetime.end.p0i8(i64 4, i8* %45) #6
  ret i32 0
}

; Function Attrs: argmemonly nofree nosync nounwind willreturn
declare void @llvm.lifetime.start.p0i8(i64 immarg, i8* nocapture) #1

declare i8* @polybench_alloc_data(i64 noundef, i32 noundef) #2

; Function Attrs: nounwind uwtable
define internal void @init_array(i32 noundef %0, double* noundef %1, [1024 x double]* noundef %2) #0 {
  %4 = alloca i32, align 4
  %5 = alloca double*, align 8
  %6 = alloca [1024 x double]*, align 8
  %7 = alloca i32, align 4
  %8 = alloca i32, align 4
  store i32 %0, i32* %4, align 4, !tbaa !5
  store double* %1, double** %5, align 8, !tbaa !9
  store [1024 x double]* %2, [1024 x double]** %6, align 8, !tbaa !9
  %9 = bitcast i32* %7 to i8*
  call void @llvm.lifetime.start.p0i8(i64 4, i8* %9) #6
  %10 = bitcast i32* %8 to i8*
  call void @llvm.lifetime.start.p0i8(i64 4, i8* %10) #6
  store i32 0, i32* %7, align 4, !tbaa !5
  br label %11

11:                                               ; preds = %42, %3
  %12 = load i32, i32* %7, align 4, !tbaa !5
  %13 = load i32, i32* %4, align 4, !tbaa !5
  %14 = icmp slt i32 %12, %13
  br i1 %14, label %15, label %45

15:                                               ; preds = %11
  %16 = load i32, i32* %4, align 4, !tbaa !5
  %17 = sitofp i32 %16 to double
  %18 = fdiv double 1.000000e+00, %17
  %19 = load double*, double** %5, align 8, !tbaa !9
  %20 = load i32, i32* %7, align 4, !tbaa !5
  %21 = sext i32 %20 to i64
  %22 = getelementptr inbounds double, double* %19, i64 %21
  store double %18, double* %22, align 8, !tbaa !11
  store i32 0, i32* %8, align 4, !tbaa !5
  br label %23

23:                                               ; preds = %38, %15
  %24 = load i32, i32* %8, align 4, !tbaa !5
  %25 = load i32, i32* %4, align 4, !tbaa !5
  %26 = icmp slt i32 %24, %25
  br i1 %26, label %27, label %41

27:                                               ; preds = %23
  %28 = load i32, i32* %4, align 4, !tbaa !5
  %29 = sitofp i32 %28 to double
  %30 = fdiv double 1.000000e+00, %29
  %31 = load [1024 x double]*, [1024 x double]** %6, align 8, !tbaa !9
  %32 = load i32, i32* %7, align 4, !tbaa !5
  %33 = sext i32 %32 to i64
  %34 = getelementptr inbounds [1024 x double], [1024 x double]* %31, i64 %33
  %35 = load i32, i32* %8, align 4, !tbaa !5
  %36 = sext i32 %35 to i64
  %37 = getelementptr inbounds [1024 x double], [1024 x double]* %34, i64 0, i64 %36
  store double %30, double* %37, align 8, !tbaa !11
  br label %38

38:                                               ; preds = %27
  %39 = load i32, i32* %8, align 4, !tbaa !5
  %40 = add nsw i32 %39, 1
  store i32 %40, i32* %8, align 4, !tbaa !5
  br label %23, !llvm.loop !13

41:                                               ; preds = %23
  br label %42

42:                                               ; preds = %41
  %43 = load i32, i32* %7, align 4, !tbaa !5
  %44 = add nsw i32 %43, 1
  store i32 %44, i32* %7, align 4, !tbaa !5
  br label %11, !llvm.loop !16

45:                                               ; preds = %11
  %46 = bitcast i32* %8 to i8*
  call void @llvm.lifetime.end.p0i8(i64 4, i8* %46) #6
  %47 = bitcast i32* %7 to i8*
  call void @llvm.lifetime.end.p0i8(i64 4, i8* %47) #6
  ret void
}

; Function Attrs: nounwind uwtable
define internal void @kernel_cholesky(i32 noundef %0, double* noundef %1, [1024 x double]* noundef %2) #0 {
  %4 = alloca i32, align 4
  %5 = alloca double*, align 8
  %6 = alloca [1024 x double]*, align 8
  %7 = alloca i32, align 4
  %8 = alloca i32, align 4
  %9 = alloca i32, align 4
  %10 = alloca double, align 8
  store i32 %0, i32* %4, align 4, !tbaa !5
  store double* %1, double** %5, align 8, !tbaa !9
  store [1024 x double]* %2, [1024 x double]** %6, align 8, !tbaa !9
  %11 = bitcast i32* %7 to i8*
  call void @llvm.lifetime.start.p0i8(i64 4, i8* %11) #6
  %12 = bitcast i32* %8 to i8*
  call void @llvm.lifetime.start.p0i8(i64 4, i8* %12) #6
  %13 = bitcast i32* %9 to i8*
  call void @llvm.lifetime.start.p0i8(i64 4, i8* %13) #6
  %14 = bitcast double* %10 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* %14) #6
  store i32 0, i32* %7, align 4, !tbaa !5
  br label %15

15:                                               ; preds = %126, %3
  %16 = load i32, i32* %7, align 4, !tbaa !5
  %17 = load i32, i32* %4, align 4, !tbaa !5
  %18 = icmp slt i32 %16, %17
  br i1 %18, label %19, label %129

19:                                               ; preds = %15
  %20 = load [1024 x double]*, [1024 x double]** %6, align 8, !tbaa !9
  %21 = load i32, i32* %7, align 4, !tbaa !5
  %22 = sext i32 %21 to i64
  %23 = getelementptr inbounds [1024 x double], [1024 x double]* %20, i64 %22
  %24 = load i32, i32* %7, align 4, !tbaa !5
  %25 = sext i32 %24 to i64
  %26 = getelementptr inbounds [1024 x double], [1024 x double]* %23, i64 0, i64 %25
  %27 = load double, double* %26, align 8, !tbaa !11
  store double %27, double* %10, align 8, !tbaa !11
  store i32 0, i32* %8, align 4, !tbaa !5
  br label %28

28:                                               ; preds = %53, %19
  %29 = load i32, i32* %8, align 4, !tbaa !5
  %30 = load i32, i32* %7, align 4, !tbaa !5
  %31 = sub nsw i32 %30, 1
  %32 = icmp sle i32 %29, %31
  br i1 %32, label %33, label %56

33:                                               ; preds = %28
  %34 = load double, double* %10, align 8, !tbaa !11
  %35 = load [1024 x double]*, [1024 x double]** %6, align 8, !tbaa !9
  %36 = load i32, i32* %7, align 4, !tbaa !5
  %37 = sext i32 %36 to i64
  %38 = getelementptr inbounds [1024 x double], [1024 x double]* %35, i64 %37
  %39 = load i32, i32* %8, align 4, !tbaa !5
  %40 = sext i32 %39 to i64
  %41 = getelementptr inbounds [1024 x double], [1024 x double]* %38, i64 0, i64 %40
  %42 = load double, double* %41, align 8, !tbaa !11
  %43 = load [1024 x double]*, [1024 x double]** %6, align 8, !tbaa !9
  %44 = load i32, i32* %7, align 4, !tbaa !5
  %45 = sext i32 %44 to i64
  %46 = getelementptr inbounds [1024 x double], [1024 x double]* %43, i64 %45
  %47 = load i32, i32* %8, align 4, !tbaa !5
  %48 = sext i32 %47 to i64
  %49 = getelementptr inbounds [1024 x double], [1024 x double]* %46, i64 0, i64 %48
  %50 = load double, double* %49, align 8, !tbaa !11
  %51 = fneg double %42
  %52 = call double @llvm.fmuladd.f64(double %51, double %50, double %34)
  store double %52, double* %10, align 8, !tbaa !11
  br label %53

53:                                               ; preds = %33
  %54 = load i32, i32* %8, align 4, !tbaa !5
  %55 = add nsw i32 %54, 1
  store i32 %55, i32* %8, align 4, !tbaa !5
  br label %28, !llvm.loop !17

56:                                               ; preds = %28
  %57 = load double, double* %10, align 8, !tbaa !11
  %58 = call double @sqrt(double noundef %57) #6
  %59 = fdiv double 1.000000e+00, %58
  %60 = load double*, double** %5, align 8, !tbaa !9
  %61 = load i32, i32* %7, align 4, !tbaa !5
  %62 = sext i32 %61 to i64
  %63 = getelementptr inbounds double, double* %60, i64 %62
  store double %59, double* %63, align 8, !tbaa !11
  %64 = load i32, i32* %7, align 4, !tbaa !5
  %65 = add nsw i32 %64, 1
  store i32 %65, i32* %8, align 4, !tbaa !5
  br label %66

66:                                               ; preds = %122, %56
  %67 = load i32, i32* %8, align 4, !tbaa !5
  %68 = load i32, i32* %4, align 4, !tbaa !5
  %69 = icmp slt i32 %67, %68
  br i1 %69, label %70, label %125

70:                                               ; preds = %66
  %71 = load [1024 x double]*, [1024 x double]** %6, align 8, !tbaa !9
  %72 = load i32, i32* %7, align 4, !tbaa !5
  %73 = sext i32 %72 to i64
  %74 = getelementptr inbounds [1024 x double], [1024 x double]* %71, i64 %73
  %75 = load i32, i32* %8, align 4, !tbaa !5
  %76 = sext i32 %75 to i64
  %77 = getelementptr inbounds [1024 x double], [1024 x double]* %74, i64 0, i64 %76
  %78 = load double, double* %77, align 8, !tbaa !11
  store double %78, double* %10, align 8, !tbaa !11
  store i32 0, i32* %9, align 4, !tbaa !5
  br label %79

79:                                               ; preds = %104, %70
  %80 = load i32, i32* %9, align 4, !tbaa !5
  %81 = load i32, i32* %7, align 4, !tbaa !5
  %82 = sub nsw i32 %81, 1
  %83 = icmp sle i32 %80, %82
  br i1 %83, label %84, label %107

84:                                               ; preds = %79
  %85 = load double, double* %10, align 8, !tbaa !11
  %86 = load [1024 x double]*, [1024 x double]** %6, align 8, !tbaa !9
  %87 = load i32, i32* %8, align 4, !tbaa !5
  %88 = sext i32 %87 to i64
  %89 = getelementptr inbounds [1024 x double], [1024 x double]* %86, i64 %88
  %90 = load i32, i32* %9, align 4, !tbaa !5
  %91 = sext i32 %90 to i64
  %92 = getelementptr inbounds [1024 x double], [1024 x double]* %89, i64 0, i64 %91
  %93 = load double, double* %92, align 8, !tbaa !11
  %94 = load [1024 x double]*, [1024 x double]** %6, align 8, !tbaa !9
  %95 = load i32, i32* %7, align 4, !tbaa !5
  %96 = sext i32 %95 to i64
  %97 = getelementptr inbounds [1024 x double], [1024 x double]* %94, i64 %96
  %98 = load i32, i32* %9, align 4, !tbaa !5
  %99 = sext i32 %98 to i64
  %100 = getelementptr inbounds [1024 x double], [1024 x double]* %97, i64 0, i64 %99
  %101 = load double, double* %100, align 8, !tbaa !11
  %102 = fneg double %93
  %103 = call double @llvm.fmuladd.f64(double %102, double %101, double %85)
  store double %103, double* %10, align 8, !tbaa !11
  br label %104

104:                                              ; preds = %84
  %105 = load i32, i32* %9, align 4, !tbaa !5
  %106 = add nsw i32 %105, 1
  store i32 %106, i32* %9, align 4, !tbaa !5
  br label %79, !llvm.loop !18

107:                                              ; preds = %79
  %108 = load double, double* %10, align 8, !tbaa !11
  %109 = load double*, double** %5, align 8, !tbaa !9
  %110 = load i32, i32* %7, align 4, !tbaa !5
  %111 = sext i32 %110 to i64
  %112 = getelementptr inbounds double, double* %109, i64 %111
  %113 = load double, double* %112, align 8, !tbaa !11
  %114 = fmul double %108, %113
  %115 = load [1024 x double]*, [1024 x double]** %6, align 8, !tbaa !9
  %116 = load i32, i32* %8, align 4, !tbaa !5
  %117 = sext i32 %116 to i64
  %118 = getelementptr inbounds [1024 x double], [1024 x double]* %115, i64 %117
  %119 = load i32, i32* %7, align 4, !tbaa !5
  %120 = sext i32 %119 to i64
  %121 = getelementptr inbounds [1024 x double], [1024 x double]* %118, i64 0, i64 %120
  store double %114, double* %121, align 8, !tbaa !11
  br label %122

122:                                              ; preds = %107
  %123 = load i32, i32* %8, align 4, !tbaa !5
  %124 = add nsw i32 %123, 1
  store i32 %124, i32* %8, align 4, !tbaa !5
  br label %66, !llvm.loop !19

125:                                              ; preds = %66
  br label %126

126:                                              ; preds = %125
  %127 = load i32, i32* %7, align 4, !tbaa !5
  %128 = add nsw i32 %127, 1
  store i32 %128, i32* %7, align 4, !tbaa !5
  br label %15, !llvm.loop !20

129:                                              ; preds = %15
  %130 = bitcast double* %10 to i8*
  call void @llvm.lifetime.end.p0i8(i64 8, i8* %130) #6
  %131 = bitcast i32* %9 to i8*
  call void @llvm.lifetime.end.p0i8(i64 4, i8* %131) #6
  %132 = bitcast i32* %8 to i8*
  call void @llvm.lifetime.end.p0i8(i64 4, i8* %132) #6
  %133 = bitcast i32* %7 to i8*
  call void @llvm.lifetime.end.p0i8(i64 4, i8* %133) #6
  ret void
}

; Function Attrs: nounwind readonly willreturn
declare i32 @strcmp(i8* noundef, i8* noundef) #3

; Function Attrs: nounwind uwtable
define internal void @print_array(i32 noundef %0, [1024 x double]* noundef %1) #0 {
  %3 = alloca i32, align 4
  %4 = alloca [1024 x double]*, align 8
  %5 = alloca i32, align 4
  %6 = alloca i32, align 4
  store i32 %0, i32* %3, align 4, !tbaa !5
  store [1024 x double]* %1, [1024 x double]** %4, align 8, !tbaa !9
  %7 = bitcast i32* %5 to i8*
  call void @llvm.lifetime.start.p0i8(i64 4, i8* %7) #6
  %8 = bitcast i32* %6 to i8*
  call void @llvm.lifetime.start.p0i8(i64 4, i8* %8) #6
  store i32 0, i32* %5, align 4, !tbaa !5
  br label %9

9:                                                ; preds = %43, %2
  %10 = load i32, i32* %5, align 4, !tbaa !5
  %11 = load i32, i32* %3, align 4, !tbaa !5
  %12 = icmp slt i32 %10, %11
  br i1 %12, label %13, label %46

13:                                               ; preds = %9
  store i32 0, i32* %6, align 4, !tbaa !5
  br label %14

14:                                               ; preds = %39, %13
  %15 = load i32, i32* %6, align 4, !tbaa !5
  %16 = load i32, i32* %3, align 4, !tbaa !5
  %17 = icmp slt i32 %15, %16
  br i1 %17, label %18, label %42

18:                                               ; preds = %14
  %19 = load %struct._IO_FILE*, %struct._IO_FILE** @stderr, align 8, !tbaa !9
  %20 = load [1024 x double]*, [1024 x double]** %4, align 8, !tbaa !9
  %21 = load i32, i32* %5, align 4, !tbaa !5
  %22 = sext i32 %21 to i64
  %23 = getelementptr inbounds [1024 x double], [1024 x double]* %20, i64 %22
  %24 = load i32, i32* %6, align 4, !tbaa !5
  %25 = sext i32 %24 to i64
  %26 = getelementptr inbounds [1024 x double], [1024 x double]* %23, i64 0, i64 %25
  %27 = load double, double* %26, align 8, !tbaa !11
  %28 = call i32 (%struct._IO_FILE*, i8*, ...) @fprintf(%struct._IO_FILE* noundef %19, i8* noundef getelementptr inbounds ([8 x i8], [8 x i8]* @.str.1, i64 0, i64 0), double noundef %27)
  %29 = load i32, i32* %5, align 4, !tbaa !5
  %30 = mul nsw i32 %29, 1024
  %31 = load i32, i32* %6, align 4, !tbaa !5
  %32 = add nsw i32 %30, %31
  %33 = srem i32 %32, 20
  %34 = icmp eq i32 %33, 0
  br i1 %34, label %35, label %38

35:                                               ; preds = %18
  %36 = load %struct._IO_FILE*, %struct._IO_FILE** @stderr, align 8, !tbaa !9
  %37 = call i32 (%struct._IO_FILE*, i8*, ...) @fprintf(%struct._IO_FILE* noundef %36, i8* noundef getelementptr inbounds ([2 x i8], [2 x i8]* @.str.2, i64 0, i64 0))
  br label %38

38:                                               ; preds = %35, %18
  br label %39

39:                                               ; preds = %38
  %40 = load i32, i32* %6, align 4, !tbaa !5
  %41 = add nsw i32 %40, 1
  store i32 %41, i32* %6, align 4, !tbaa !5
  br label %14, !llvm.loop !21

42:                                               ; preds = %14
  br label %43

43:                                               ; preds = %42
  %44 = load i32, i32* %5, align 4, !tbaa !5
  %45 = add nsw i32 %44, 1
  store i32 %45, i32* %5, align 4, !tbaa !5
  br label %9, !llvm.loop !22

46:                                               ; preds = %9
  %47 = bitcast i32* %6 to i8*
  call void @llvm.lifetime.end.p0i8(i64 4, i8* %47) #6
  %48 = bitcast i32* %5 to i8*
  call void @llvm.lifetime.end.p0i8(i64 4, i8* %48) #6
  ret void
}

; Function Attrs: nounwind
declare void @free(i8* noundef) #4

; Function Attrs: argmemonly nofree nosync nounwind willreturn
declare void @llvm.lifetime.end.p0i8(i64 immarg, i8* nocapture) #1

; Function Attrs: nofree nosync nounwind readnone speculatable willreturn
declare double @llvm.fmuladd.f64(double, double, double) #5

; Function Attrs: nounwind
declare double @sqrt(double noundef) #4

declare i32 @fprintf(%struct._IO_FILE* noundef, i8* noundef, ...) #2

attributes #0 = { nounwind uwtable "frame-pointer"="none" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #1 = { argmemonly nofree nosync nounwind willreturn }
attributes #2 = { "frame-pointer"="none" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #3 = { nounwind readonly willreturn "frame-pointer"="none" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #4 = { nounwind "frame-pointer"="none" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #5 = { nofree nosync nounwind readnone speculatable willreturn }
attributes #6 = { nounwind }
attributes #7 = { nounwind readonly willreturn }

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
!12 = !{!"double", !7, i64 0}
!13 = distinct !{!13, !14, !15}
!14 = !{!"llvm.loop.mustprogress"}
!15 = !{!"llvm.loop.unroll.disable"}
!16 = distinct !{!16, !14, !15}
!17 = distinct !{!17, !14, !15}
!18 = distinct !{!18, !14, !15}
!19 = distinct !{!19, !14, !15}
!20 = distinct !{!20, !14, !15}
!21 = distinct !{!21, !14, !15}
!22 = distinct !{!22, !14, !15}
