Hello, Function kernel_cholesky:
  %4 = alloca i32, align 4
  %5 = alloca ptr, align 8
  %6 = alloca ptr, align 8
  %7 = alloca i32, align 4
  %8 = alloca i32, align 4
  %9 = alloca i32, align 4
  %10 = alloca double, align 8
  store i32 %0, ptr %4, align 4, !tbaa !5
  store ptr %1, ptr %5, align 8, !tbaa !9
  store ptr %2, ptr %6, align 8, !tbaa !9
  %11 = bitcast ptr %7 to ptr
  call void @llvm.lifetime.start.p0(i64 4, ptr %11) #6
  %12 = bitcast ptr %8 to ptr
  call void @llvm.lifetime.start.p0(i64 4, ptr %12) #6
  %13 = bitcast ptr %9 to ptr
  call void @llvm.lifetime.start.p0(i64 4, ptr %13) #6
  %14 = bitcast ptr %10 to ptr
  call void @llvm.lifetime.start.p0(i64 8, ptr %14) #6
  store i32 0, ptr %7, align 4, !tbaa !5
  br label %15
  %16 = load i32, ptr %7, align 4, !tbaa !5	 --- Doesn't Depend on Anything?
  %17 = load i32, ptr %4, align 4, !tbaa !5	 --- Doesn't Depend on Anything?
  %18 = icmp slt i32 %16, %17
  br i1 %18, label %19, label %129
  %20 = load ptr, ptr %6, align 8, !tbaa !9	 --- Doesn't Depend on Anything?
  %21 = load i32, ptr %7, align 4, !tbaa !5	 --- Doesn't Depend on Anything?
  %22 = sext i32 %21 to i64
  %23 = getelementptr inbounds [1024 x double], ptr %20, i64 %22
  %24 = load i32, ptr %7, align 4, !tbaa !5	 --- DEPENDS ON --->   %21 = load i32, ptr %7, align 4, !tbaa !5
  %25 = sext i32 %24 to i64
  %26 = getelementptr inbounds [1024 x double], ptr %23, i64 0, i64 %25
  %27 = load double, ptr %26, align 8, !tbaa !11	 --- Doesn't Depend on Anything?
  store double %27, ptr %10, align 8, !tbaa !11
  store i32 0, ptr %8, align 4, !tbaa !5
  br label %28
  %29 = load i32, ptr %8, align 4, !tbaa !5	 --- Doesn't Depend on Anything?
  %30 = load i32, ptr %7, align 4, !tbaa !5	 --- Doesn't Depend on Anything?
  %31 = sub nsw i32 %30, 1
  %32 = icmp sle i32 %29, %31
  br i1 %32, label %33, label %56
  %34 = load double, ptr %10, align 8, !tbaa !11	 --- Doesn't Depend on Anything?
  %35 = load ptr, ptr %6, align 8, !tbaa !9	 --- Doesn't Depend on Anything?
  %36 = load i32, ptr %7, align 4, !tbaa !5	 --- Doesn't Depend on Anything?
  %37 = sext i32 %36 to i64
  %38 = getelementptr inbounds [1024 x double], ptr %35, i64 %37
  %39 = load i32, ptr %8, align 4, !tbaa !5	 --- Doesn't Depend on Anything?
  %40 = sext i32 %39 to i64
  %41 = getelementptr inbounds [1024 x double], ptr %38, i64 0, i64 %40
  %42 = load double, ptr %41, align 8, !tbaa !11	 --- Doesn't Depend on Anything?
  %43 = load ptr, ptr %6, align 8, !tbaa !9	 --- DEPENDS ON --->   %35 = load ptr, ptr %6, align 8, !tbaa !9
  %44 = load i32, ptr %7, align 4, !tbaa !5	 --- DEPENDS ON --->   %36 = load i32, ptr %7, align 4, !tbaa !5
  %45 = sext i32 %44 to i64
  %46 = getelementptr inbounds [1024 x double], ptr %43, i64 %45
  %47 = load i32, ptr %8, align 4, !tbaa !5	 --- DEPENDS ON --->   %39 = load i32, ptr %8, align 4, !tbaa !5
  %48 = sext i32 %47 to i64
  %49 = getelementptr inbounds [1024 x double], ptr %46, i64 0, i64 %48
  %50 = load double, ptr %49, align 8, !tbaa !11	 --- Doesn't Depend on Anything?
  %51 = fneg double %42
  %52 = call double @llvm.fmuladd.f64(double %51, double %50, double %34)
  store double %52, ptr %10, align 8, !tbaa !11
  br label %53
  %54 = load i32, ptr %8, align 4, !tbaa !5	 --- Doesn't Depend on Anything?
  %55 = add nsw i32 %54, 1
  store i32 %55, ptr %8, align 4, !tbaa !5
  br label %28, !llvm.loop !13
  %57 = load double, ptr %10, align 8, !tbaa !11	 --- Doesn't Depend on Anything?
  %58 = call double @sqrt(double noundef %57) #6
  %59 = fdiv double 1.000000e+00, %58
  %60 = load ptr, ptr %5, align 8, !tbaa !9	 --- Doesn't Depend on Anything?
  %61 = load i32, ptr %7, align 4, !tbaa !5	 --- Doesn't Depend on Anything?
  %62 = sext i32 %61 to i64
  %63 = getelementptr inbounds double, ptr %60, i64 %62
  store double %59, ptr %63, align 8, !tbaa !11
  %64 = load i32, ptr %7, align 4, !tbaa !5	 --- DEPENDS ON --->   %61 = load i32, ptr %7, align 4, !tbaa !5
  %65 = add nsw i32 %64, 1
  store i32 %65, ptr %8, align 4, !tbaa !5
  br label %66
  %67 = load i32, ptr %8, align 4, !tbaa !5	 --- Doesn't Depend on Anything?
  %68 = load i32, ptr %4, align 4, !tbaa !5	 --- Doesn't Depend on Anything?
  %69 = icmp slt i32 %67, %68
  br i1 %69, label %70, label %125
  %71 = load ptr, ptr %6, align 8, !tbaa !9	 --- Doesn't Depend on Anything?
  %72 = load i32, ptr %7, align 4, !tbaa !5	 --- Doesn't Depend on Anything?
  %73 = sext i32 %72 to i64
  %74 = getelementptr inbounds [1024 x double], ptr %71, i64 %73
  %75 = load i32, ptr %8, align 4, !tbaa !5	 --- Doesn't Depend on Anything?
  %76 = sext i32 %75 to i64
  %77 = getelementptr inbounds [1024 x double], ptr %74, i64 0, i64 %76
  %78 = load double, ptr %77, align 8, !tbaa !11	 --- Doesn't Depend on Anything?
  store double %78, ptr %10, align 8, !tbaa !11
  store i32 0, ptr %9, align 4, !tbaa !5
  br label %79
  %80 = load i32, ptr %9, align 4, !tbaa !5	 --- Doesn't Depend on Anything?
  %81 = load i32, ptr %7, align 4, !tbaa !5	 --- Doesn't Depend on Anything?
  %82 = sub nsw i32 %81, 1
  %83 = icmp sle i32 %80, %82
  br i1 %83, label %84, label %107
  %85 = load double, ptr %10, align 8, !tbaa !11	 --- Doesn't Depend on Anything?
  %86 = load ptr, ptr %6, align 8, !tbaa !9	 --- Doesn't Depend on Anything?
  %87 = load i32, ptr %8, align 4, !tbaa !5	 --- Doesn't Depend on Anything?
  %88 = sext i32 %87 to i64
  %89 = getelementptr inbounds [1024 x double], ptr %86, i64 %88
  %90 = load i32, ptr %9, align 4, !tbaa !5	 --- Doesn't Depend on Anything?
  %91 = sext i32 %90 to i64
  %92 = getelementptr inbounds [1024 x double], ptr %89, i64 0, i64 %91
  %93 = load double, ptr %92, align 8, !tbaa !11	 --- Doesn't Depend on Anything?
  %94 = load ptr, ptr %6, align 8, !tbaa !9	 --- DEPENDS ON --->   %86 = load ptr, ptr %6, align 8, !tbaa !9
  %95 = load i32, ptr %7, align 4, !tbaa !5	 --- Doesn't Depend on Anything?
  %96 = sext i32 %95 to i64
  %97 = getelementptr inbounds [1024 x double], ptr %94, i64 %96
  %98 = load i32, ptr %9, align 4, !tbaa !5	 --- DEPENDS ON --->   %90 = load i32, ptr %9, align 4, !tbaa !5
  %99 = sext i32 %98 to i64
  %100 = getelementptr inbounds [1024 x double], ptr %97, i64 0, i64 %99
  %101 = load double, ptr %100, align 8, !tbaa !11	 --- Doesn't Depend on Anything?
  %102 = fneg double %93
  %103 = call double @llvm.fmuladd.f64(double %102, double %101, double %85)
  store double %103, ptr %10, align 8, !tbaa !11
  br label %104
  %105 = load i32, ptr %9, align 4, !tbaa !5	 --- Doesn't Depend on Anything?
  %106 = add nsw i32 %105, 1
  store i32 %106, ptr %9, align 4, !tbaa !5
  br label %79, !llvm.loop !16
  %108 = load double, ptr %10, align 8, !tbaa !11	 --- Doesn't Depend on Anything?
  %109 = load ptr, ptr %5, align 8, !tbaa !9	 --- Doesn't Depend on Anything?
  %110 = load i32, ptr %7, align 4, !tbaa !5	 --- Doesn't Depend on Anything?
  %111 = sext i32 %110 to i64
  %112 = getelementptr inbounds double, ptr %109, i64 %111
  %113 = load double, ptr %112, align 8, !tbaa !11	 --- Doesn't Depend on Anything?
  %114 = fmul double %108, %113
  %115 = load ptr, ptr %6, align 8, !tbaa !9	 --- Doesn't Depend on Anything?
  %116 = load i32, ptr %8, align 4, !tbaa !5	 --- Doesn't Depend on Anything?
  %117 = sext i32 %116 to i64
  %118 = getelementptr inbounds [1024 x double], ptr %115, i64 %117
  %119 = load i32, ptr %7, align 4, !tbaa !5	 --- DEPENDS ON --->   %110 = load i32, ptr %7, align 4, !tbaa !5
  %120 = sext i32 %119 to i64
  %121 = getelementptr inbounds [1024 x double], ptr %118, i64 0, i64 %120
  store double %114, ptr %121, align 8, !tbaa !11
  br label %122
  %123 = load i32, ptr %8, align 4, !tbaa !5	 --- Doesn't Depend on Anything?
  %124 = add nsw i32 %123, 1
  store i32 %124, ptr %8, align 4, !tbaa !5
  br label %66, !llvm.loop !17
  br label %126
  %127 = load i32, ptr %7, align 4, !tbaa !5	 --- Doesn't Depend on Anything?
  %128 = add nsw i32 %127, 1
  store i32 %128, ptr %7, align 4, !tbaa !5
  br label %15, !llvm.loop !18
  %130 = bitcast ptr %10 to ptr
  call void @llvm.lifetime.end.p0(i64 8, ptr %130) #6
  %131 = bitcast ptr %9 to ptr
  call void @llvm.lifetime.end.p0(i64 4, ptr %131) #6
  %132 = bitcast ptr %8 to ptr
  call void @llvm.lifetime.end.p0(i64 4, ptr %132) #6
  %133 = bitcast ptr %7 to ptr
  call void @llvm.lifetime.end.p0(i64 4, ptr %133) #6
  ret void