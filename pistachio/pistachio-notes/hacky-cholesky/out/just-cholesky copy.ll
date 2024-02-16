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
  %16 = load i32, ptr %7, align 4, !tbaa !5	 --- Depends on instruction outside current block
  %17 = load i32, ptr %4, align 4, !tbaa !5	 --- Depends on instruction outside current block
  %18 = icmp slt i32 %16, %17
  br i1 %18, label %19, label %129
  %20 = load ptr, ptr %6, align 8, !tbaa !9	 --- Depends on instruction outside current block
  %21 = load i32, ptr %7, align 4, !tbaa !5	 --- Depends on instruction outside current block
  %22 = sext i32 %21 to i64
  %23 = getelementptr inbounds [1024 x double], ptr %20, i64 %22
  %24 = load i32, ptr %7, align 4, !tbaa !5	 --- DEPENDS ON --->   %21 = load i32, ptr %7, align 4, !tbaa !5
  %25 = sext i32 %24 to i64
  %26 = getelementptr inbounds [1024 x double], ptr %23, i64 0, i64 %25
  %27 = load double, ptr %26, align 8, !tbaa !11	 --- Depends on instruction outside current block
  store double %27, ptr %10, align 8, !tbaa !11
  store i32 0, ptr %8, align 4, !tbaa !5
  br label %28
  %29 = load i32, ptr %8, align 4, !tbaa !5	 --- Depends on instruction outside current block
  %30 = load i32, ptr %7, align 4, !tbaa !5	 --- Depends on instruction outside current block
  %31 = sub nsw i32 %30, 1
  %32 = icmp sle i32 %29, %31
  br i1 %32, label %33, label %56
  %34 = load double, ptr %10, align 8, !tbaa !11	 --- Depends on instruction outside current block
  %35 = load ptr, ptr %6, align 8, !tbaa !9	 --- Depends on instruction outside current block
  %36 = load i32, ptr %7, align 4, !tbaa !5	 --- Depends on instruction outside current block
  %37 = sext i32 %36 to i64
  %38 = getelementptr inbounds [1024 x double], ptr %35, i64 %37
  %39 = load i32, ptr %8, align 4, !tbaa !5	 --- Depends on instruction outside current block
  %40 = sext i32 %39 to i64
  %41 = getelementptr inbounds [1024 x double], ptr %38, i64 0, i64 %40
  %42 = load double, ptr %41, align 8, !tbaa !11	 --- Depends on instruction outside current block
  %43 = load ptr, ptr %6, align 8, !tbaa !9	 --- DEPENDS ON --->   %35 = load ptr, ptr %6, align 8, !tbaa !9
  %44 = load i32, ptr %7, align 4, !tbaa !5	 --- DEPENDS ON --->   %36 = load i32, ptr %7, align 4, !tbaa !5
  %45 = sext i32 %44 to i64
  %46 = getelementptr inbounds [1024 x double], ptr %43, i64 %45
  %47 = load i32, ptr %8, align 4, !tbaa !5	 --- DEPENDS ON --->   %39 = load i32, ptr %8, align 4, !tbaa !5
  %48 = sext i32 %47 to i64
  %49 = getelementptr inbounds [1024 x double], ptr %46, i64 0, i64 %48
  %50 = load double, ptr %49, align 8, !tbaa !11	 --- Depends on instruction outside current block
  %51 = fneg double %42
  %52 = call double @llvm.fmuladd.f64(double %51, double %50, double %34)
  store double %52, ptr %10, align 8, !tbaa !11
  br label %53
  %54 = load i32, ptr %8, align 4, !tbaa !5	 --- Depends on instruction outside current block
  %55 = add nsw i32 %54, 1
  store i32 %55, ptr %8, align 4, !tbaa !5
  br label %28, !llvm.loop !13
  %57 = load double, ptr %10, align 8, !tbaa !11	 --- Depends on instruction outside current block
  %58 = call double @sqrt(double noundef %57) #6
  %59 = fdiv double 1.000000e+00, %58
  %60 = load ptr, ptr %5, align 8, !tbaa !9	 --- Depends on instruction outside current block
  %61 = load i32, ptr %7, align 4, !tbaa !5	 --- Depends on instruction outside current block
  %62 = sext i32 %61 to i64
  %63 = getelementptr inbounds double, ptr %60, i64 %62
  store double %59, ptr %63, align 8, !tbaa !11
  %64 = load i32, ptr %7, align 4, !tbaa !5	 --- DEPENDS ON --->   %61 = load i32, ptr %7, align 4, !tbaa !5
  %65 = add nsw i32 %64, 1
  store i32 %65, ptr %8, align 4, !tbaa !5
  br label %66
  %67 = load i32, ptr %8, align 4, !tbaa !5	 --- Depends on instruction outside current block
  %68 = load i32, ptr %4, align 4, !tbaa !5	 --- Depends on instruction outside current block
  %69 = icmp slt i32 %67, %68
  br i1 %69, label %70, label %125
  %71 = load ptr, ptr %6, align 8, !tbaa !9	 --- Depends on instruction outside current block
  %72 = load i32, ptr %7, align 4, !tbaa !5	 --- Depends on instruction outside current block
  %73 = sext i32 %72 to i64
  %74 = getelementptr inbounds [1024 x double], ptr %71, i64 %73
  %75 = load i32, ptr %8, align 4, !tbaa !5	 --- Depends on instruction outside current block
  %76 = sext i32 %75 to i64
  %77 = getelementptr inbounds [1024 x double], ptr %74, i64 0, i64 %76
  %78 = load double, ptr %77, align 8, !tbaa !11	 --- Depends on instruction outside current block
  store double %78, ptr %10, align 8, !tbaa !11
  store i32 0, ptr %9, align 4, !tbaa !5
  br label %79
  %80 = load i32, ptr %9, align 4, !tbaa !5	 --- Depends on instruction outside current block
  %81 = load i32, ptr %7, align 4, !tbaa !5	 --- Depends on instruction outside current block
  %82 = sub nsw i32 %81, 1
  %83 = icmp sle i32 %80, %82
  br i1 %83, label %84, label %107
  %85 = load double, ptr %10, align 8, !tbaa !11	 --- Depends on instruction outside current block
  %86 = load ptr, ptr %6, align 8, !tbaa !9	 --- Depends on instruction outside current block
  %87 = load i32, ptr %8, align 4, !tbaa !5	 --- Depends on instruction outside current block
  %88 = sext i32 %87 to i64
  %89 = getelementptr inbounds [1024 x double], ptr %86, i64 %88
  %90 = load i32, ptr %9, align 4, !tbaa !5	 --- Depends on instruction outside current block
  %91 = sext i32 %90 to i64
  %92 = getelementptr inbounds [1024 x double], ptr %89, i64 0, i64 %91
  %93 = load double, ptr %92, align 8, !tbaa !11	 --- Depends on instruction outside current block
  %94 = load ptr, ptr %6, align 8, !tbaa !9	 --- DEPENDS ON --->   %86 = load ptr, ptr %6, align 8, !tbaa !9
  %95 = load i32, ptr %7, align 4, !tbaa !5	 --- Depends on instruction outside current block
  %96 = sext i32 %95 to i64
  %97 = getelementptr inbounds [1024 x double], ptr %94, i64 %96
  %98 = load i32, ptr %9, align 4, !tbaa !5	 --- DEPENDS ON --->   %90 = load i32, ptr %9, align 4, !tbaa !5
  %99 = sext i32 %98 to i64
  %100 = getelementptr inbounds [1024 x double], ptr %97, i64 0, i64 %99
  %101 = load double, ptr %100, align 8, !tbaa !11	 --- Depends on instruction outside current block
  %102 = fneg double %93
  %103 = call double @llvm.fmuladd.f64(double %102, double %101, double %85)
  store double %103, ptr %10, align 8, !tbaa !11
  br label %104
  %105 = load i32, ptr %9, align 4, !tbaa !5	 --- Depends on instruction outside current block
  %106 = add nsw i32 %105, 1
  store i32 %106, ptr %9, align 4, !tbaa !5
  br label %79, !llvm.loop !16
  %108 = load double, ptr %10, align 8, !tbaa !11	 --- Depends on instruction outside current block
  %109 = load ptr, ptr %5, align 8, !tbaa !9	 --- Depends on instruction outside current block
  %110 = load i32, ptr %7, align 4, !tbaa !5	 --- Depends on instruction outside current block
  %111 = sext i32 %110 to i64
  %112 = getelementptr inbounds double, ptr %109, i64 %111
  %113 = load double, ptr %112, align 8, !tbaa !11	 --- Depends on instruction outside current block
  %114 = fmul double %108, %113
  %115 = load ptr, ptr %6, align 8, !tbaa !9	 --- Depends on instruction outside current block
  %116 = load i32, ptr %8, align 4, !tbaa !5	 --- Depends on instruction outside current block
  %117 = sext i32 %116 to i64
  %118 = getelementptr inbounds [1024 x double], ptr %115, i64 %117
  %119 = load i32, ptr %7, align 4, !tbaa !5	 --- DEPENDS ON --->   %110 = load i32, ptr %7, align 4, !tbaa !5
  %120 = sext i32 %119 to i64
  %121 = getelementptr inbounds [1024 x double], ptr %118, i64 0, i64 %120
  store double %114, ptr %121, align 8, !tbaa !11
  br label %122
  %123 = load i32, ptr %8, align 4, !tbaa !5	 --- Depends on instruction outside current block
  %124 = add nsw i32 %123, 1
  store i32 %124, ptr %8, align 4, !tbaa !5
  br label %66, !llvm.loop !17
  br label %126
  %127 = load i32, ptr %7, align 4, !tbaa !5	 --- Depends on instruction outside current block
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
Function kernel_cholesky:

  store double %52, ptr %10, align 8, !tbaa !11 DOES NOT ALIAS WITH ANY LOADS 

  store i32 %55, ptr %8, align 4, !tbaa !5 DOES NOT ALIAS WITH ANY LOADS 

  store double %114, ptr %121, align 8, !tbaa !11 DOES NOT ALIAS WITH ANY LOADS 

  store i32 %124, ptr %8, align 4, !tbaa !5 DOES NOT ALIAS WITH ANY LOADS 

  store i32 %0, ptr %4, align 4, !tbaa !5 DOES NOT ALIAS WITH ANY LOADS 

  store ptr %1, ptr %5, align 8, !tbaa !9 DOES NOT ALIAS WITH ANY LOADS 

  store ptr %2, ptr %6, align 8, !tbaa !9 DOES NOT ALIAS WITH ANY LOADS 

  store i32 0, ptr %7, align 4, !tbaa !5 DOES NOT ALIAS WITH ANY LOADS 

  store double %27, ptr %10, align 8, !tbaa !11 DOES NOT ALIAS WITH ANY LOADS 

  store i32 0, ptr %8, align 4, !tbaa !5 DOES NOT ALIAS WITH ANY LOADS 

  store double %78, ptr %10, align 8, !tbaa !11 DOES NOT ALIAS WITH ANY LOADS 

  store i32 0, ptr %9, align 4, !tbaa !5 DOES NOT ALIAS WITH ANY LOADS 

  store double %59, ptr %63, align 8, !tbaa !11 DOES NOT ALIAS WITH ANY LOADS 

  store i32 %65, ptr %8, align 4, !tbaa !5 DOES NOT ALIAS WITH ANY LOADS 

  store double %103, ptr %10, align 8, !tbaa !11 DOES NOT ALIAS WITH ANY LOADS 

  store i32 %106, ptr %9, align 4, !tbaa !5 DOES NOT ALIAS WITH ANY LOADS 

  store i32 %128, ptr %7, align 4, !tbaa !5 DOES NOT ALIAS WITH ANY LOADS 