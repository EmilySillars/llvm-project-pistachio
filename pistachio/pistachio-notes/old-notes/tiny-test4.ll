; ModuleID = '<stdin>'
source_filename = "emily-notes/hello2.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-linux-gnu"

@.str = private unnamed_addr constant [13 x i8] c"hello world\0A\00", align 1

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @foo() #0 {
  %1 = alloca i32, align 4
  store i32 5, i32* %1, align 4
  %2 = load i32, i32* %1, align 4
  ret i32 %2
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local void @bar() #0 {
  ret void
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @main() #0 {
  %1 = alloca i32, align 4
  store i32 0, i32* %1, align 4
  %2 = call i32 (i8*, ...) @printf(i8* noundef getelementptr inbounds ([13 x i8], [13 x i8]* @.str, i64 0, i64 0))
  %3 = call i32 @foo()
  ret i32 %3
}

declare i32 @printf(i8* noundef, ...) #1

