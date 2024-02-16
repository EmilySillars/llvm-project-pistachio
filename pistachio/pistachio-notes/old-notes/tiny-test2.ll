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