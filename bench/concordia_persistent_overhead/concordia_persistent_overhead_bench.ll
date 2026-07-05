; ModuleID = 'builtin.module'
source_filename = "concordia_persistent_overhead_bench"
target datalayout = "e-i64:64-i128:128-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

define ptx_kernel void @copy_kernel(ptr %v0, i64 %v1, ptr %v2, i64 %v3) #0 {
entry:
  %v4 = insertvalue { ptr, i64 } undef, ptr %v0, 0
  %v5 = insertvalue { ptr, i64 } %v4, i64 %v1, 1
  %v6 = insertvalue { ptr, i64 } undef, ptr %v2, 0
  %v7 = insertvalue { ptr, i64 } %v6, i64 %v3, 1
  br label %bb0
bb0:
  %v8 = phi { ptr, i64 } [ %v5, %entry ]
  %v9 = phi { ptr, i64 } [ %v7, %entry ]
  %v10 = alloca {  }, align 1
  %v11 = bitcast ptr %v10 to ptr
  %v12 = call i64 @cuda_device____internal__index_1d(ptr %v11) #0
  br label %bb1
bb1:
  %v13 = extractvalue { ptr, i64 } %v8, 1
  %v14 = icmp ult i64 %v12, %v13
  %v15 = xor i1 %v14, 1
  br i1 %v15, label %bb5, label %bb2
bb2:
  %v16 = extractvalue { ptr, i64 } %v9, 1
  %v17 = icmp ult i64 %v12, %v16
  %v18 = xor i1 %v17, 1
  br i1 %v18, label %bb7, label %bb6
bb3:
  %v19 = extractvalue { i8, ptr } %v29, 1
  %v20 = extractvalue { ptr, i64 } %v8, 0
  %v21 = getelementptr inbounds float, ptr %v20, i64 %v12
  %v22 = load float, ptr %v21, align 4
  %v23 = fadd contract float %v22, 1.0
  store float %v23, ptr %v19, align 4
  br label %bb5
bb4:
  br label %bb5
bb5:
  ret void
bb6:
  %v24 = extractvalue { ptr, i64 } %v9, 0
  %v25 = getelementptr inbounds float, ptr %v24, i64 %v12
  %v26 = insertvalue { i8, ptr } undef, i8 1, 0
  %v27 = insertvalue { i8, ptr } %v26, ptr %v25, 1
  br label %bb8
bb7:
  %v28 = insertvalue { i8, ptr } undef, i8 0, 0
  br label %bb8
bb8:
  %v29 = phi { i8, ptr } [ %v27, %bb6 ], [ %v28, %bb7 ]
  %v30 = extractvalue { i8, ptr } %v29, 0
  %v31 = zext i8 %v30 to i64
  %v32 = icmp eq i64 %v31, 1
  br i1 %v32, label %bb3, label %bb9
bb9:
  %v33 = icmp eq i64 %v31, 0
  br i1 %v33, label %bb4, label %bb10
bb10:
  unreachable
}

define ptx_kernel void @compute_kernel(ptr %v0, i64 %v1, ptr %v2, i64 %v3, i32 %v4) #0 {
entry:
  %v5 = insertvalue { ptr, i64 } undef, ptr %v0, 0
  %v6 = insertvalue { ptr, i64 } %v5, i64 %v1, 1
  %v7 = insertvalue { ptr, i64 } undef, ptr %v2, 0
  %v8 = insertvalue { ptr, i64 } %v7, i64 %v3, 1
  br label %bb0
bb0:
  %v9 = phi { ptr, i64 } [ %v6, %entry ]
  %v10 = phi { ptr, i64 } [ %v8, %entry ]
  %v11 = phi i32 [ %v4, %entry ]
  %v12 = alloca {  }, align 1
  %v13 = bitcast ptr %v12 to ptr
  %v14 = call i64 @cuda_device____internal__index_1d(ptr %v13) #0
  br label %bb1
bb1:
  %v15 = extractvalue { ptr, i64 } %v9, 1
  %v16 = icmp ult i64 %v14, %v15
  %v17 = xor i1 %v16, 1
  br i1 %v17, label %bb9, label %bb2
bb2:
  %v18 = extractvalue { ptr, i64 } %v9, 0
  %v19 = getelementptr inbounds float, ptr %v18, i64 %v14
  %v20 = load float, ptr %v19, align 4
  br label %bb3
bb3:
  %v21 = phi float [ %v20, %bb2 ], [ %v26, %bb4 ]
  %v22 = phi i32 [ 0, %bb2 ], [ %v27, %bb4 ]
  %v23 = icmp ult i32 %v22, %v11
  %v24 = xor i1 %v23, 1
  br i1 %v24, label %bb5, label %bb4
bb4:
  %v25 = fmul contract float %v21, 1.0000001192092896
  %v26 = fadd contract float %v25, 0.00000011899999918796311
  %v27 = add i32 %v22, 1
  br label %bb3
bb5:
  %v28 = extractvalue { ptr, i64 } %v10, 1
  %v29 = icmp ult i64 %v14, %v28
  %v30 = xor i1 %v29, 1
  br i1 %v30, label %bb11, label %bb10
bb6:
  %v31 = extractvalue { i8, ptr } %v37, 1
  store float %v21, ptr %v31, align 4
  br label %bb8
bb7:
  br label %bb8
bb8:
  br label %bb9
bb9:
  ret void
bb10:
  %v32 = extractvalue { ptr, i64 } %v10, 0
  %v33 = getelementptr inbounds float, ptr %v32, i64 %v14
  %v34 = insertvalue { i8, ptr } undef, i8 1, 0
  %v35 = insertvalue { i8, ptr } %v34, ptr %v33, 1
  br label %bb12
bb11:
  %v36 = insertvalue { i8, ptr } undef, i8 0, 0
  br label %bb12
bb12:
  %v37 = phi { i8, ptr } [ %v35, %bb10 ], [ %v36, %bb11 ]
  %v38 = extractvalue { i8, ptr } %v37, 0
  %v39 = zext i8 %v38 to i64
  %v40 = icmp eq i64 %v39, 1
  br i1 %v40, label %bb6, label %bb13
bb13:
  %v41 = icmp eq i64 %v39, 0
  br i1 %v41, label %bb7, label %bb14
bb14:
  unreachable
}

define ptx_kernel void @persistent_worker(ptr %v0, i64 %v1, i64 %v2, ptr %v3, i64 %v4) #0 {
entry:
  %v5 = insertvalue { ptr, i64 } undef, ptr %v0, 0
  %v6 = insertvalue { ptr, i64 } %v5, i64 %v1, 1
  %v7 = insertvalue { ptr, i64 } undef, ptr %v3, 0
  %v8 = insertvalue { ptr, i64 } %v7, i64 %v4, 1
  br label %bb0
bb0:
  %v9 = phi { ptr, i64 } [ %v6, %entry ]
  %v10 = phi i64 [ %v2, %entry ]
  %v11 = phi { ptr, i64 } [ %v8, %entry ]
  %v12 = alloca {  }, align 1
  %v13 = bitcast ptr %v12 to ptr
  %v14 = call i64 @cuda_device____internal__index_1d(ptr %v13) #0
  br label %bb1
bb1:
  %v15 = extractvalue { ptr, i64 } %v9, 0
  %v16 = bitcast i64 %v14 to i64
  %v17 = mul i64 %v16, 11400714819323198485
  br label %bb2
bb2:
  %v18 = phi i64 [ 0, %bb1 ], [ %v31, %bb8 ]
  %v19 = phi i64 [ %v17, %bb1 ], [ %v25, %bb8 ]
  %v20 = load atomic i32, ptr %v15 syncscope("device") acquire, align 4
  br label %bb3
bb3:
  %v21 = icmp eq i32 %v20, 0
  br i1 %v21, label %bb4, label %bb10
bb4:
  %v22 = icmp ult i64 %v18, %v10
  %v23 = xor i1 %v22, 1
  br i1 %v23, label %bb9, label %bb5
bb5:
  br label %bb6
bb6:
  %v24 = phi i32 [ 0, %bb5 ], [ %v30, %bb7 ]
  %v25 = phi i64 [ %v19, %bb5 ], [ %v29, %bb7 ]
  %v26 = icmp ult i32 %v24, 512
  %v27 = xor i1 %v26, 1
  br i1 %v27, label %bb8, label %bb7
bb7:
  %v28 = mul i64 %v25, 2862933555777941757
  %v29 = add i64 %v28, 3037000493
  %v30 = add i32 %v24, 1
  br label %bb6
bb8:
  %v31 = add i64 %v18, 1
  br label %bb2
bb9:
  br label %bb11
bb10:
  br label %bb11
bb11:
  %v32 = extractvalue { ptr, i64 } %v11, 1
  %v33 = icmp ult i64 %v14, %v32
  %v34 = xor i1 %v33, 1
  br i1 %v34, label %bb16, label %bb15
bb12:
  %v35 = extractvalue { i8, ptr } %v42, 1
  %v36 = xor i64 %v19, %v18
  store i64 %v36, ptr %v35, align 8
  br label %bb14
bb13:
  br label %bb14
bb14:
  ret void
bb15:
  %v37 = extractvalue { ptr, i64 } %v11, 0
  %v38 = getelementptr inbounds i64, ptr %v37, i64 %v14
  %v39 = insertvalue { i8, ptr } undef, i8 1, 0
  %v40 = insertvalue { i8, ptr } %v39, ptr %v38, 1
  br label %bb17
bb16:
  %v41 = insertvalue { i8, ptr } undef, i8 0, 0
  br label %bb17
bb17:
  %v42 = phi { i8, ptr } [ %v40, %bb15 ], [ %v41, %bb16 ]
  %v43 = extractvalue { i8, ptr } %v42, 0
  %v44 = zext i8 %v43 to i64
  %v45 = icmp eq i64 %v44, 1
  br i1 %v45, label %bb12, label %bb18
bb18:
  %v46 = icmp eq i64 %v44, 0
  br i1 %v46, label %bb13, label %bb19
bb19:
  unreachable
}

declare i32 @llvm.nvvm.read.ptx.sreg.tid.x()
declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
declare i32 @llvm.nvvm.read.ptx.sreg.ntid.x()

define i64 @cuda_device____internal__index_1d(ptr %v0) alwaysinline #0 {
entry:
  br label %bb0
bb0:
  %v1 = phi ptr [ %v0, %entry ]
  %v2 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #0
  br label %bb1
bb1:
  %v3 = zext i32 %v2 to i64
  %v4 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #0
  br label %bb2
bb2:
  %v5 = zext i32 %v4 to i64
  %v6 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #0
  br label %bb3
bb3:
  %v7 = zext i32 %v6 to i64
  %v8 = mul i64 %v5, %v7
  %v9 = add i64 %v8, %v3
  ret i64 %v9
}


attributes #0 = { convergent }
