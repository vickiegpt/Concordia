; ModuleID = '/home/victoryang00/hetGPU/bench/concordia_persistent_overhead/concordia_persistent_overhead_bench.ll'
source_filename = "concordia_persistent_overhead_bench"
target datalayout = "e-i64:64-i128:128-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

; Function Attrs: convergent mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite)
define ptx_kernel void @copy_kernel(ptr readonly captures(none) %v0, i64 %v1, ptr writeonly captures(none) %v2, i64 %v3) local_unnamed_addr #0 {
entry:
  %v2.i = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #6
  %v3.i = zext nneg i32 %v2.i to i64
  %v4.i = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #6
  %v5.i = zext nneg i32 %v4.i to i64
  %v6.i = tail call i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #6
  %v7.i = zext nneg i32 %v6.i to i64
  %v8.i = mul nuw nsw i64 %v5.i, %v7.i
  %v9.i = add nuw nsw i64 %v8.i, %v3.i
  %v14.not = icmp ult i64 %v9.i, %v1
  %v17.not = icmp ult i64 %v9.i, %v3
  %or.cond = select i1 %v14.not, i1 %v17.not, i1 false
  br i1 %or.cond, label %bb3, label %bb5

bb3:                                              ; preds = %entry
  %v21 = getelementptr inbounds nuw float, ptr %v0, i64 %v9.i
  %v25 = getelementptr inbounds nuw float, ptr %v2, i64 %v9.i
  %v22 = load float, ptr %v21, align 4
  %v23 = fadd contract float %v22, 1.000000e+00
  store float %v23, ptr %v25, align 4
  br label %bb5

bb5:                                              ; preds = %bb3, %entry
  ret void
}

; Function Attrs: convergent nofree norecurse nosync nounwind memory(argmem: readwrite)
define ptx_kernel void @compute_kernel(ptr readonly captures(none) %v0, i64 %v1, ptr writeonly captures(none) %v2, i64 %v3, i32 %v4) local_unnamed_addr #1 {
entry:
  %v2.i = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #6
  %v3.i = zext nneg i32 %v2.i to i64
  %v4.i = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #6
  %v5.i = zext nneg i32 %v4.i to i64
  %v6.i = tail call i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #6
  %v7.i = zext nneg i32 %v6.i to i64
  %v8.i = mul nuw nsw i64 %v5.i, %v7.i
  %v9.i = add nuw nsw i64 %v8.i, %v3.i
  %v16.not = icmp ult i64 %v9.i, %v1
  br i1 %v16.not, label %bb2, label %bb9

bb2:                                              ; preds = %entry
  %v19 = getelementptr inbounds nuw float, ptr %v0, i64 %v9.i
  %v20 = load float, ptr %v19, align 4
  %v23.not1.not = icmp eq i32 %v4, 0
  br i1 %v23.not1.not, label %bb5, label %bb4.preheader

bb4.preheader:                                    ; preds = %bb2
  %xtraiter = and i32 %v4, 7
  %0 = icmp ult i32 %v4, 8
  br i1 %0, label %bb4.epil.preheader, label %bb4.preheader.new

bb4.preheader.new:                                ; preds = %bb4.preheader
  %unroll_iter = and i32 %v4, -8
  br label %bb4

bb4:                                              ; preds = %bb4, %bb4.preheader.new
  %v212 = phi float [ %v20, %bb4.preheader.new ], [ %v26.7, %bb4 ]
  %niter = phi i32 [ 0, %bb4.preheader.new ], [ %niter.next.7, %bb4 ]
  %v25 = fmul contract float %v212, 0x3FF0000020000000
  %v26 = fadd contract float %v25, 0x3E7FF19E20000000
  %v25.1 = fmul contract float %v26, 0x3FF0000020000000
  %v26.1 = fadd contract float %v25.1, 0x3E7FF19E20000000
  %v25.2 = fmul contract float %v26.1, 0x3FF0000020000000
  %v26.2 = fadd contract float %v25.2, 0x3E7FF19E20000000
  %v25.3 = fmul contract float %v26.2, 0x3FF0000020000000
  %v26.3 = fadd contract float %v25.3, 0x3E7FF19E20000000
  %v25.4 = fmul contract float %v26.3, 0x3FF0000020000000
  %v26.4 = fadd contract float %v25.4, 0x3E7FF19E20000000
  %v25.5 = fmul contract float %v26.4, 0x3FF0000020000000
  %v26.5 = fadd contract float %v25.5, 0x3E7FF19E20000000
  %v25.6 = fmul contract float %v26.5, 0x3FF0000020000000
  %v26.6 = fadd contract float %v25.6, 0x3E7FF19E20000000
  %v25.7 = fmul contract float %v26.6, 0x3FF0000020000000
  %v26.7 = fadd contract float %v25.7, 0x3E7FF19E20000000
  %niter.next.7 = add i32 %niter, 8
  %niter.ncmp.7 = icmp eq i32 %niter.next.7, %unroll_iter
  br i1 %niter.ncmp.7, label %bb5.loopexit.unr-lcssa, label %bb4

bb5.loopexit.unr-lcssa:                           ; preds = %bb4
  %lcmp.mod.not = icmp eq i32 %xtraiter, 0
  br i1 %lcmp.mod.not, label %bb5, label %bb4.epil.preheader

bb4.epil.preheader:                               ; preds = %bb5.loopexit.unr-lcssa, %bb4.preheader
  %v212.epil.init = phi float [ %v20, %bb4.preheader ], [ %v26.7, %bb5.loopexit.unr-lcssa ]
  %lcmp.mod5 = icmp ne i32 %xtraiter, 0
  tail call void @llvm.assume(i1 %lcmp.mod5)
  br label %bb4.epil

bb4.epil:                                         ; preds = %bb4.epil, %bb4.epil.preheader
  %v212.epil = phi float [ %v26.epil, %bb4.epil ], [ %v212.epil.init, %bb4.epil.preheader ]
  %epil.iter = phi i32 [ %epil.iter.next, %bb4.epil ], [ 0, %bb4.epil.preheader ]
  %v25.epil = fmul contract float %v212.epil, 0x3FF0000020000000
  %v26.epil = fadd contract float %v25.epil, 0x3E7FF19E20000000
  %epil.iter.next = add i32 %epil.iter, 1
  %epil.iter.cmp.not = icmp eq i32 %epil.iter.next, %xtraiter
  br i1 %epil.iter.cmp.not, label %bb5, label %bb4.epil, !llvm.loop !0

bb5:                                              ; preds = %bb5.loopexit.unr-lcssa, %bb4.epil, %bb2
  %v21.lcssa = phi float [ %v20, %bb2 ], [ %v26.7, %bb5.loopexit.unr-lcssa ], [ %v26.epil, %bb4.epil ]
  %v29.not = icmp ult i64 %v9.i, %v3
  br i1 %v29.not, label %bb6, label %bb9

bb6:                                              ; preds = %bb5
  %v33 = getelementptr inbounds nuw float, ptr %v2, i64 %v9.i
  store float %v21.lcssa, ptr %v33, align 4
  br label %bb9

bb9:                                              ; preds = %bb5, %bb6, %entry
  ret void
}

; Function Attrs: convergent nofree norecurse nounwind memory(argmem: readwrite)
define ptx_kernel void @persistent_worker(ptr readonly captures(none) %v0, i64 %v1, i64 %v2, ptr writeonly captures(none) %v3, i64 %v4) local_unnamed_addr #2 {
entry:
  %v2.i = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #6
  %v3.i = zext nneg i32 %v2.i to i64
  %v4.i = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #6
  %v5.i = zext nneg i32 %v4.i to i64
  %v6.i = tail call i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #6
  %v7.i = zext nneg i32 %v6.i to i64
  %v8.i = mul nuw nsw i64 %v5.i, %v7.i
  %v9.i = add nuw nsw i64 %v8.i, %v3.i
  %v17 = mul i64 %v9.i, -7046029254386353131
  %v203 = load atomic i32, ptr %v0 syncscope("device") acquire, align 4
  %v214 = icmp ne i32 %v203, 0
  %v225 = icmp eq i64 %v2, 0
  %or.cond6 = select i1 %v214, i1 true, i1 %v225
  br i1 %or.cond6, label %bb11, label %bb6.preheader

bb6.preheader:                                    ; preds = %entry, %bb8
  %v198 = phi i64 [ %v29.3, %bb8 ], [ %v17, %entry ]
  %v187 = phi i64 [ %v31, %bb8 ], [ 0, %entry ]
  br label %bb7

bb7:                                              ; preds = %bb7, %bb6.preheader
  %v252 = phi i64 [ %v198, %bb6.preheader ], [ %v29.3, %bb7 ]
  %v241 = phi i32 [ 0, %bb6.preheader ], [ %v30.3, %bb7 ]
  %0 = mul i64 %v252, -8227340366007020463
  %v29.3 = add i64 %0, 8921595095711710844
  %v30.3 = add nuw nsw i32 %v241, 4
  %exitcond.3 = icmp eq i32 %v30.3, 512
  br i1 %exitcond.3, label %bb8, label %bb7

bb8:                                              ; preds = %bb7
  %v31 = add nuw i64 %v187, 1
  %v20 = load atomic i32, ptr %v0 syncscope("device") acquire, align 4
  %v21 = icmp ne i32 %v20, 0
  %v22 = icmp uge i64 %v31, %v2
  %or.cond = select i1 %v21, i1 true, i1 %v22
  br i1 %or.cond, label %bb11.loopexit, label %bb6.preheader

bb11.loopexit:                                    ; preds = %bb8
  %1 = xor i64 %v29.3, %v31
  br label %bb11

bb11:                                             ; preds = %bb11.loopexit, %entry
  %v19.lcssa = phi i64 [ %v17, %entry ], [ %1, %bb11.loopexit ]
  %v33.not = icmp ult i64 %v9.i, %v4
  br i1 %v33.not, label %bb12, label %bb14

bb12:                                             ; preds = %bb11
  %v38 = getelementptr inbounds nuw i64, ptr %v3, i64 %v9.i
  store i64 %v19.lcssa, ptr %v38, align 8
  br label %bb14

bb14:                                             ; preds = %bb11, %bb12
  ret void
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.nvvm.read.ptx.sreg.tid.x() #3

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 2147483647) i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #3

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 1, 1025) i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #3

; Function Attrs: alwaysinline convergent mustprogress nofree norecurse nosync nounwind willreturn memory(none)
define range(i64 0, 2199023254528) i64 @cuda_device____internal__index_1d(ptr readnone captures(none) %v0) local_unnamed_addr #4 {
entry:
  %v2 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #6
  %v3 = zext nneg i32 %v2 to i64
  %v4 = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #6
  %v5 = zext nneg i32 %v4 to i64
  %v6 = tail call i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #6
  %v7 = zext nneg i32 %v6 to i64
  %v8 = mul nuw nsw i64 %v5, %v7
  %v9 = add nuw nsw i64 %v8, %v3
  ret i64 %v9
}

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: write)
declare void @llvm.assume(i1 noundef) #5

attributes #0 = { convergent mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite) }
attributes #1 = { convergent nofree norecurse nosync nounwind memory(argmem: readwrite) }
attributes #2 = { convergent nofree norecurse nounwind memory(argmem: readwrite) }
attributes #3 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #4 = { alwaysinline convergent mustprogress nofree norecurse nosync nounwind willreturn memory(none) }
attributes #5 = { nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: write) }
attributes #6 = { convergent }

!0 = distinct !{!0, !1}
!1 = !{!"llvm.loop.unroll.disable"}
