fn main() {
    // Only build and link embedded shims when the feature is enabled.
    // We ship cudart/cublas/cublasLt shims so PyTorch can start even if the
    // system libraries are absent (e.g. missing cublasGetMathMode).
    let embed = std::env::var("CARGO_FEATURE_EMBED_CUDART").is_ok();
    if embed {
        println!("cargo:rerun-if-changed=src/cudart_shim.c");
        println!("cargo:rerun-if-changed=src/cudart_dax_pool.c");
        println!("cargo:rerun-if-changed=src/cudart_dax_pool.h");
        println!("cargo:rerun-if-changed=src/cublas_shim.c");
        println!("cargo:rerun-if-changed=src/cublaslt_shim.c");
        println!("cargo:rerun-if-changed=src/cusparse_shim.c");
        println!("cargo:rerun-if-changed=src/cufft_shim.c");
        println!("cargo:rerun-if-changed=src/nccl_shim.c");
        println!("cargo:rerun-if-changed=src/torch_abi_shim.c");

        let cargo_manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
        let tools_dir = std::path::Path::new(&cargo_manifest_dir).parent().unwrap();

        let cudart_map = tools_dir.join("tools/cudart_shim/cudart_shim.map");
        let cublas_map = tools_dir.join("tools/cublas_shim/cublas_shim.map");
        let cublaslt_map = tools_dir.join("tools/cublaslt_shim/cublaslt_shim.map");
        println!("cargo:rerun-if-changed={}", cudart_map.display());
        println!("cargo:rerun-if-changed={}", cublas_map.display());
        println!("cargo:rerun-if-changed={}", cublaslt_map.display());

        let out_dir = std::env::var("OUT_DIR").unwrap();
        let profile_dir = std::path::Path::new(&out_dir)
            .parent()
            .and_then(|p| p.parent())
            .and_then(|p| p.parent())
            .expect("OUT_DIR should be under target/<profile>/build/<pkg>/out");

        let enable_logs = std::env::var("HETGPU_DEBUG_LOGS")
            .map(|v| matches!(v.as_str(), "1" | "true" | "TRUE" | "on"))
            .unwrap_or(false);

        let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
        let is_macos = target_os == "macos";

        let shim_so = profile_dir.join(if is_macos {
            "libhetgpu_cuda_shim.dylib"
        } else {
            "libhetgpu_cuda_shim.so"
        });
        let version_script = profile_dir.join("hetgpu_cuda_shim.map");
        if !is_macos {
            let mut combined_version_script = String::new();
            for map in [&cudart_map, &cublas_map, &cublaslt_map] {
                combined_version_script.push_str(
                    &std::fs::read_to_string(map).expect("failed to read CUDA shim version map"),
                );
                combined_version_script.push('\n');
            }
            std::fs::write(&version_script, combined_version_script)
                .expect("failed to write combined CUDA shim version map");
        }
        let compiler = cc::Build::new().get_compiler();
        let mut shim_build = compiler.to_command();
        if is_macos {
            shim_build.arg("-dynamiclib");
            shim_build.arg("-undefined");
            shim_build.arg("dynamic_lookup");
        } else {
            shim_build.arg("-shared");
        }
        shim_build.arg("-fPIC");
        shim_build.arg("-Wno-unused-parameter");
        shim_build.arg("-D_GLIBCXX_USE_CXX11_ABI=0");
        if enable_logs {
            shim_build.arg("-DHETGPU_DEBUG_LOGS");
        }
        if std::env::var("CARGO_FEATURE_NVIDIA").is_ok()
            && std::env::var("CARGO_FEATURE_SIFIVE").is_err()
        {
            shim_build.arg("-DHETGPU_SHIM_ENABLE_REAL_CUBLAS_BY_DEFAULT");
        }
        shim_build.arg("-o");
        shim_build.arg(&shim_so);
        shim_build.arg("src/cudart_shim.c");
        shim_build.arg("src/cudart_dax_pool.c");
        shim_build.arg("src/cublas_shim.c");
        shim_build.arg("src/cublaslt_shim.c");
        shim_build.arg("src/cusparse_shim.c");
        shim_build.arg("src/cufft_shim.c");
        shim_build.arg("src/nccl_shim.c");
        shim_build.arg("src/torch_abi_shim.c");
        if !is_macos {
            shim_build.arg("-Wl,-soname,libhetgpu_cuda_shim.so");
            shim_build.arg(format!("-Wl,--version-script={}", version_script.display()));
            shim_build.arg("-ldl");
        }
        shim_build.arg("-pthread");
        shim_build.arg("-lz");
        let status = shim_build.status().unwrap();
        assert!(status.success(), "failed to build embedded CUDA shim");

        let nccl_so = profile_dir.join(if is_macos {
            "libnccl.2.dylib"
        } else {
            "libnccl.so.2"
        });
        let mut nccl_build = compiler.to_command();
        if is_macos {
            nccl_build.arg("-dynamiclib");
            nccl_build.arg("-undefined");
            nccl_build.arg("dynamic_lookup");
        } else {
            nccl_build.arg("-shared");
        }
        nccl_build.arg("-fPIC");
        nccl_build.arg("-Wno-unused-parameter");
        if enable_logs {
            nccl_build.arg("-DHETGPU_DEBUG_LOGS");
        }
        nccl_build.arg("-o");
        nccl_build.arg(&nccl_so);
        nccl_build.arg("src/nccl_shim.c");
        if !is_macos {
            nccl_build.arg("-Wl,-soname,libnccl.so.2");
        }
        let status = nccl_build.status().unwrap();
        assert!(status.success(), "failed to build embedded NCCL shim");

        let nccl_link = profile_dir.join(if is_macos {
            "libnccl.dylib"
        } else {
            "libnccl.so"
        });
        let _ = std::fs::remove_file(&nccl_link);
        #[cfg(unix)]
        std::os::unix::fs::symlink(
            if is_macos {
                "libnccl.2.dylib"
            } else {
                "libnccl.so.2"
            },
            &nccl_link,
        )
        .unwrap();

        println!("cargo:rustc-link-search=native={}", profile_dir.display());
        if is_macos {
            println!("cargo:rustc-link-arg=-Wl,-rpath,@loader_path");
            println!("cargo:rustc-link-lib=dylib=hetgpu_cuda_shim");
        } else {
            println!("cargo:rustc-link-arg=-Wl,-rpath,$ORIGIN");
            println!("cargo:rustc-link-arg=-Wl,--no-as-needed");
            println!("cargo:rustc-link-arg=-lhetgpu_cuda_shim");
            println!("cargo:rustc-link-arg=-Wl,--as-needed");
        }
    }
}
