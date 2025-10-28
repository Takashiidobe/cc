use assert_cmd::cargo::CommandCargoExt as _;
use libtest_mimic::{Arguments, Failed, Trial};
use serde::Serialize;
use std::{
    fs,
    path::{Path, PathBuf},
    process::{Command as StdCommand, Output, Stdio},
    time::Duration,
};
use tempfile::tempdir;

fn main() {
    let args = Arguments::from_args();

    let count = 100usize;
    let base_seed = 0xC0D3_C0DEu64;

    let mut tests = Vec::with_capacity(count);
    for i in 0..count {
        let name = format!("fuzz_{:03}", i);
        let seed = base_seed ^ (i as u64).wrapping_mul(0x9E3779B185EBCA87);

        tests.push(Trial::test(name, move || {
            run_one(seed).map_err(Failed::from)
        }));
    }

    libtest_mimic::run(&args, tests).exit();
}

fn run_one(seed: u64) -> Result<(), String> {
    let tmp = tempdir().map_err(|e| e.to_string())?;
    let tmp_path = tmp.path();
    let src_path = make_random_c(tmp_path, seed)?;

    let result = (|| {
        let asm_path = tmp_path.join("a.S");
        let exe_mine = tmp_path.join("mine");
        let exe_ref = tmp_path.join("ref");

        must_success("--codegen", &src_path, codegen(&src_path, &asm_path)?)?;
        must_success("cc(asm)", &asm_path, compile(&asm_path, &exe_mine)?)?;
        let mine = to_runlog(run_exe(&exe_mine)?);

        must_success("cc(src)", &src_path, compile(&src_path, &exe_ref)?)?;
        let reference = to_runlog(run_exe(&exe_ref)?);

        if mine != reference {
            return Err(format!(
                "output mismatch:\n mine={mine:?}\n ref={reference:?}"
            ));
        }
        Ok(())
    })();

    if let Err(e) = result {
        let msg = persist_failure(&src_path, tmp_path, seed, &e);
        return Err(msg);
    }
    Ok(())
}

#[derive(Serialize, Debug, Clone, PartialEq, Eq)]
struct RunLog {
    stdout: String,
    status: i32,
}
fn to_runlog(out: Output) -> RunLog {
    RunLog {
        stdout: String::from_utf8_lossy(&out.stdout).to_string(),
        status: out.status.code().unwrap_or(-1),
    }
}

fn codegen(src: &Path, asm: &Path) -> Result<Output, String> {
    let asm_file = std::fs::File::create(asm).map_err(|e| e.to_string())?;
    let mut bin =
        insta_cmd::Command::cargo_bin(env!("CARGO_PKG_NAME")).map_err(|e| e.to_string())?;
    bin.args(["--codegen"])
        .arg(src)
        .stdout(Stdio::from(asm_file))
        .stderr(Stdio::piped())
        .output()
        .map_err(|e| e.to_string())
}

fn make_random_c(temp: &Path, seed: u64) -> Result<PathBuf, String> {
    let mut bin =
        insta_cmd::Command::cargo_bin(env!("CARGO_PKG_NAME")).map_err(|e| e.to_string())?;
    let fuzz_out = bin
        .arg("--fuzz")
        .env("SEED", seed.to_string())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .map_err(|e| e.to_string())?;
    if !fuzz_out.status.success() {
        return Err(format!(
            "--fuzz failed:\n{}",
            String::from_utf8_lossy(&fuzz_out.stderr)
        ));
    }
    let src_path = temp.join("prog.c");
    fs::write(&src_path, &fuzz_out.stdout).map_err(|e| e.to_string())?;
    Ok(src_path)
}

fn compile(src: &Path, exe: &Path) -> Result<Output, String> {
    let mut cmd = StdCommand::new("cc");
    cmd.arg("-o").arg(exe).arg(src);
    cmd.stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .map_err(|e| e.to_string())
}

fn run_exe(exe: &Path) -> Result<Output, String> {
    let duration = Duration::from_secs(1);
    let mut child = StdCommand::new(exe)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| e.to_string())?;
    let start = std::time::Instant::now();
    loop {
        match child.try_wait().map_err(|e| e.to_string())? {
            Some(status) => {
                let output = child.wait_with_output().map_err(|e| e.to_string())?;
                return Ok(Output { status, ..output });
            }
            None if start.elapsed() > duration => {
                let _ = child.kill();
                let _ = child.wait();
                return Err(format!("program timed out after {:?}", duration));
            }
            None => std::thread::sleep(Duration::from_millis(10)),
        }
    }
}

fn must_success(tag: &str, target: &Path, out: Output) -> Result<(), String> {
    if out.status.success() {
        return Ok(());
    }
    Err(format!(
        "[{}] {} failed (status: {:?})\n--- stdout ---\n{}\n--- stderr ---\n{}",
        target.display(),
        tag,
        out.status.code(),
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr),
    ))
}

fn persist_failure(src: &Path, tmp: &Path, seed: u64, msg: &str) -> String {
    let outdir =
        PathBuf::from(std::env::var("CARGO_TARGET_DIR").unwrap_or_else(|_| "target".into()))
            .join("fuzz-failures")
            .join(format!("{seed:016x}"));
    let _ = fs::create_dir_all(&outdir);
    let _ = fs::copy(src, outdir.join("prog.c"));
    for name in ["a.S", "mine", "ref"] {
        let p = tmp.join(name);
        if p.exists() {
            let _ = fs::copy(&p, outdir.join(name));
        }
    }
    format!("{msg}\nArtifacts: {}", outdir.display())
}
