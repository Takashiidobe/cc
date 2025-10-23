use assert_cmd::cargo::CommandCargoExt;
use insta_cmd::Command;
use serde::Serialize;
use std::{
    fs,
    fs::File,
    path::{Path, PathBuf},
    process::{Command as StdCommand, Output, Stdio},
};
use tempfile::tempdir;

fn compile(src: &Path, exe: &Path) -> std::io::Result<Output> {
    let mut cmd = StdCommand::new("cc");
    cmd.arg("-o").arg(exe).arg(src);
    cmd.stdout(Stdio::piped()).stderr(Stdio::piped()).output()
}

fn run_exe(exe: &Path) -> std::io::Result<Output> {
    StdCommand::new(exe)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
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

fn ensure_success(tag: &str, target: &Path, out: &Output) {
    assert!(
        out.status.success(),
        "[{}] {} failed (status: {:?})\n--- stdout ---\n{}\n--- stderr ---\n{}",
        target.display(),
        tag,
        out.status.code(),
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr),
    );
}

fn ensure_failure(tag: &str, target: &Path, out: &Output) {
    assert!(
        !out.status.success(),
        "[{}] {} unexpectedly succeeded\n--- stdout ---\n{}\n--- stderr ---\n{}",
        target.display(),
        tag,
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr),
    );
}

fn codegen(src: &Path, asm: &Path) -> Output {
    let asm_file = File::create(asm).expect("failed to create asm file");
    let mut bin = Command::cargo_bin(env!("CARGO_PKG_NAME")).expect("missing binary under test");
    bin.args(["--codegen"])
        .arg(src)
        .stdout(Stdio::from(asm_file))
        .stderr(Stdio::piped())
        .output()
        .expect("failed to execute compiler --codegen")
}

fn make_random_c(temp: &Path) -> PathBuf {
    let mut bin = Command::cargo_bin(env!("CARGO_PKG_NAME")).expect("missing binary under test");
    let fuzz_out = bin
        .arg("--fuzz")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .expect("failed to execute compiler --fuzz");

    ensure_success("--fuzz", temp, &fuzz_out);

    let src_path = temp.join("something.c");
    fs::write(&src_path, &fuzz_out.stdout).expect("failed to write fuzz source");
    src_path
}

#[test]
fn fuzz_codegen_matches_cc() {
    let tmp = tempdir().expect("failed to create temp dir");
    let tmp_path = tmp.path();

    let src_path = make_random_c(tmp_path);
    let asm_path = tmp_path.join("something.S");
    let exe_mine = tmp_path.join("something.mine");
    let exe_ref = tmp_path.join("something.ref");

    let codegen_out = codegen(&src_path, &asm_path);
    ensure_success("--codegen", &src_path, &codegen_out);

    let compile_out_mine = compile(&asm_path, &exe_mine).expect("failed to run cc on asm");
    ensure_success("cc(asm)", &asm_path, &compile_out_mine);

    let run_out_mine = run_exe(&exe_mine).expect("failed to run mine exe");
    let mine = to_runlog(run_out_mine);

    let compile_out_ref = compile(&src_path, &exe_ref).expect("failed to run cc on original src");
    ensure_success("cc(src)", &src_path, &compile_out_ref);

    let run_out_ref = run_exe(&exe_ref).expect("failed to run ref exe");
    let reference = to_runlog(run_out_ref);

    if mine != reference {
        let mut msg = String::new();
        use std::fmt::Write;
        writeln!(&mut msg, "\n=== MISMATCH for {} ===", src_path.display()).ok();

        if mine.status != reference.status {
            writeln!(
                &mut msg,
                "Exit code differs: mine={} ref={}",
                mine.status, reference.status
            )
            .ok();
        }
        if mine.stdout != reference.stdout {
            writeln!(&mut msg, "\n--- stdout (mine) ---\n{}", mine.stdout).ok();
            writeln!(&mut msg, "\n--- stdout (ref)  ---\n{}", reference.stdout).ok();
        }

        panic!("{msg}");
    }

    let invalid_path = tmp_path.join("invalid.c");
    fs::write(&invalid_path, "int main() { return; }\n").expect("failed to write invalid source");

    let asm_invalid = tmp_path.join("invalid.S");
    let exe_invalid = tmp_path.join("invalid.mine");

    let invalid_codegen = codegen(&invalid_path, &asm_invalid);
    ensure_failure("--codegen", &invalid_path, &invalid_codegen);

    let invalid_compile =
        compile(&invalid_path, &exe_invalid).expect("failed to run cc on invalid");
    ensure_failure("cc(src)", &invalid_path, &invalid_compile);
}
