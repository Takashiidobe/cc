use assert_cmd::cargo::CommandCargoExt;
use insta::assert_yaml_snapshot;
use insta_cmd::Command;
use std::{
    fs::File,
    path::Path,
    process::{Command as StdCommand, Stdio},
};

fn compile_asm_with_cc(asm: &Path, exe: &Path) -> std::io::Result<std::process::Output> {
    let mut cmd = StdCommand::new("cc");
    cmd.arg("-o").arg(exe).arg(asm);
    cmd.stdout(Stdio::piped()).stderr(Stdio::piped()).output()
}

fn run_exe(exe: &Path) -> std::io::Result<std::process::Output> {
    StdCommand::new(exe)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
}

fn run_case(path: &Path) -> datatest_stable::Result<()> {
    let tmp = tempfile::tempdir()?;
    let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("out");
    let asm_path = tmp.path().join(format!("{stem}.S"));
    let exe_path = tmp.path().join(stem);

    let mut bin = Command::cargo_bin(env!("CARGO_PKG_NAME"))?;
    let asm_file = File::create(&asm_path)?;
    let codegen_out = bin
        .args(["--codegen"])
        .arg(path)
        .stdout(Stdio::from(asm_file))
        .stderr(Stdio::piped())
        .output()?;
    eprintln!(
        "[{}] codegen status: {:?}",
        path.display(),
        codegen_out.status.code()
    );
    assert!(codegen_out.status.success(), "codegen failed");

    let compile_out = compile_asm_with_cc(&asm_path, &exe_path)?;
    eprintln!(
        "[{}] cc status: {:?}",
        path.display(),
        compile_out.status.code()
    );
    assert!(compile_out.status.success(), "cc failed");

    let run_out = run_exe(&exe_path)?;
    let exit_code = run_out.status.code().unwrap_or(-1);

    assert_yaml_snapshot!(format!("exit_code: {}", stem), exit_code);

    Ok(())
}

datatest_stable::harness! {
    { test = run_case, root = "./test-files", pattern = r".*" },
}
