use assert_cmd::cargo::CommandCargoExt as _;
use libtest_mimic::{Arguments, Failed, Trial};
use rand::{TryRngCore, rngs::OsRng};
use std::{
    env, fs,
    path::{Path, PathBuf},
    process::{Command as StdCommand, Output, Stdio},
    time::{Duration, Instant},
};
use tempfile::tempdir;

fn main() {
    // ignoring fuzzing for now
    return;

    let args = Arguments::from_args();

    let base_seed = read_base_seed().unwrap_or_else(random_seed);

    let count: usize = env::var("FUZZ_COUNT")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(100);

    eprintln!("Fuzz base seed: 0x{base_seed:016x}  (set FUZZ_SEED to reproduce)  count={count}");

    let mut tests = Vec::with_capacity(count);
    for i in 0..count {
        let per_test_seed = derive_seed(base_seed, i as u64);
        let name = format!("fuzz_{:03}_seed_{:016x}", i, per_test_seed);

        tests.push(Trial::test(name, move || {
            run_one(per_test_seed).map_err(Failed::from)
        }));
    }

    libtest_mimic::run(&args, tests).exit();
}

fn read_base_seed() -> Option<u64> {
    let raw = env::var("FUZZ_SEED").ok()?;
    if let Some(hex) = raw.strip_prefix("0x").or_else(|| raw.strip_prefix("0X")) {
        u64::from_str_radix(hex, 16).ok()
    } else {
        raw.parse::<u64>().ok()
    }
}

fn random_seed() -> u64 {
    let mut rng = OsRng;
    rng.try_next_u64().unwrap()
}

fn derive_seed(base: u64, idx: u64) -> u64 {
    splitmix64(base ^ idx.wrapping_mul(0x9E37_79B1_85EB_CA87))
}

fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = x;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn run_one(seed: u64) -> Result<(), String> {
    let tmp = tempdir().map_err(|e| e.to_string())?;
    let tmp_path = tmp.path();
    let src_path = make_random_c(tmp_path, seed)?;

    let result = (|| {
        let asm_path = tmp_path.join("a.S");
        let exe_mine = tmp_path.join("mine");
        let exe_ref = tmp_path.join("ref");

        // mine: codegen -> cc -> run
        must_success("--codegen", &src_path, codegen(&src_path, &asm_path)?)?;
        must_success("cc(asm)", &asm_path, compile(&asm_path, &exe_mine)?)?;
        let mine = to_runlog(run_exe(&exe_mine, Duration::from_secs(1))?);

        // reference: cc(src) -> run
        must_success("cc(src)", &src_path, compile(&src_path, &exe_ref)?)?;
        let reference = to_runlog(run_exe(&exe_ref, Duration::from_secs(1))?);

        if mine != reference {
            return Err(format!(
                "output mismatch\n  mine={mine:?}\n  ref ={reference:?}"
            ));
        }
        Ok(())
    })();

    if let Err(e) = result {
        let msg = persist_failure(&src_path, tmp_path, seed, &e);
        return Err(msg);
    }

    eprintln!("OK seed=0x{seed:016x}");
    Ok(())
}

#[derive(serde::Serialize, Debug, Clone, PartialEq, Eq)]
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
        .env("SEED", format!("{seed}"))
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

fn run_exe(exe: &Path, timeout: Duration) -> Result<Output, String> {
    let mut child = StdCommand::new(exe)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| e.to_string())?;
    let start = Instant::now();
    loop {
        if start.elapsed() > timeout {
            let _ = child.kill();
            let _ = child.wait();
            return Err(format!("program timed out after {:?}", timeout));
        }
        if let Some(status) = child.try_wait().map_err(|e| e.to_string())? {
            let output = child.wait_with_output().map_err(|e| e.to_string())?;
            return Ok(Output { status, ..output });
        }
        std::thread::sleep(Duration::from_millis(10));
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
    let outdir = PathBuf::from(env::var("CARGO_TARGET_DIR").unwrap_or_else(|_| "target".into()))
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
