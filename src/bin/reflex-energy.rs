//! `reflex-energy`: measures the GPU energy a command spends, from *outside* it.
//!
//! Usage: `reflex-energy [--idle-ms N] [--settle-ms N] [--device N] -- <command> [args...]`
//!
//! 1. Reads NVML's total-energy counter (`nvmlDeviceGetTotalEnergyConsumption`, Volta
//!    and newer, millijoules) over `--idle-ms` (default 2000) of idle time, to get the
//!    GPU's idle power.
//! 2. Reads the counter, spawns the command (stdio inherited), waits for it to exit,
//!    waits `--settle-ms` more (default 100) so the counter registers the tail of the
//!    run, and reads the counter again.
//! 3. Prints one `REFLEX_ENERGY_OK` line (see `reflex_engine::energy::ExternalEnergy`):
//!    wall time, gross energy over the window, the idle baseline for that window, and
//!    energy net of idle. Exits with the command's exit code.
//!
//! This complements `reflex`'s own in-process energy fields, which can only bracket
//! `main()`: this also covers exec, dynamic linking and CUDA init before `main()`, and
//! the driver's teardown after exit -- matching how the cold-start benchmarks already
//! time from outside the process. The counter is device-wide, so run it on a dedicated
//! GPU with nothing else active. Built only with `--features nvml`.

use reflex_engine::energy::ExternalEnergy;
use std::process::{Command, ExitCode};
use std::thread::sleep;
use std::time::{Duration, Instant};

const USAGE: &str =
    "usage: reflex-energy [--idle-ms N] [--settle-ms N] [--device N] -- <command> [args...]";

struct Opts {
    idle_ms: u64,
    settle_ms: u64,
    device: u32,
    command: Vec<String>,
}

fn parse(args: Vec<String>) -> Result<Opts, String> {
    let mut opts = Opts {
        idle_ms: 2000,
        settle_ms: 100,
        device: 0,
        command: Vec::new(),
    };
    let mut args = args.into_iter();
    while let Some(arg) = args.next() {
        let mut number = |flag: &str| -> Result<u64, String> {
            let raw = args.next().ok_or(format!("{flag} requires a number"))?;
            raw.parse()
                .map_err(|_| format!("{flag}: not a valid number: {raw}"))
        };
        match arg.as_str() {
            "--idle-ms" => opts.idle_ms = number("--idle-ms")?,
            "--settle-ms" => opts.settle_ms = number("--settle-ms")?,
            "--device" => opts.device = number("--device")? as u32,
            "--" => {
                opts.command = args.collect();
                break;
            }
            other => return Err(format!("unexpected argument {other:?}\n{USAGE}")),
        }
    }
    if opts.command.is_empty() {
        return Err(format!("no command given after --\n{USAGE}"));
    }
    if opts.idle_ms == 0 {
        return Err("--idle-ms must be at least 1".to_string());
    }
    Ok(opts)
}

fn run(opts: Opts) -> Result<(ExternalEnergy, i32), String> {
    let nvml = nvml_wrapper::Nvml::init().map_err(|e| format!("NVML init failed: {e}"))?;
    let device = nvml
        .device_by_index(opts.device)
        .map_err(|e| format!("NVML device {}: {e}", opts.device))?;
    let counter = || {
        device.total_energy_consumption().map_err(|e| {
            format!("reading the NVML energy counter (needs a Volta or newer GPU): {e}")
        })
    };

    let idle_start_mj = counter()?;
    let idle_start = Instant::now();
    sleep(Duration::from_millis(opts.idle_ms));
    let idle_mj = counter()?.saturating_sub(idle_start_mj);
    let idle_power_mw = idle_mj as f64 / idle_start.elapsed().as_secs_f64();

    let start_mj = counter()?;
    let start = Instant::now();
    let status = Command::new(&opts.command[0])
        .args(&opts.command[1..])
        .status()
        .map_err(|e| format!("running {:?}: {e}", opts.command[0]))?;
    let wall = start.elapsed();
    sleep(Duration::from_millis(opts.settle_ms));
    let window = start.elapsed();
    let gross_mj = counter()?.saturating_sub(start_mj);

    let measurement = ExternalEnergy {
        wall_ms: wall.as_secs_f64() * 1000.0,
        window_ms: window.as_secs_f64() * 1000.0,
        gross_mj: gross_mj as f64,
        idle_power_mw,
    };
    Ok((measurement, status.code().unwrap_or(-1)))
}

fn main() -> ExitCode {
    let opts = match parse(std::env::args().skip(1).collect()) {
        Ok(o) => o,
        Err(e) => {
            eprintln!("{e}");
            return ExitCode::from(2);
        }
    };
    match run(opts) {
        Ok((measurement, exit_code)) => {
            println!("{}", measurement.report_line(exit_code));
            ExitCode::from(exit_code.clamp(0, 255) as u8)
        }
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::from(1)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::parse;

    fn args(s: &[&str]) -> Vec<String> {
        s.iter().map(|a| a.to_string()).collect()
    }

    #[test]
    fn parses_flags_and_command() {
        let o = parse(args(&["--idle-ms", "500", "--", "reflex", "smoke"])).unwrap();
        assert_eq!((o.idle_ms, o.settle_ms, o.device), (500, 100, 0));
        assert_eq!(o.command, ["reflex", "smoke"]);
    }

    #[test]
    fn rejects_missing_command_and_bad_numbers() {
        assert!(parse(args(&["--idle-ms", "500"])).is_err());
        assert!(parse(args(&["--idle-ms", "soon", "--", "x"])).is_err());
        assert!(parse(args(&["--idle-ms", "0", "--", "x"])).is_err());
        assert!(parse(args(&["--bogus", "--", "x"])).is_err());
    }
}
