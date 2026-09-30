//! GPU energy measurement via NVML, bracketing the same process-start-to-
//! result interval every subcommand's `Instant`-based wall-clock timing
//! already measures -- validates this project's core "cold-start energy"
//! thesis (see docs/DEVELOPMENT.md), currently otherwise completely unmeasured
//! anywhere in this codebase (only wall-clock ms).
//!
//! Unconditional module (like `diagnostics.rs`): every call site
//! (`generate`/`system1`/`smoke`/`bench`/`doctor`) calls
//! [`EnergySampler::start`]/[`EnergySampler::measure`] with no `#[cfg]` of
//! its own. Compiled without the `nvml` Cargo feature, `start` always
//! returns a sampler whose `measure` always returns `None` -- callers
//! already treat a `None` energy measurement as "omit the field", the same
//! precedent `REFLEX_LORA_OK`'s `if let Some(lora_path)` established, so no
//! call site needs its own `#[cfg(feature = "nvml")]`.
//!
//! NVML is loaded at runtime by `nvml-wrapper` via `dlopen`/`LoadLibrary`
//! (through `libloading`), never linked at build time -- so the `nvml`
//! feature never breaks a `REFLEX_SKIP_CUDA=1` dev build, and a binary built
//! with `--features nvml` still runs fine (energy fields simply absent) on a
//! machine/container lacking `libnvidia-ml.so`/`nvml.dll` entirely.
//!
//! **Device-wide, not per-process**: NVML has no per-process energy API.
//! `total_energy_consumption`/`power_usage` report the whole GPU's draw --
//! accurate on a dedicated/rented instance (this project's stated target),
//! a known overcount on a GPU shared with other workloads. Not solved here.
//!
//! **T4-class fallback**: `nvmlDeviceGetTotalEnergyConsumption` (a
//! monotonic millijoule counter) is Volta+ only. On a GPU/driver lacking it
//! (`NvmlError::NotSupported`), this module falls back to polling
//! `nvmlDeviceGetPowerUsage` (instantaneous milliwatts, supported back to
//! Fermi) on its own background thread and numerically integrating
//! `power_mw * dt_s`, at `REFLEX_NVML_POLL_MS`-millisecond cadence (default
//! 10ms -- negligible next to the hundred-ms-to-second cold starts this
//! project measures). **This is not a Non-goals violation**: it is a single
//! internal measurement thread, analogous to a stopwatch, that never
//! accepts a work item and never serves a request -- not the
//! `batch_size`/thread-pool concurrency model docs/DEVELOPMENT.md's Non-goals section
//! governs. It never needs a join/shutdown handshake either: every one-shot
//! `reflex` subcommand calls `reflex_engine::fast_exit` when done, which
//! kills every thread the process owns for free.
//!
//! The fallback is normally reached only when the counter is unsupported. To
//! exercise it on hardware that *does* have the counter (needed because the
//! polled path had never actually run before M4 -- every real-hardware
//! verification had taken the counter branch), set
//! **`REFLEX_NVML_FORCE_POLLED=1`**: that makes [`EnergySampler::start`]/
//! [`probe_availability`] skip the counter branch and use polling even when a
//! counter is available. It is default-off and does **not** change the normal
//! preference order (counter first, polled only as a fallback) -- see
//! [`force_polled_requested`]. `reflex doctor`'s `nvml_energy` check reports
//! whichever method would actually be used, including under this override.
//!
//! Any NVML failure (library missing, driver too old, no permission,
//! unsupported GPU) degrades gracefully to "no measurement available" --
//! never a panic.
//!
//! **Per-phase deltas**: `measure()` returns *cumulative* joules since
//! `start()`, so a caller wanting per-phase energy (see `generate.rs`/
//! `system1.rs`'s `REFLEX_PHASE_OK` lines) snapshots it at each phase
//! boundary and subtracts the previous snapshot. Granularity follows the
//! mode: `PolledPower` integrates continuously so short-phase deltas are
//! meaningful; `TotalEnergyCounter` is a hardware counter the driver updates
//! coarsely, so a very short phase's delta can read as zero or lumpy -- a
//! property of NVML's counter, not of this module's arithmetic.

use std::time::Instant;

#[cfg(feature = "nvml")]
use std::sync::{Arc, Mutex};

/// How an [`EnergyMeasurement`]'s `joules` figure was obtained.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EnergyMethod {
    /// `nvmlDeviceGetTotalEnergyConsumption`: a monotonic hardware counter,
    /// Volta+ only. The precise figure.
    TotalEnergyCounter,
    /// `nvmlDeviceGetPowerUsage` polled and numerically integrated -- used
    /// when the GPU/driver lacks the total-energy counter.
    PolledPower,
}

impl EnergyMethod {
    pub fn as_str(&self) -> &'static str {
        match self {
            EnergyMethod::TotalEnergyCounter => "total_energy_counter",
            EnergyMethod::PolledPower => "polled_power",
        }
    }
}

/// One energy snapshot, from [`EnergySampler::measure`].
#[derive(Debug, Clone)]
pub struct EnergyMeasurement {
    pub joules: f64,
    pub method: EnergyMethod,
}

#[cfg(feature = "nvml")]
enum Inner {
    /// `nvml`/`device_index` are kept (rather than a borrowed `Device<'_>`)
    /// so `Inner` owns everything it needs with no lifetime parameter --
    /// `measure()` re-resolves the device handle from `device_index` each
    /// time, a cheap driver call. `nvml_wrapper::Nvml` is itself a large
    /// struct (~12KB, all the function-pointer table NVML's dynamically
    /// loaded library resolves at init), so it's boxed here rather than
    /// inlined -- otherwise every `Inner` value, even the zero-cost
    /// `Unavailable` one, would pay that size.
    Counter {
        nvml: Box<nvml_wrapper::Nvml>,
        device_index: u32,
        start_mj: u64,
    },
    Polled {
        joules: Arc<Mutex<f64>>,
    },
    Unavailable,
}

/// Measures GPU energy draw from the moment [`EnergySampler::start`] is
/// called. Bracket it around the exact same interval a subcommand's own
/// `Instant::now()`-based wall-clock timing covers (call `start` as the
/// first statement in `run()`, alongside `let t0 = Instant::now();"), so the
/// reported joules and ms figures describe the identical window.
pub struct EnergySampler {
    #[cfg(feature = "nvml")]
    inner: Inner,
    _start: Instant,
}

#[cfg(feature = "nvml")]
fn explain_nvml_error(e: &nvml_wrapper::error::NvmlError) -> String {
    use nvml_wrapper::error::NvmlError;
    match e {
        NvmlError::LibloadingError(_) => {
            "NVML library not found (libnvidia-ml.so/nvml.dll not present on this system)"
                .to_string()
        }
        NvmlError::DriverNotLoaded => "NVIDIA driver is not loaded".to_string(),
        NvmlError::Uninitialized => "NVML failed to initialize".to_string(),
        NvmlError::NoPermission => "no permission to query NVML on this device".to_string(),
        NvmlError::NotSupported => {
            "not supported by this GPU/driver (e.g. no total-energy counter)".to_string()
        }
        other => format!("{other}"),
    }
}

/// Whether `REFLEX_NVML_FORCE_POLLED` asks this process to skip the
/// total-energy-counter branch and use [`poll_power`] even when the counter is
/// available. Default-off (unset, or `0`/`false`/`no`/`off`, case-insensitive)
/// leaves the normal preference order -- counter first, polled only as a
/// fallback -- completely unchanged. Exists so the fallback path can be
/// exercised on real hardware that does have the counter (see this module's
/// doc comment and README's energy section); `reflex doctor` reads it too so
/// its `nvml_energy` check reports what a real run would actually use.
pub fn force_polled_requested() -> bool {
    match std::env::var("REFLEX_NVML_FORCE_POLLED") {
        Ok(v) => !matches!(
            v.trim().to_ascii_lowercase().as_str(),
            "" | "0" | "false" | "no" | "off"
        ),
        Err(_) => false,
    }
}

#[cfg(feature = "nvml")]
fn poll_power(joules: Arc<Mutex<f64>>, device_index: u32) {
    let poll_ms: u64 = std::env::var("REFLEX_NVML_POLL_MS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(10);
    let nvml = match nvml_wrapper::Nvml::init() {
        Ok(n) => n,
        Err(_) => return,
    };
    let device = match nvml.device_by_index(device_index) {
        Ok(d) => d,
        Err(_) => return,
    };
    let mut last = Instant::now();
    loop {
        std::thread::sleep(std::time::Duration::from_millis(poll_ms));
        let now = Instant::now();
        let dt_s = now.duration_since(last).as_secs_f64();
        last = now;
        if let Ok(power_mw) = device.power_usage() {
            if let Ok(mut j) = joules.lock() {
                *j += (power_mw as f64 / 1000.0) * dt_s;
            }
        }
    }
}

impl EnergySampler {
    /// Starts measuring immediately. Never panics -- a fully-degraded
    /// sampler (compiled without `nvml`, or NVML unavailable on this
    /// machine) is a valid, cheap value whose `measure()` always returns
    /// `None`.
    pub fn start(device_ordinal: usize) -> EnergySampler {
        let _start = Instant::now();
        #[cfg(feature = "nvml")]
        {
            let inner = Self::start_inner(device_ordinal as u32);
            EnergySampler { inner, _start }
        }
        #[cfg(not(feature = "nvml"))]
        {
            let _ = device_ordinal;
            EnergySampler { _start }
        }
    }

    #[cfg(feature = "nvml")]
    fn start_inner(device_index: u32) -> Inner {
        let nvml = match nvml_wrapper::Nvml::init() {
            Ok(n) => n,
            Err(_) => return Inner::Unavailable,
        };
        let device = match nvml.device_by_index(device_index) {
            Ok(d) => d,
            Err(_) => return Inner::Unavailable,
        };
        // Prefer the precise hardware counter unless the fallback was
        // explicitly forced (see `force_polled_requested`).
        if !force_polled_requested() {
            if let Ok(start_mj) = device.total_energy_consumption() {
                return Inner::Counter {
                    nvml: Box::new(nvml),
                    device_index,
                    start_mj,
                };
            }
        }
        // This handshake's `nvml`/`device` are dropped here; the polling
        // thread does its own fresh `Nvml::init()` instead of borrowing
        // these, so `Inner` never has to name a borrowed `Device<'_>`'s
        // lifetime.
        let joules = Arc::new(Mutex::new(0.0));
        let joules_thread = joules.clone();
        std::thread::spawn(move || poll_power(joules_thread, device_index));
        Inner::Polled { joules }
    }

    /// Snapshot at the checkpoint a subcommand wants to report (e.g. "first
    /// token"). `None` means no energy figure is available -- compiled
    /// without `nvml`, or NVML unavailable/unsupported on this machine.
    /// Callers should omit the field entirely in that case, not print a
    /// sentinel.
    pub fn measure(&self) -> Option<EnergyMeasurement> {
        #[cfg(feature = "nvml")]
        {
            match &self.inner {
                Inner::Counter {
                    nvml,
                    device_index,
                    start_mj,
                } => {
                    let device = nvml.device_by_index(*device_index).ok()?;
                    let now_mj = device.total_energy_consumption().ok()?;
                    let delta_mj = now_mj.saturating_sub(*start_mj);
                    Some(EnergyMeasurement {
                        joules: delta_mj as f64 / 1000.0,
                        method: EnergyMethod::TotalEnergyCounter,
                    })
                }
                Inner::Polled { joules } => {
                    let joules = *joules.lock().ok()?;
                    Some(EnergyMeasurement {
                        joules,
                        method: EnergyMethod::PolledPower,
                    })
                }
                Inner::Unavailable => None,
            }
        }
        #[cfg(not(feature = "nvml"))]
        {
            None
        }
    }
}

/// One-shot check of whether energy measurement would work on this device --
/// used by `reflex doctor` (see `src/bin/reflex/doctor.rs`) to report the
/// method that would be used, or why neither is available, without starting
/// any actual sampling.
pub fn probe_availability(device_ordinal: usize) -> Result<EnergyMethod, String> {
    #[cfg(feature = "nvml")]
    {
        let nvml = nvml_wrapper::Nvml::init().map_err(|e| explain_nvml_error(&e))?;
        let device = nvml
            .device_by_index(device_ordinal as u32)
            .map_err(|e| explain_nvml_error(&e))?;
        if !force_polled_requested() && device.total_energy_consumption().is_ok() {
            return Ok(EnergyMethod::TotalEnergyCounter);
        }
        device
            .power_usage()
            .map(|_| EnergyMethod::PolledPower)
            .map_err(|e| explain_nvml_error(&e))
    }
    #[cfg(not(feature = "nvml"))]
    {
        let _ = device_ordinal;
        Err(
            "this binary was built without the `nvml` Cargo feature (rebuild with --features nvml)"
                .to_string(),
        )
    }
}
