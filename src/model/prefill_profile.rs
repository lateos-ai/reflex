//! `REFLEX_PREFILL_PROFILE=1`: a per-phase GPU-time split of a prefill, by layer
//! type (Gemma 4's sliding vs global layers). Off by default. When it's on, it
//! records one timing event on the compute stream at every phase boundary and
//! drains the device once at the end of the prefill to read them back. The
//! events don't serialize the stream, so the totals are what the prefill costs.

use super::*;
use crate::error::ReflexError;

pub(super) fn prefill_profile_enabled() -> bool {
    std::env::var("REFLEX_PREFILL_PROFILE").is_ok_and(|v| v == "1")
}

/// The phase a span of GPU time is charged to: everything enqueued since the
/// previous mark.
#[derive(Clone, Copy)]
pub(super) enum PrefillPhase {
    Embed,
    QkvGemm,
    NormRope,
    KvCopy,
    Attention,
    OProj,
    Ffn,
    FfnTail,
}

const PHASES: [&str; 8] = [
    "embed",
    "qkv_gemm",
    "norm_rope",
    "kv_copy",
    "attn",
    "o_proj",
    "ffn",
    "ffn_tail",
];

pub(super) struct PrefillProfile {
    stream: sys::CUstream,
    start: sys::CUevent,
    /// (phase, layer kind: 0 = sliding/plain, 1 = global, event at the end).
    marks: Vec<(PrefillPhase, usize, sys::CUevent)>,
    wall: std::time::Instant,
}

impl PrefillProfile {
    pub(super) fn start(device: &Arc<CudaDevice>) -> Result<Self, ReflexError> {
        let stream = *device.cu_stream();
        let start = Self::record(stream)?;
        Ok(Self {
            stream,
            start,
            marks: Vec::new(),
            wall: std::time::Instant::now(),
        })
    }

    fn record(stream: sys::CUstream) -> Result<sys::CUevent, ReflexError> {
        let ev = result::event::create(sys::CUevent_flags::CU_EVENT_DEFAULT)
            .map_err(|e| crate::gpu_err!(e, "create prefill profile event: {e}"))?;
        unsafe { result::event::record(ev, stream) }
            .map_err(|e| crate::gpu_err!(e, "record prefill profile event: {e}"))?;
        Ok(ev)
    }

    /// Charges everything enqueued since the last mark to `phase` on a layer
    /// of kind `global` (false for sliding layers and for non-layer phases).
    pub(super) fn mark(&mut self, phase: PrefillPhase, global: bool) -> Result<(), ReflexError> {
        let ev = Self::record(self.stream)?;
        self.marks.push((phase, global as usize, ev));
        Ok(())
    }

    /// Drains the device and prints one `REFLEX_PREFILL_PROFILE` line: GPU ms
    /// per phase and layer type, the GPU total, the host enqueue time (wall
    /// time until this call) and the wall time including the drain.
    pub(super) fn report(self, device: &Arc<CudaDevice>, rows: usize) -> Result<(), ReflexError> {
        let enqueue_ms = self.wall.elapsed().as_secs_f64() * 1000.0;
        device
            .synchronize()
            .map_err(|e| crate::gpu_err!(e, "prefill profile: synchronize: {e}"))?;
        let wall_ms = self.wall.elapsed().as_secs_f64() * 1000.0;
        let mut table = [[0f64; PHASES.len()]; 2];
        let mut prev = self.start;
        let mut total = 0f64;
        for &(phase, kind, ev) in &self.marks {
            let ms = unsafe { result::event::elapsed(prev, ev) }.unwrap_or(0.0) as f64;
            table[kind][phase as usize] += ms;
            total += ms;
            prev = ev;
        }
        unsafe {
            let _ = result::event::destroy(self.start);
            for &(_, _, ev) in &self.marks {
                let _ = result::event::destroy(ev);
            }
        }
        let mut line = format!(
            "REFLEX_PREFILL_PROFILE rows={rows} embed_ms={:.2}",
            table[0][PrefillPhase::Embed as usize]
        );
        for (kind, name) in [(0, "sliding"), (1, "global")] {
            for (i, phase) in PHASES.iter().enumerate().skip(1) {
                if table[kind][i] > 0.0 {
                    line += &format!(" {name}.{phase}_ms={:.2}", table[kind][i]);
                }
            }
        }
        line += &format!(
            " gpu_total_ms={total:.2} host_enqueue_ms={enqueue_ms:.2} wall_ms={wall_ms:.2}"
        );
        eprintln!("{line}");
        Ok(())
    }
}

/// [`PrefillProfile::mark`] if profiling, a no-op otherwise.
pub(super) fn prof_mark(
    prof: &mut Option<PrefillProfile>,
    phase: PrefillPhase,
    global: bool,
) -> Result<(), ReflexError> {
    match prof {
        Some(p) => p.mark(phase, global),
        None => Ok(()),
    }
}
