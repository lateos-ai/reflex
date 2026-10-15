//! `REFLEX_PREFILL_PROFILE=1`: a per-phase GPU-time split of a batched prefill,
//! by layer kind. Off by default. When it's on, each `prefill_*` entry point
//! starts a profile with [`prefill_profile_begin`], the layer code calls [`prof_mark`] at every
//! phase boundary (one timing event on the compute stream), and
//! [`PrefillProfileGuard::finish`] drains the device once at the end to read
//! the events back. The events don't serialize the stream, so the totals are
//! what the prefill costs.
//!
//! The active profile lives in a thread-local rather than a parameter, so the
//! layer functions shared by several architectures (and by decode) don't each
//! grow one: [`prof_mark`] is a no-op when no profile is active, which is always
//! the case outside a profiled prefill.
//!
//! Every phase is charged everything enqueued since the previous mark, so a
//! phase includes the small norms and adds that precede its own work (the
//! layer's input RMSNorm is in `qkv_gemm`, for example). Qwen3.5's Gated
//! DeltaNet layers run each prompt row through the mixer and the FFN in turn,
//! so their whole per-row loop is one `ssm.mixer` span.

use std::cell::RefCell;

use super::*;
use crate::error::ReflexError;

pub(super) fn prefill_profile_enabled() -> bool {
    std::env::var("REFLEX_PREFILL_PROFILE").is_ok_and(|v| v == "1")
}

/// The phase a span of GPU time is charged to: everything enqueued since the
/// previous mark.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) enum PrefillPhase {
    Embed,
    QkvGemm,
    NormRope,
    KvCopy,
    Attention,
    /// MLA's per-head `wk_b` absorption and `wv_b` decompression GEMVs.
    Absorb,
    OProj,
    /// A recurrent token mixer: Gated DeltaNet or Mamba-2.
    Mixer,
    Ffn,
    FfnTail,
}

const PHASES: [&str; 10] = [
    "embed",
    "qkv_gemm",
    "norm_rope",
    "kv_copy",
    "attn",
    "absorb",
    "o_proj",
    "mixer",
    "ffn",
    "ffn_tail",
];

/// The kind of layer a span belongs to, printed as the phase's prefix
/// (`sliding.attn_ms`). Non-layer phases (`embed`) use [`LayerKind::Layer`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) enum LayerKind {
    /// A uniform transformer layer: dense, MoE, MLA.
    Layer,
    /// Sliding-window attention (Gemma 4, Kolibri-1).
    Sliding,
    /// Full attention in a model that also has sliding layers (Gemma 4).
    Global,
    /// Full attention in a hybrid with recurrent layers (Qwen3.5, Granite 4.0-H).
    Attn,
    /// A recurrent layer (Gated DeltaNet, Mamba-2).
    Ssm,
}

const KINDS: [&str; 5] = ["layer", "sliding", "global", "attn", "ssm"];

struct PrefillProfile {
    stream: sys::CUstream,
    start: sys::CUevent,
    /// (phase, layer kind, event at the end of the span).
    marks: Vec<(PrefillPhase, LayerKind, sys::CUevent)>,
    wall: std::time::Instant,
}

impl PrefillProfile {
    fn record(stream: sys::CUstream) -> Result<sys::CUevent, ReflexError> {
        let ev = result::event::create(sys::CUevent_flags::CU_EVENT_DEFAULT)
            .map_err(|e| crate::gpu_err!(e, "create prefill profile event: {e}"))?;
        unsafe { result::event::record(ev, stream) }
            .map_err(|e| crate::gpu_err!(e, "record prefill profile event: {e}"))?;
        Ok(ev)
    }

    fn destroy_events(&self) {
        unsafe {
            let _ = result::event::destroy(self.start);
            for &(_, _, ev) in &self.marks {
                let _ = result::event::destroy(ev);
            }
        }
    }
}

thread_local! {
    static ACTIVE: RefCell<Option<PrefillProfile>> = const { RefCell::new(None) };
}

/// Ends the active profile when dropped, so a prefill that fails partway
/// doesn't leave marks to be charged to the next one.
pub(super) struct PrefillProfileGuard {
    device: Arc<CudaDevice>,
}

/// Starts a profile if `REFLEX_PREFILL_PROFILE=1` and none is active on this
/// thread yet (a prefill nested in another one reports as part of the outer).
pub(super) fn prefill_profile_begin(
    device: &Arc<CudaDevice>,
) -> Result<Option<PrefillProfileGuard>, ReflexError> {
    if !prefill_profile_enabled() || ACTIVE.with(|a| a.borrow().is_some()) {
        return Ok(None);
    }
    let stream = *device.cu_stream();
    let start = PrefillProfile::record(stream)?;
    ACTIVE.with(|a| {
        *a.borrow_mut() = Some(PrefillProfile {
            stream,
            start,
            marks: Vec::new(),
            wall: std::time::Instant::now(),
        })
    });
    Ok(Some(PrefillProfileGuard {
        device: device.clone(),
    }))
}

/// Charges everything enqueued since the last mark to `phase` on a layer of
/// kind `kind`. A no-op when no profile is active.
pub(super) fn prof_mark(phase: PrefillPhase, kind: LayerKind) -> Result<(), ReflexError> {
    ACTIVE.with(|a| match a.borrow_mut().as_mut() {
        Some(p) => {
            let ev = PrefillProfile::record(p.stream)?;
            p.marks.push((phase, kind, ev));
            Ok(())
        }
        None => Ok(()),
    })
}

impl PrefillProfileGuard {
    /// Drains the device and prints one `REFLEX_PREFILL_PROFILE` line (see
    /// [`summary_line`]).
    pub(super) fn finish(self, rows: usize) -> Result<(), ReflexError> {
        let Some(p) = ACTIVE.with(|a| a.borrow_mut().take()) else {
            return Ok(());
        };
        let enqueue_ms = p.wall.elapsed().as_secs_f64() * 1000.0;
        let synced = self
            .device
            .synchronize()
            .map_err(|e| crate::gpu_err!(e, "prefill profile: synchronize: {e}"));
        if let Err(e) = synced {
            p.destroy_events();
            return Err(e);
        }
        let wall_ms = p.wall.elapsed().as_secs_f64() * 1000.0;
        let mut prev = p.start;
        let spans: Vec<(PrefillPhase, LayerKind, f64)> = p
            .marks
            .iter()
            .map(|&(phase, kind, ev)| {
                let ms = unsafe { result::event::elapsed(prev, ev) }.unwrap_or(0.0) as f64;
                prev = ev;
                (phase, kind, ms)
            })
            .collect();
        p.destroy_events();
        eprintln!("{}", summary_line(&spans, rows, enqueue_ms, wall_ms));
        Ok(())
    }
}

impl Drop for PrefillProfileGuard {
    fn drop(&mut self) {
        if let Some(p) = ACTIVE.with(|a| a.borrow_mut().take()) {
            p.destroy_events();
        }
    }
}

/// The `REFLEX_PREFILL_PROFILE` line for `spans` (phase, layer kind, GPU ms):
/// `embed_ms`, then `<kind>.<phase>_ms` for every nonzero (kind, phase) in
/// [`KINDS`] then [`PHASES`] order, then the attention total over all kinds
/// and its share of the GPU total, the GPU total, the host enqueue time (wall
/// time until the drain) and the wall time including the drain.
fn summary_line(
    spans: &[(PrefillPhase, LayerKind, f64)],
    rows: usize,
    enqueue_ms: f64,
    wall_ms: f64,
) -> String {
    let mut table = [[0f64; PHASES.len()]; KINDS.len()];
    let mut embed = 0f64;
    let mut total = 0f64;
    for &(phase, kind, ms) in spans {
        if phase == PrefillPhase::Embed {
            embed += ms;
        } else {
            table[kind as usize][phase as usize] += ms;
        }
        total += ms;
    }
    let attn: f64 = table
        .iter()
        .map(|row| row[PrefillPhase::Attention as usize])
        .sum();
    let attn_pct = if total > 0.0 {
        100.0 * attn / total
    } else {
        0.0
    };
    let mut line = format!("REFLEX_PREFILL_PROFILE rows={rows} embed_ms={embed:.2}");
    for (kind, kind_name) in KINDS.iter().enumerate() {
        for (phase, phase_name) in PHASES.iter().enumerate() {
            if table[kind][phase] > 0.0 {
                line += &format!(" {kind_name}.{phase_name}_ms={:.2}", table[kind][phase]);
            }
        }
    }
    line += &format!(
        " attn_ms={attn:.2} attn_pct={attn_pct:.1} gpu_total_ms={total:.2} host_enqueue_ms={enqueue_ms:.2} wall_ms={wall_ms:.2}"
    );
    line
}

#[cfg(test)]
mod tests {
    use super::*;
    use LayerKind::*;
    use PrefillPhase::*;

    #[test]
    fn summary_sums_per_kind_and_phase_in_fixed_order() {
        let spans = [
            (Embed, Layer, 1.0),
            (QkvGemm, Global, 2.0),
            (Attention, Global, 3.0),
            (QkvGemm, Sliding, 4.0),
            (Attention, Sliding, 5.0),
            (Attention, Sliding, 6.0),
            (Ffn, Sliding, 7.0),
        ];
        let line = summary_line(&spans, 9, 10.0, 30.0);
        assert_eq!(
            line,
            "REFLEX_PREFILL_PROFILE rows=9 embed_ms=1.00 \
             sliding.qkv_gemm_ms=4.00 sliding.attn_ms=11.00 sliding.ffn_ms=7.00 \
             global.qkv_gemm_ms=2.00 global.attn_ms=3.00 \
             attn_ms=14.00 attn_pct=50.0 gpu_total_ms=28.00 \
             host_enqueue_ms=10.00 wall_ms=30.00"
        );
    }

    #[test]
    fn summary_covers_hybrid_and_mla_phases() {
        let spans = [
            (Mixer, Ssm, 8.0),
            (Ffn, Ssm, 0.0),
            (Attention, Attn, 1.0),
            (Absorb, Layer, 0.5),
            (FfnTail, Layer, 0.5),
        ];
        let line = summary_line(&spans, 1, 0.0, 0.0);
        assert!(line.contains(
            " layer.absorb_ms=0.50 layer.ffn_tail_ms=0.50 attn.attn_ms=1.00 ssm.mixer_ms=8.00 "
        ));
        // Zero spans print nothing; the embed field always prints.
        assert!(!line.contains("ssm.ffn_ms"));
        assert!(line.contains("embed_ms=0.00"));
        assert!(line.contains(" attn_ms=1.00 attn_pct=10.0 gpu_total_ms=10.00 "));
    }

    #[test]
    fn summary_of_no_spans_has_zero_share() {
        let line = summary_line(&[], 0, 0.0, 0.0);
        assert!(line.contains(" attn_ms=0.00 attn_pct=0.0 gpu_total_ms=0.00 "));
    }

    #[test]
    fn name_tables_match_enums() {
        assert_eq!(PHASES[FfnTail as usize], "ffn_tail");
        assert_eq!(PHASES[Absorb as usize], "absorb");
        assert_eq!(PHASES[Mixer as usize], "mixer");
        assert_eq!(KINDS[Ssm as usize], "ssm");
        assert_eq!(KINDS[Global as usize], "global");
    }

    #[test]
    fn mark_without_an_active_profile_is_a_no_op() {
        assert!(prof_mark(Attention, Layer).is_ok());
        assert!(ACTIVE.with(|a| a.borrow().is_none()));
    }
}
