//! Owns exactly one managed `reflex stdio <gguf>` child process and serializes every
//! HTTP-originated request into it one at a time, matching the core engine's
//! permanent `batch_size == 1`/strictly-sequential contract (see root CLAUDE.md/
//! README.md's Non-goals) even though this sidecar's HTTP side happily accepts
//! concurrent connections. A single background worker task is the only thing that
//! ever touches the child's stdin/stdout; every request handler talks to it through
//! an mpsc queue instead, so two racing HTTP requests can never interleave writes to
//! the same IPC connection -- the worker doesn't dequeue the next job until it has
//! read the previous job's `"event": "final"` line.
//!
//! Because jobs run strictly one at a time, the queue is also the sidecar's only
//! backpressure point: [`ReflexClient::request`] refuses a job once
//! `max_in_flight` jobs are already queued or running (see [`InFlightPermit`]).

use anyhow::{anyhow, Context, Result};
use serde_json::Value;
use std::process::Stdio;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use tokio::io::{AsyncBufRead, AsyncBufReadExt, AsyncWrite, AsyncWriteExt, BufReader};
use tokio::process::{Child, Command};
use tokio::sync::mpsc;

struct Job {
    line: String,
    events_tx: mpsc::Sender<Value>,
    /// Held until the worker is completely done with this job (its `final` line
    /// drained, or the job skipped), so the in-flight count covers engine work, not
    /// just open HTTP responses: a client that timed out or disconnected frees its
    /// connection, but the engine has no cancel operation and is still busy.
    _permit: InFlightPermit,
}

/// One slot of the `max_in_flight` budget; releases it on drop.
struct InFlightPermit(Arc<AtomicUsize>);

impl InFlightPermit {
    fn try_acquire(counter: &Arc<AtomicUsize>, max: usize) -> Option<Self> {
        counter
            .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |n| {
                (n < max).then_some(n + 1)
            })
            .ok()
            .map(|_| InFlightPermit(Arc::clone(counter)))
    }
}

impl Drop for InFlightPermit {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::SeqCst);
    }
}

/// [`ReflexClient::request`] refused a job: `max_in_flight` jobs are already queued
/// or running.
#[derive(Debug)]
pub struct QueueFull {
    pub max_in_flight: usize,
}

/// Handle to the managed `reflex stdio` process. Cloning is cheap (just clones the
/// job-queue sender and the shared liveness flag) since the actual child process and
/// its I/O are owned by the background worker task, not by any individual handle.
#[derive(Clone)]
pub struct ReflexClient {
    job_tx: mpsc::Sender<Job>,
    child_alive: Arc<AtomicBool>,
    ready: Arc<AtomicBool>,
    in_flight: Arc<AtomicUsize>,
    max_in_flight: usize,
}

impl ReflexClient {
    /// Spawns `<reflex_bin> stdio <gguf_path> [--lora <lora_path>]` and a background
    /// worker task that owns its stdin/stdout for the lifetime of the process. The
    /// child's stderr (its `REFLEX_STDIO_READY`/diagnostic lines) is forwarded to
    /// this process's own stderr, prefixed, so it's visible in the sidecar's logs.
    /// `max_in_flight` (at least 1) caps queued-plus-running jobs -- see
    /// [`Self::request`].
    pub async fn spawn(
        reflex_bin: &str,
        gguf_path: &str,
        lora_path: Option<&str>,
        max_in_flight: usize,
    ) -> Result<Self> {
        let mut cmd = Command::new(reflex_bin);
        cmd.arg("stdio").arg(gguf_path);
        if let Some(lora) = lora_path {
            cmd.arg("--lora").arg(lora);
        }
        cmd.stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true);

        let mut child: Child = cmd
            .spawn()
            .with_context(|| format!("spawning `{reflex_bin} stdio {gguf_path}`"))?;

        let stdin = child.stdin.take().expect("piped stdin");
        let stdout = BufReader::new(child.stdout.take().expect("piped stdout"));
        let stderr = child.stderr.take().expect("piped stderr");

        let ready = Arc::new(AtomicBool::new(false));

        // Forward the child's stderr line-by-line so `REFLEX_STDIO_READY`/model-load
        // diagnostics/panics are still visible, just prefixed to distinguish them
        // from the adapter's own log lines. Also watches for the `REFLEX_STDIO_READY`
        // line specifically (printed by `src/bin/reflex/stdio.rs` only after
        // `Model::load` succeeds, right before it starts its stdin/stdout request
        // loop) to flip `ready` -- this is what lets `/healthz` distinguish "process
        // spawned but still loading the model" from "actually able to serve a
        // request," which matters for any caller (e.g. a RunPod Serverless
        // load-balancing endpoint's health check) that measures cold-start duration
        // from this signal.
        {
            let ready = Arc::clone(&ready);
            tokio::spawn(async move {
                let mut lines = BufReader::new(stderr).lines();
                loop {
                    match lines.next_line().await {
                        Ok(Some(line)) => {
                            if line.starts_with("REFLEX_STDIO_READY") {
                                ready.store(true, Ordering::SeqCst);
                            }
                            eprintln!("[reflex] {line}");
                        }
                        Ok(None) => break, // child closed stderr (process exited)
                        Err(e) => {
                            eprintln!("[reflex] (stderr read error: {e})");
                            break;
                        }
                    }
                }
            });
        }

        // Leaking the `Child` handle into this task (instead of holding it in
        // `ReflexClient`) is deliberate: `kill_on_drop(true)` above means the process
        // is torn down when this task's `Child` is dropped, and keeping it alive for
        // the process lifetime here (via `let _child = child;` at the end of an
        // otherwise-forever task) is simpler than threading a second owner through
        // `ReflexClient`. If the child exits, `child.wait()` returns and we log it;
        // in-flight/future jobs then fail via the stdin/stdout-closed error paths in
        // `run_worker` below.
        let child_alive = Arc::new(AtomicBool::new(true));
        // Capacity matches the in-flight cap, so `request`'s `send` never waits: a
        // job only reaches the channel after winning one of `max_in_flight` permits.
        let (job_tx, job_rx) = mpsc::channel::<Job>(max_in_flight.max(1));
        tokio::spawn(run_worker(stdin, stdout, job_rx, Arc::clone(&child_alive)));
        // `child.wait()` resolving is the authoritative "the process is gone" signal
        // -- it fires the moment the process exits, regardless of whether any HTTP
        // request has ever been sent (the worker's own EOF/error detection above only
        // ever runs while a job is in flight, so relying on it alone would leave
        // `child_alive` stuck at `true` for an instance that died before its first
        // request -- confirmed as a real gap via a real `docker run` where a `reflex`
        // panic at startup left `/healthz` reporting "ok" indefinitely).
        {
            let child_alive = Arc::clone(&child_alive);
            tokio::spawn(async move {
                match child.wait().await {
                    Ok(status) => eprintln!("[adapter] reflex process exited: {status}"),
                    Err(e) => eprintln!("[adapter] reflex process wait() failed: {e}"),
                }
                child_alive.store(false, Ordering::SeqCst);
            });
        }

        Ok(ReflexClient {
            job_tx,
            child_alive,
            ready,
            in_flight: Arc::new(AtomicUsize::new(0)),
            max_in_flight: max_in_flight.max(1),
        })
    }

    /// `false` once the worker has observed the managed `reflex` process's stdio
    /// close or error -- see `run_worker`'s EOF/read-error branches below. Backs
    /// `main.rs`'s `/healthz` handler; this type never tries to respawn the child
    /// itself (see this module's doc comment and the crate README's "Known
    /// limitations" -- recovery is the outer process supervisor's job).
    pub fn is_child_alive(&self) -> bool {
        self.child_alive.load(Ordering::SeqCst)
    }

    /// `true` once the child has printed `REFLEX_STDIO_READY` (model loaded, about to
    /// enter its request loop) -- `false` while it's still spawning/loading. Distinct
    /// from `is_child_alive`: a freshly-spawned child is alive but not yet ready.
    pub fn is_ready(&self) -> bool {
        self.ready.load(Ordering::SeqCst)
    }

    /// Enqueues one already-serialized IPC request line and returns a channel that
    /// yields every JSON response line the worker reads back for it -- one
    /// [`Value`] per `"event": "token"` line (if the request was `"stream": true`),
    /// followed by exactly one `"event": "final"` line, after which the channel is
    /// closed. A non-streaming request's channel yields just the one `"final"` line.
    ///
    /// Errs with [`QueueFull`] without enqueuing anything when `max_in_flight` jobs
    /// are already queued or running. Dropping the returned receiver early (client
    /// timeout/disconnect) is safe: the worker still drains the job's output so the
    /// IPC stream stays in sync, or skips the job outright if it hadn't started.
    pub async fn request(&self, line: String) -> Result<mpsc::Receiver<Value>, QueueFull> {
        let permit =
            InFlightPermit::try_acquire(&self.in_flight, self.max_in_flight).ok_or(QueueFull {
                max_in_flight: self.max_in_flight,
            })?;
        let (events_tx, events_rx) = mpsc::channel(64);
        if self
            .job_tx
            .send(Job {
                line,
                events_tx: events_tx.clone(),
                _permit: permit,
            })
            .await
            .is_err()
        {
            let _ = events_tx
                .send(serde_json::json!({
                    "event": "final",
                    "ok": false,
                    "error": "sidecar: reflex worker task is no longer running",
                }))
                .await;
        }
        Ok(events_rx)
    }
}

/// The single task that ever touches the child's stdin/stdout. Processes exactly one
/// [`Job`] at a time: writes its request line, then reads response lines until it
/// sees `"event": "final"` (or the child closes stdout / a line fails to parse),
/// before dequeuing the next job -- this loop, not any locking, is what keeps two
/// concurrent HTTP requests from ever interleaving on the underlying IPC connection.
/// Generic over the pipe types only so tests can drive it with in-memory streams.
async fn run_worker<W, R>(
    mut stdin: W,
    mut stdout: R,
    mut job_rx: mpsc::Receiver<Job>,
    child_alive: Arc<AtomicBool>,
) where
    W: AsyncWrite + Unpin,
    R: AsyncBufRead + Unpin,
{
    while let Some(job) = job_rx.recv().await {
        if job.events_tx.is_closed() {
            // The client gave up (timeout/disconnect) while this job was still
            // queued. Nothing has been written to reflex yet, so skipping it keeps
            // the IPC stream in sync and saves the engine a job nobody will read.
            continue;
        }
        if let Err(e) = write_request(&mut stdin, &job.line).await {
            // A broken stdin pipe means the child is gone -- mark it dead so
            // `/healthz` reflects reality instead of silently returning "ok"
            // for every request from here on.
            child_alive.store(false, Ordering::SeqCst);
            let _ = job.events_tx.send(worker_error(&e.to_string())).await;
            continue;
        }

        loop {
            let mut line = String::new();
            match stdout.read_line(&mut line).await {
                Ok(0) => {
                    // EOF: the reflex process closed stdout (crashed or exited).
                    child_alive.store(false, Ordering::SeqCst);
                    let _ = job
                        .events_tx
                        .send(worker_error(
                            "reflex process closed its stdout (crashed or exited)",
                        ))
                        .await;
                    break;
                }
                Ok(_) => {
                    let trimmed = line.trim();
                    if trimmed.is_empty() {
                        continue;
                    }
                    match serde_json::from_str::<Value>(trimmed) {
                        Ok(value) => {
                            let is_final =
                                value.get("event").and_then(Value::as_str) == Some("final");
                            // Receiver may have dropped (client disconnected mid-stream);
                            // ignore the send error and keep draining reflex's output so
                            // the next job still starts from a clean stream position.
                            let _ = job.events_tx.send(value).await;
                            if is_final {
                                break;
                            }
                        }
                        Err(e) => {
                            // Malformed output is a protocol-level surprise, not
                            // evidence the process itself died -- don't flip
                            // `child_alive` here.
                            let _ = job
                                .events_tx
                                .send(worker_error(&format!("malformed reflex output line: {e}")))
                                .await;
                            break;
                        }
                    }
                }
                Err(e) => {
                    // A read error on the pipe (not a clean EOF) is also treated as
                    // the child being gone -- both are "we can no longer talk to it."
                    child_alive.store(false, Ordering::SeqCst);
                    let _ = job
                        .events_tx
                        .send(worker_error(&format!("reading reflex stdout: {e}")))
                        .await;
                    break;
                }
            }
        }
    }
}

async fn write_request<W: AsyncWrite + Unpin>(stdin: &mut W, line: &str) -> Result<()> {
    stdin
        .write_all(line.as_bytes())
        .await
        .map_err(|e| anyhow!("writing to reflex stdin: {e}"))?;
    stdin
        .write_all(b"\n")
        .await
        .map_err(|e| anyhow!("writing to reflex stdin: {e}"))?;
    stdin
        .flush()
        .await
        .map_err(|e| anyhow!("flushing reflex stdin: {e}"))
}

fn worker_error(message: &str) -> Value {
    serde_json::json!({
        "event": "final",
        "ok": false,
        "error": format!("sidecar: {message}"),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::io::duplex;

    fn job(line: &str, permit_counter: &Arc<AtomicUsize>) -> (Job, mpsc::Receiver<Value>) {
        let (events_tx, events_rx) = mpsc::channel(64);
        let permit = InFlightPermit::try_acquire(permit_counter, usize::MAX).unwrap();
        (
            Job {
                line: line.to_string(),
                events_tx,
                _permit: permit,
            },
            events_rx,
        )
    }

    #[test]
    fn permits_cap_in_flight_and_release_on_drop() {
        let counter = Arc::new(AtomicUsize::new(0));
        let a = InFlightPermit::try_acquire(&counter, 2).unwrap();
        let _b = InFlightPermit::try_acquire(&counter, 2).unwrap();
        assert!(InFlightPermit::try_acquire(&counter, 2).is_none());
        drop(a);
        assert_eq!(counter.load(Ordering::SeqCst), 1);
        assert!(InFlightPermit::try_acquire(&counter, 2).is_some());
    }

    /// A client that drops its receiver mid-job (the HTTP timeout path) must not
    /// desynchronize the IPC stream: the worker drains that job's remaining lines,
    /// so the next job reads its own `final`, not the abandoned job's leftovers.
    /// Also checks that a job abandoned while still queued is never sent to reflex.
    #[tokio::test]
    async fn dropped_receiver_is_drained_and_next_job_stays_in_sync() {
        // Fake `reflex stdio`: the worker gets one end of two in-memory pipes and
        // this test plays the child process on the other ends.
        let (worker_stdin, child_stdin) = duplex(4096);
        let (mut child_stdout, worker_stdout) = duplex(4096);
        let counter = Arc::new(AtomicUsize::new(0));
        let (job_tx, job_rx) = mpsc::channel(8);
        let alive = Arc::new(AtomicBool::new(true));

        let (job1, rx1) = job("request-1", &counter);
        let (job2, rx2) = job("request-2-abandoned-in-queue", &counter);
        let (job3, mut rx3) = job("request-3", &counter);
        drop(rx2);
        job_tx.send(job1).await.unwrap();
        job_tx.send(job2).await.unwrap();
        job_tx.send(job3).await.unwrap();
        drop(job_tx);

        let worker = tokio::spawn(run_worker(
            worker_stdin,
            BufReader::new(worker_stdout),
            job_rx,
            Arc::clone(&alive),
        ));
        let mut child_in = BufReader::new(child_stdin).lines();

        // Job 1: the client disconnects before any output arrives.
        assert_eq!(child_in.next_line().await.unwrap().unwrap(), "request-1");
        drop(rx1);
        for tok in ["a", "b", "c"] {
            let line = format!("{{\"event\":\"token\",\"text\":\"{tok}\"}}\n");
            child_stdout.write_all(line.as_bytes()).await.unwrap();
        }
        child_stdout
            .write_all(b"{\"event\":\"final\",\"ok\":true,\"text\":\"abc\"}\n")
            .await
            .unwrap();

        // Job 2 was abandoned while queued, so the next line reflex sees is job 3.
        assert_eq!(child_in.next_line().await.unwrap().unwrap(), "request-3");
        child_stdout
            .write_all(b"{\"event\":\"final\",\"ok\":true,\"text\":\"three\"}\n")
            .await
            .unwrap();

        let v = rx3.recv().await.unwrap();
        assert_eq!(v["text"], "three");
        assert!(rx3.recv().await.is_none());

        worker.await.unwrap();
        assert!(alive.load(Ordering::SeqCst));
        assert_eq!(counter.load(Ordering::SeqCst), 0, "every permit released");
    }
}
