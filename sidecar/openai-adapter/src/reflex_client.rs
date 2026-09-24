//! Owns exactly one managed `reflex stdio <gguf>` child process and serializes every
//! HTTP-originated request into it one at a time, matching the core engine's
//! permanent `batch_size == 1`/strictly-sequential contract (see root CLAUDE.md/
//! README.md's Non-goals) even though this sidecar's HTTP side happily accepts
//! concurrent connections. A single background worker task is the only thing that
//! ever touches the child's stdin/stdout; every request handler talks to it through
//! an mpsc queue instead, so two racing HTTP requests can never interleave writes to
//! the same IPC connection -- the worker doesn't dequeue the next job until it has
//! read the previous job's `"event": "final"` line.

use anyhow::{anyhow, Context, Result};
use serde_json::Value;
use std::process::Stdio;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, Command};
use tokio::sync::mpsc;

struct Job {
    line: String,
    events_tx: mpsc::Sender<Value>,
}

/// Handle to the managed `reflex stdio` process. Cloning is cheap (just clones the
/// job-queue sender) since the actual child process and its I/O are owned by the
/// background worker task, not by any individual handle.
#[derive(Clone)]
pub struct ReflexClient {
    job_tx: mpsc::Sender<Job>,
}

impl ReflexClient {
    /// Spawns `<reflex_bin> stdio <gguf_path> [--lora <lora_path>]` and a background
    /// worker task that owns its stdin/stdout for the lifetime of the process. The
    /// child's stderr (its `REFLEX_STDIO_READY`/diagnostic lines) is forwarded to
    /// this process's own stderr, prefixed, so it's visible in the sidecar's logs.
    pub async fn spawn(reflex_bin: &str, gguf_path: &str, lora_path: Option<&str>) -> Result<Self> {
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

        // Forward the child's stderr line-by-line so `REFLEX_STDIO_READY`/model-load
        // diagnostics/panics are still visible, just prefixed to distinguish them
        // from the adapter's own log lines.
        tokio::spawn(async move {
            let mut lines = BufReader::new(stderr).lines();
            loop {
                match lines.next_line().await {
                    Ok(Some(line)) => eprintln!("[reflex] {line}"),
                    Ok(None) => break, // child closed stderr (process exited)
                    Err(e) => {
                        eprintln!("[reflex] (stderr read error: {e})");
                        break;
                    }
                }
            }
        });

        // Leaking the `Child` handle into this task (instead of holding it in
        // `ReflexClient`) is deliberate: `kill_on_drop(true)` above means the process
        // is torn down when this task's `Child` is dropped, and keeping it alive for
        // the process lifetime here (via `let _child = child;` at the end of an
        // otherwise-forever task) is simpler than threading a second owner through
        // `ReflexClient`. If the child exits, `child.wait()` returns and we log it;
        // in-flight/future jobs then fail via the stdin/stdout-closed error paths in
        // `run_worker` below.
        let (job_tx, job_rx) = mpsc::channel::<Job>(256);
        tokio::spawn(run_worker(stdin, stdout, job_rx));
        tokio::spawn(async move {
            match child.wait().await {
                Ok(status) => eprintln!("[adapter] reflex process exited: {status}"),
                Err(e) => eprintln!("[adapter] reflex process wait() failed: {e}"),
            }
        });

        Ok(ReflexClient { job_tx })
    }

    /// Enqueues one already-serialized IPC request line and returns a channel that
    /// yields every JSON response line the worker reads back for it -- one
    /// [`Value`] per `"event": "token"` line (if the request was `"stream": true`),
    /// followed by exactly one `"event": "final"` line, after which the channel is
    /// closed. A non-streaming request's channel yields just the one `"final"` line.
    pub async fn request(&self, line: String) -> mpsc::Receiver<Value> {
        let (events_tx, events_rx) = mpsc::channel(64);
        if self
            .job_tx
            .send(Job {
                line,
                events_tx: events_tx.clone(),
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
        events_rx
    }
}

/// The single task that ever touches the child's stdin/stdout. Processes exactly one
/// [`Job`] at a time: writes its request line, then reads response lines until it
/// sees `"event": "final"` (or the child closes stdout / a line fails to parse),
/// before dequeuing the next job -- this loop, not any locking, is what keeps two
/// concurrent HTTP requests from ever interleaving on the underlying IPC connection.
async fn run_worker(
    mut stdin: tokio::process::ChildStdin,
    mut stdout: BufReader<tokio::process::ChildStdout>,
    mut job_rx: mpsc::Receiver<Job>,
) {
    while let Some(job) = job_rx.recv().await {
        if let Err(e) = write_request(&mut stdin, &job.line).await {
            let _ = job.events_tx.send(worker_error(&e.to_string())).await;
            continue;
        }

        loop {
            let mut line = String::new();
            match stdout.read_line(&mut line).await {
                Ok(0) => {
                    // EOF: the reflex process closed stdout (crashed or exited).
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
                            let _ = job
                                .events_tx
                                .send(worker_error(&format!("malformed reflex output line: {e}")))
                                .await;
                            break;
                        }
                    }
                }
                Err(e) => {
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

async fn write_request(stdin: &mut tokio::process::ChildStdin, line: &str) -> Result<()> {
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
