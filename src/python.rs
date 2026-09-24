//! PyO3 Python bindings (`import reflex_engine` after a `maturin build
//! --features python`), feature-gated. Calls `crate::model::Model`/
//! `system1_evaluate` directly, not through `src/ffi.rs`'s C ABI -- there is no
//! reason to pay a second serialization/indirection layer when PyO3 can hold a
//! `Model` natively in the same process. See CLAUDE.md/README.md's Non-goals: this
//! is an in-process embedding surface, not a network server -- a Python process
//! holding a `PyModel` still only ever runs one request at a time (nothing here
//! spawns a thread or a request queue; the underlying `Model` methods already only
//! ever process one prompt, matching this engine's permanent `batch_size == 1`
//! constraint).

use crate::diagnostics;
use crate::gguf::GgufFile;
use crate::model::{Model, System1Candidate};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

fn to_py_err(e: String) -> PyErr {
    PyErr::new::<PyRuntimeError, _>(e)
}

/// A loaded model, held for the Python object's whole lifetime. `PyModel(path)`
/// loads a GGUF file from `path` onto GPU 0 (the only device index this engine's
/// Python/FFI/CLI surfaces ever target -- see CLAUDE.md's Non-goals).
#[pyclass]
struct PyModel {
    inner: Model,
}

#[pymethods]
impl PyModel {
    #[new]
    #[pyo3(signature = (gguf_path, lora_path=None))]
    fn load(gguf_path: &str, lora_path: Option<&str>) -> PyResult<Self> {
        let file = GgufFile::open(gguf_path).map_err(to_py_err)?;
        let device = diagnostics::init_device_with_diagnostics(0).map_err(to_py_err)?;
        let mut model = Model::load(device, &file).map_err(to_py_err)?;
        if let Some(lora_path) = lora_path {
            model
                .apply_lora(std::path::Path::new(lora_path))
                .map_err(to_py_err)?;
        }
        Ok(PyModel { inner: model })
    }

    /// Ordinary greedy decode: `(token_ids, text)`, same result shape as
    /// `Model::generate` with no imported KV cache (state import/export isn't
    /// exposed through this binding this round).
    fn generate(&self, prompt: &str, max_new_tokens: usize) -> PyResult<(Vec<u32>, String)> {
        self.inner
            .generate(prompt, max_new_tokens, None, |_logits| {})
            .map_err(to_py_err)
    }

    /// Single-pass candidate scoring (`Model::system1_evaluate`). Returns a dict:
    /// `{"results": [{"text", "token_ids", "score", "probability"}, ...], "entropy": float}`
    /// -- `entropy` is `crate::calibration::shannon_entropy` (bits) of the
    /// `probability` distribution across `results`.
    #[pyo3(signature = (prompt, candidates, temperature=1.0))]
    fn system1_evaluate(
        &self,
        py: Python<'_>,
        prompt: &str,
        candidates: Vec<String>,
        temperature: f32,
    ) -> PyResult<PyObject> {
        let candidates: Vec<System1Candidate> = candidates
            .into_iter()
            .map(|text| System1Candidate { text })
            .collect();
        let response = self
            .inner
            .system1_evaluate(prompt, &candidates, temperature)
            .map_err(to_py_err)?;

        let results = PyList::empty_bound(py);
        for (r, probability) in response.results.into_iter().zip(response.probabilities) {
            let d = PyDict::new_bound(py);
            d.set_item("text", r.text)?;
            d.set_item("token_ids", r.token_ids)?;
            d.set_item("score", r.score)?;
            d.set_item("probability", probability)?;
            results.append(d)?;
        }

        let out = PyDict::new_bound(py);
        out.set_item("results", results)?;
        out.set_item("entropy", response.entropy)?;
        Ok(out.into())
    }
}

#[pymodule]
fn reflex_engine(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyModel>()
}
