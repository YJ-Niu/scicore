//! Matrix Market I/O CPython extension.
//!
//! Rust replacement for the C++ `_fmm_core` extension; backs
//! `scicore.io.mmread` / `mmwrite` / `mminfo`.

use std::path::PathBuf;

use pyo3::exceptions::{PyFileNotFoundError, PyOSError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyComplex, PyList, PyTuple};

mod format;
mod parse;

use parse::{Val, is_stored, parse_array_body, parse_coo_body, parse_header};

const VERSION: &str = "1.2.4";

// ---------------------------------------------------------------------------
// Value -> Python object
// ---------------------------------------------------------------------------

impl Val {
    #[allow(clippy::wrong_self_convention)]
    fn to_py(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        Ok(match self {
            Val::Real(x) => (*x).into_pyobject(py)?.into_any().unbind(),
            Val::Int(x) => (*x).into_pyobject(py)?.into_any().unbind(),
            Val::Uint(x) => (*x).into_pyobject(py)?.into_any().unbind(),
            Val::Complex { re, im } => PyComplex::from_doubles(py, *re, *im).into_any().unbind(),
        })
    }
}

// ---------------------------------------------------------------------------
// Header
// ---------------------------------------------------------------------------

#[pyclass]
struct PyHeader {
    comment: String,
    symmetry: String,
    format: String,
    field: String,
    nrows: i64,
    ncols: i64,
    nnz: i64,
}

#[pymethods]
impl PyHeader {
    #[getter]
    fn comment(&self) -> &str {
        &self.comment
    }
    #[getter]
    fn symmetry(&self) -> &str {
        &self.symmetry
    }
    #[getter]
    fn format(&self) -> &str {
        &self.format
    }
    #[getter]
    fn field(&self) -> &str {
        &self.field
    }
    #[getter]
    fn nrows(&self) -> i64 {
        self.nrows
    }
    #[getter]
    fn ncols(&self) -> i64 {
        self.ncols
    }
    #[getter]
    fn nnz(&self) -> i64 {
        self.nnz
    }
    #[getter]
    fn shape(&self) -> (i64, i64) {
        (self.nrows, self.ncols)
    }
}

#[pyfunction]
fn header(comment: Option<String>, symmetry: &str) -> PyResult<PyHeader> {
    Ok(PyHeader {
        comment: comment.unwrap_or_default(),
        symmetry: symmetry.to_string(),
        format: String::new(),
        field: String::new(),
        nrows: 0,
        ncols: 0,
        nnz: 0,
    })
}

// ---------------------------------------------------------------------------
// Read cursor
// ---------------------------------------------------------------------------

#[pyclass]
struct ReadCursor {
    format: String,
    field: String,
    symmetry: String,
    nrows: i64,
    ncols: i64,
    nnz: i64,
    comment: String,
    body: String,
}

#[pymethods]
impl ReadCursor {
    #[getter]
    fn header(&self, py: Python<'_>) -> PyResult<Py<PyHeader>> {
        Py::new(
            py,
            PyHeader {
                comment: self.comment.clone(),
                symmetry: self.symmetry.clone(),
                format: self.format.clone(),
                field: self.field.clone(),
                nrows: self.nrows,
                ncols: self.ncols,
                nnz: self.nnz,
            },
        )
    }

    fn close(&mut self) {
        self.body.clear();
    }
}

fn make_read_cursor(text: &str) -> PyResult<ReadCursor> {
    let (h, body) = parse_header(text).map_err(PyValueError::new_err)?;
    Ok(ReadCursor {
        format: h.format,
        field: h.field,
        symmetry: h.symmetry,
        nrows: h.nrows,
        ncols: h.ncols,
        nnz: h.nnz,
        comment: h.comment,
        body,
    })
}

#[pyfunction]
fn open_read_file(path: &str, _parallelism: Option<i64>) -> PyResult<ReadCursor> {
    let text = match std::fs::read_to_string(path) {
        Ok(t) => t,
        Err(e) => {
            let msg = format!("{path}: {e}");
            return if e.kind() == std::io::ErrorKind::NotFound {
                Err(PyFileNotFoundError::new_err(msg))
            } else {
                Err(PyOSError::new_err(msg))
            };
        }
    };
    make_read_cursor(&text)
}

#[pyfunction]
fn open_read_stream(source: &Bound<'_, PyAny>, _parallelism: Option<i64>) -> PyResult<ReadCursor> {
    let read = source.getattr("read")?;
    let mut chunks: Vec<u8> = Vec::new();
    loop {
        let chunk = read.call((65536i64,), None)?;
        let b: Vec<u8> = chunk.extract()?;
        if b.is_empty() {
            break;
        }
        chunks.extend_from_slice(&b);
    }
    let text = String::from_utf8_lossy(&chunks).into_owned();
    make_read_cursor(&text)
}

// ---------------------------------------------------------------------------
// Read body
// ---------------------------------------------------------------------------

#[pyfunction]
fn read_body_array(cursor: &ReadCursor, py: Python<'_>) -> PyResult<Py<PyList>> {
    let mat = parse_array_body(
        &cursor.body,
        &cursor.field,
        &cursor.symmetry,
        cursor.nrows,
        cursor.ncols,
    )
    .map_err(PyValueError::new_err)?;
    let mut objs: Vec<Py<PyAny>> = Vec::with_capacity(mat.len());
    for v in mat.iter() {
        objs.push(v.to_py(py)?);
    }
    let list = PyList::new(py, objs)?;
    Ok(list.into())
}

#[pyfunction]
fn read_body_coo(cursor: &ReadCursor, py: Python<'_>) -> PyResult<Py<PyTuple>> {
    let (rows, cols, data) =
        parse_coo_body(&cursor.body, &cursor.field, cursor.nnz).map_err(PyValueError::new_err)?;
    let mut i_objs: Vec<Py<PyAny>> = Vec::with_capacity(rows.len());
    for &r in rows.iter() {
        i_objs.push(r.into_pyobject(py)?.into_any().unbind());
    }
    let mut j_objs: Vec<Py<PyAny>> = Vec::with_capacity(cols.len());
    for &c in cols.iter() {
        j_objs.push(c.into_pyobject(py)?.into_any().unbind());
    }
    let mut d_objs: Vec<Py<PyAny>> = Vec::with_capacity(data.len());
    for v in data.iter() {
        d_objs.push(v.to_py(py)?);
    }
    let i_list = PyList::new(py, i_objs)?.into_any().unbind();
    let j_list = PyList::new(py, j_objs)?.into_any().unbind();
    let d_list = PyList::new(py, d_objs)?.into_any().unbind();
    Ok(PyTuple::new(py, [i_list, j_list, d_list])?.into())
}

// ---------------------------------------------------------------------------
// Write cursor
// ---------------------------------------------------------------------------

enum OutTarget {
    File(PathBuf),
    Stream(Py<PyAny>),
}

#[pyclass]
struct WriteCursor {
    target: OutTarget,
    comment: String,
    symmetry: String,
    precision: i32,
    _written: bool,
}

#[pymethods]
impl WriteCursor {
    fn close(&mut self, py: Python<'_>) -> PyResult<()> {
        if let OutTarget::Stream(s) = &self.target {
            let _ = s.getattr(py, "flush").and_then(|f| f.call0(py));
        }
        Ok(())
    }
}

#[pyfunction]
fn open_write_file(
    target: &str,
    h: &PyHeader,
    _parallelism: Option<i64>,
    precision: Option<i32>,
) -> PyResult<WriteCursor> {
    Ok(WriteCursor {
        target: OutTarget::File(PathBuf::from(target)),
        comment: h.comment.clone(),
        symmetry: h.symmetry.clone(),
        precision: precision.unwrap_or(-1),
        _written: false,
    })
}

#[pyfunction]
fn open_write_stream(
    target: &Bound<'_, PyAny>,
    h: &PyHeader,
    _parallelism: Option<i64>,
    precision: Option<i32>,
) -> PyResult<WriteCursor> {
    Ok(WriteCursor {
        target: OutTarget::Stream(target.clone().unbind()),
        comment: h.comment.clone(),
        symmetry: h.symmetry.clone(),
        precision: precision.unwrap_or(-1),
        _written: false,
    })
}

/// Format the comment block. An empty comment yields a single `"\n"` that
/// becomes a no-op when the lines are joined; each non-empty comment line is
/// prefixed with `%`.
fn comment_block(comment: &str) -> String {
    comment
        .split('\n')
        .map(|line| format!("%{line}"))
        .collect::<Vec<_>>()
        .join("\n")
}

fn format_py_value(v: &Bound<'_, PyAny>, field: &str, precision: i32) -> PyResult<String> {
    match field {
        "real" | "double" => {
            let f: f64 = v.clone().extract()?;
            Ok(format::format_value(f, precision))
        }
        "integer" => {
            let i: i64 = v.clone().extract()?;
            Ok(i.to_string())
        }
        "unsigned-integer" => {
            let u: u64 = v.clone().extract()?;
            Ok(u.to_string())
        }
        "complex" => {
            let re: f64 = v.getattr("real")?.extract()?;
            let im: f64 = v.getattr("imag")?.extract()?;
            Ok(format::format_complex(re, im, precision))
        }
        "pattern" => Ok(String::new()),
        other => Err(PyValueError::new_err(format!(
            "unknown field type: {other}"
        ))),
    }
}

fn write_output(target: &mut OutTarget, lines: &[String], py: Python<'_>) -> PyResult<()> {
    let out = lines.join("\n");
    match target {
        OutTarget::File(path) => std::fs::write(&*path, out.as_bytes())
            .map_err(|e| PyOSError::new_err(format!("{}: {e}", path.display()))),
        OutTarget::Stream(stream) => {
            let bytes: Py<PyAny> = PyBytes::new(py, out.as_bytes()).into_any().unbind();
            stream.call_method1(py, "write", (bytes,))?;
            Ok(())
        }
    }
}

// ---------------------------------------------------------------------------
// Write body
// ---------------------------------------------------------------------------

#[pyfunction]
fn write_body_array(
    cursor: &mut WriteCursor,
    field: &str,
    shape: &Bound<'_, PyTuple>,
    values: &Bound<'_, PyList>,
    py: Python<'_>,
) -> PyResult<()> {
    let nrows: i64 = shape.get_item(0)?.extract()?;
    let ncols: i64 = shape.get_item(1)?.extract()?;
    if values.len() != (nrows * ncols) as usize {
        return Err(PyValueError::new_err(
            "values length does not match the array shape",
        ));
    }
    let symmetry = cursor.symmetry.clone();
    let mut lines: Vec<String> = Vec::new();
    lines.push(format!("%%MatrixMarket matrix array {field} {symmetry}"));
    lines.push(comment_block(&cursor.comment));
    lines.push(format!("{nrows} {ncols}"));
    // Matrix Market array format stores values in column-major order.
    for c in 0..ncols {
        for r in 0..nrows {
            if !is_stored(r, c, &symmetry) {
                continue;
            }
            // `values` is the flat row-major (C-order) list from `a.flatten()`.
            let v = values.get_item((r * ncols + c) as usize)?;
            lines.push(format_py_value(&v, field, cursor.precision)?);
        }
    }
    write_output(&mut cursor.target, &lines, py)?;
    cursor._written = true;
    Ok(())
}

#[pyfunction]
fn write_body_coo(
    cursor: &mut WriteCursor,
    field: &str,
    shape: &Bound<'_, PyTuple>,
    row: &Bound<'_, PyList>,
    col: &Bound<'_, PyList>,
    data: &Bound<'_, PyList>,
    py: Python<'_>,
) -> PyResult<()> {
    let nrows: i64 = shape.get_item(0)?.extract()?;
    let ncols: i64 = shape.get_item(1)?.extract()?;
    let nnz = row.len();
    if col.len() != nnz {
        return Err(PyValueError::new_err("len(row) must equal len(col)"));
    }
    if field != "pattern" && nnz != 0 && data.len() != nnz {
        return Err(PyValueError::new_err("len(row) must equal len(data)"));
    }
    let symmetry = cursor.symmetry.clone();
    let mut lines: Vec<String> = Vec::new();
    lines.push(format!(
        "%%MatrixMarket matrix coordinate {field} {symmetry}"
    ));
    lines.push(comment_block(&cursor.comment));
    lines.push(format!("{nrows} {ncols} {nnz}"));
    for k in 0..nnz {
        let r: i64 = row.get_item(k)?.extract()?;
        let c: i64 = col.get_item(k)?.extract()?;
        if field == "pattern" {
            lines.push(format!("{} {}", r + 1, c + 1));
        } else {
            let v = data.get_item(k)?;
            lines.push(format!(
                "{} {} {}",
                r + 1,
                c + 1,
                format_py_value(&v, field, cursor.precision)?
            ));
        }
    }
    write_output(&mut cursor.target, &lines, py)?;
    cursor._written = true;
    Ok(())
}

// ---------------------------------------------------------------------------
// Module
// ---------------------------------------------------------------------------

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", VERSION)?;
    m.add_class::<ReadCursor>()?;
    m.add_class::<WriteCursor>()?;
    m.add_class::<PyHeader>()?;
    m.add_function(wrap_pyfunction!(header, m)?)?;
    m.add_function(wrap_pyfunction!(open_read_file, m)?)?;
    m.add_function(wrap_pyfunction!(open_read_stream, m)?)?;
    m.add_function(wrap_pyfunction!(open_write_file, m)?)?;
    m.add_function(wrap_pyfunction!(open_write_stream, m)?)?;
    m.add_function(wrap_pyfunction!(read_body_array, m)?)?;
    m.add_function(wrap_pyfunction!(read_body_coo, m)?)?;
    m.add_function(wrap_pyfunction!(write_body_array, m)?)?;
    m.add_function(wrap_pyfunction!(write_body_coo, m)?)?;
    Ok(())
}
