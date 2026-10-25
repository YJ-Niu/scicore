//! Rust port of the PyCapsule helpers from `scipy/_lib/_ccallback_c.pyx`.
//!
//! `scipy._lib._ccallback.LowLevelCallable` only needs `get_raw_capsule`,
//! `get_capsule_signature` and `check_capsule`; those are provided here.
//! The `test_*` / `plus1_*` helpers of the original module exist solely for
//! `scipy/_lib/tests/test_ccallback.py` and are not part of the port.

use std::ffi::{CStr, CString, c_char, c_double, c_int, c_long, c_short, c_void};

use pyo3::exceptions::PyValueError;
use pyo3::ffi;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyList, PyTuple};

fn is_capsule(obj: &Bound<'_, PyAny>) -> bool {
    unsafe { ffi::PyCapsule_CheckExact(obj.as_ptr()) != 0 }
}

/// Frees the `strdup`-style name we hand to `PyCapsule_New`.
extern "C" fn raw_capsule_destructor(capsule: *mut ffi::PyObject) {
    unsafe {
        let name = ffi::PyCapsule_GetName(capsule);
        if !name.is_null() {
            // the pointer came from `CString::into_raw`, so give it back to Rust
            drop(CString::from_raw(name as *mut c_char));
        }
    }
}

fn parse_name(name_obj: &Bound<'_, PyAny>) -> PyResult<Option<CString>> {
    if name_obj.is_none() {
        return Ok(None);
    }
    let bytes: Vec<u8> = if let Ok(b) = name_obj.cast::<PyBytes>() {
        b.as_bytes().to_vec()
    } else {
        let s: String = name_obj.extract()?;
        if !s.is_ascii() {
            return Err(PyValueError::new_err("signature must be ASCII"));
        }
        s.into_bytes()
    };
    Ok(Some(CString::new(bytes).map_err(|_| {
        PyValueError::new_err("signature must not contain a NUL byte")
    })?))
}

fn capsule_name(obj: &Bound<'_, PyAny>) -> Option<CString> {
    unsafe {
        let raw = ffi::PyCapsule_GetName(obj.as_ptr());
        if raw.is_null() {
            ffi::PyErr_Clear();
            return None;
        }
        Some(CStr::from_ptr(raw).to_owned())
    }
}

/// get_raw_capsule(ptr, name, context)
///
/// Create a new PyCapsule with the given pointer, name and context.
#[pyfunction]
#[pyo3(signature = (func_obj, name_obj, context_obj))]
fn get_raw_capsule<'py>(
    py: Python<'py>,
    func_obj: &Bound<'py, PyAny>,
    name_obj: &Bound<'py, PyAny>,
    context_obj: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    // ---- context -------------------------------------------------------
    let mut context: *mut c_void = std::ptr::null_mut();
    if is_capsule(context_obj) {
        let cname = capsule_name(context_obj);
        let raw = cname
            .as_ref()
            .map(|c| c.as_ptr())
            .unwrap_or(std::ptr::null());
        context = unsafe { ffi::PyCapsule_GetPointer(context_obj.as_ptr(), raw) };
        if context.is_null() {
            unsafe { ffi::PyErr_Clear() };
        }
    } else if !context_obj.is_none() {
        let addr: usize = context_obj.extract()?;
        context = addr as *mut c_void;
    }

    // ---- function pointer ---------------------------------------------
    let func: *mut c_void;
    let mut name: Option<CString> = parse_name(name_obj)?;

    if is_capsule(func_obj) {
        let cname = capsule_name(func_obj);
        let raw = cname
            .as_ref()
            .map(|c| c.as_ptr())
            .unwrap_or(std::ptr::null());
        func = unsafe { ffi::PyCapsule_GetPointer(func_obj.as_ptr(), raw) };
        if func.is_null() {
            unsafe { ffi::PyErr_Clear() };
        }
        if context.is_null() {
            context = unsafe { ffi::PyCapsule_GetContext(func_obj.as_ptr()) };
            if context.is_null() {
                unsafe { ffi::PyErr_Clear() };
            }
        }
        if name.is_none() {
            name = cname;
        }
    } else {
        let addr: usize = func_obj.extract()?;
        func = addr as *mut c_void;
    }

    // ---- build the capsule --------------------------------------------
    let name_copy: *const c_char = match name {
        Some(c) => CString::into_raw(c),
        None => std::ptr::null(),
    };

    let capsule = unsafe { ffi::PyCapsule_New(func, name_copy, Some(raw_capsule_destructor)) };
    if capsule.is_null() {
        return Err(PyErr::fetch(py));
    }

    if !context.is_null() {
        let rc = unsafe { ffi::PyCapsule_SetContext(capsule, context) };
        if rc != 0 {
            return Err(PyErr::fetch(py));
        }
    }

    Ok(unsafe { Bound::from_owned_ptr(py, capsule) }.into_any())
}

/// get_capsule_signature(capsule) -> str
#[pyfunction]
fn get_capsule_signature(capsule: &Bound<'_, PyAny>) -> PyResult<String> {
    if !is_capsule(capsule) {
        return Err(PyValueError::new_err("expected a PyCapsule"));
    }
    match capsule_name(capsule) {
        Some(c) => Ok(c.to_string_lossy().into_owned()),
        None => Err(PyValueError::new_err("Capsule has no signature")),
    }
}

/// check_capsule(item) -> bool
#[pyfunction]
fn check_capsule(item: &Bound<'_, PyAny>) -> bool {
    is_capsule(item)
}

// ---------------------------------------------------------------------------
// Test helpers.  `scipy/_lib/tests/test_ccallback.py` and
// `scipy/integrate/tests/test_quadpack.py` drive these through ctypes, which
// is exactly the C ABI Rust can export natively.
// ---------------------------------------------------------------------------

/// ERROR_VALUE used by `src/_test_callback.c`; signals a failed callback.
const ERROR_VALUE: c_double = 2.0;

extern "C" fn plus1_cython(
    a: c_double,
    error_flag: *mut c_int,
    user_data: *mut c_void,
) -> c_double {
    if a == ERROR_VALUE {
        unsafe { *error_flag = 1 };
        Python::attach(|py| PyValueError::new_err("failure...").restore(py));
        return ERROR_VALUE;
    }
    if user_data.is_null() {
        a + 1.0
    } else {
        a + unsafe { *(user_data as *const c_double) }
    }
}

extern "C" fn plus1b_cython(
    a: c_double,
    b: c_double,
    error_flag: *mut c_int,
    user_data: *mut c_void,
) -> c_double {
    plus1_cython(a, error_flag, user_data) + b
}

extern "C" fn plus1bc_cython(
    a: c_double,
    b: c_double,
    c: c_double,
    error_flag: *mut c_int,
    user_data: *mut c_void,
) -> c_double {
    plus1_cython(a, error_flag, user_data) + b + c
}

extern "C" fn sine(x: c_double, _user_data: *mut c_void) -> c_double {
    x.sin()
}

fn fn_addr(f: *const ()) -> usize {
    f as usize
}

#[pyfunction]
fn plus1_addr() -> usize {
    fn_addr(plus1_cython as *const ())
}

#[pyfunction]
fn plus1b_addr() -> usize {
    fn_addr(plus1b_cython as *const ())
}

#[pyfunction]
fn plus1bc_addr() -> usize {
    fn_addr(plus1bc_cython as *const ())
}

#[pyfunction]
fn sine_addr() -> usize {
    fn_addr(sine as *const ())
}

/// Caller-side helper: invoke a low-level callback the same way the Cython
/// thunk did, i.e. dispatch on the capsule signature.
#[pyfunction]
fn test_call_cython(callback_obj: &Bound<'_, PyAny>, value: c_double) -> PyResult<c_double> {
    // `LowLevelCallable` is a tuple subclass whose first item is the capsule
    let target = match callback_obj.cast::<PyTuple>() {
        Ok(tuple) => tuple.get_item(0)?.into_any(),
        Err(_) => callback_obj.clone(),
    };

    if !is_capsule(&target) {
        return target.call1((value,))?.extract::<c_double>();
    }

    let name = capsule_name(&target).unwrap_or_default();
    let ptr = unsafe { ffi::PyCapsule_GetPointer(target.as_ptr(), name.as_ptr()) };
    if ptr.is_null() {
        unsafe { ffi::PyErr_Clear() };
        return Err(PyValueError::new_err("capsule has no function pointer"));
    }
    let mut error_flag: c_int = 0;

    // `ccallback_prepare` in the C original refuses unknown signatures, so the
    // three-argument "bad" callbacks must keep failing here too.
    let result = match name.as_bytes() {
        b"double (double, int *, void *)" => {
            let f: extern "C" fn(c_double, *mut c_int, *mut c_void) -> c_double =
                unsafe { std::mem::transmute(ptr) };
            f(value, &mut error_flag, std::ptr::null_mut())
        }
        b"double (double, double, int *, void *)" => {
            let f: extern "C" fn(c_double, c_double, *mut c_int, *mut c_void) -> c_double =
                unsafe { std::mem::transmute(ptr) };
            f(value, 0.0, &mut error_flag, std::ptr::null_mut())
        }
        other => {
            return Err(PyValueError::new_err(format!(
                "the caller does not know the signature {}",
                String::from_utf8_lossy(other)
            )));
        }
    };

    if error_flag != 0 {
        let err = PyErr::fetch(callback_obj.py());
        return Err(err);
    }
    Ok(result)
}

/// The signature table used by the low-level callback machinery.
fn make_sigs(py: Python<'_>) -> PyResult<Py<PyAny>> {
    let mut items: Vec<Py<PyAny>> = Vec::new();

    let mut push = |sig: &[u8], value: c_int| -> PyResult<()> {
        let bytes = PyBytes::new(py, sig);
        let value = value.into_pyobject(py)?;
        let tuple = PyTuple::new(py, [bytes.into_any(), value.into_any()])?;
        items.push(tuple.into_any().unbind());
        Ok(())
    };

    push(b"double (double, int *, void *)", 0)?;
    push(b"double (double, double, int *, void *)", 1)?;
    if std::mem::size_of::<c_int>() == std::mem::size_of::<c_long>() {
        push(b"double (double, long *, void *)", 0)?;
        push(b"double (double, double, long *, void *)", 1)?;
    }
    if std::mem::size_of::<c_int>() == std::mem::size_of::<c_short>() {
        push(b"double (double, short *, void *)", 0)?;
        push(b"double (double, double, short *, void *)", 1)?;
    }

    Ok(PyList::new(py, items)?.into_any().unbind())
}

pub(super) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = m.py();
    m.add_function(wrap_pyfunction!(get_raw_capsule, m)?)?;
    m.add_function(wrap_pyfunction!(get_capsule_signature, m)?)?;
    m.add_function(wrap_pyfunction!(check_capsule, m)?)?;
    m.add_function(wrap_pyfunction!(test_call_cython, m)?)?;
    m.add_function(wrap_pyfunction!(plus1_addr, m)?)?;
    m.add_function(wrap_pyfunction!(plus1b_addr, m)?)?;
    m.add_function(wrap_pyfunction!(plus1bc_addr, m)?)?;
    m.add_function(wrap_pyfunction!(sine_addr, m)?)?;
    m.add("sigs", make_sigs(py)?)?;
    Ok(())
}
