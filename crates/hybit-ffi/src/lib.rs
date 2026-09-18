use std::cell::RefCell;
use std::ffi::CString;
use std::os::raw::{c_char, c_int};
use std::ptr;
use std::panic::AssertUnwindSafe;
use std::slice;

use hybit_auto::{BackendPolicy, HybitPreparedSystem, HybitSolver};
use hybit_core::{MatrixBackend, PreconditionerKind, SolveStatus, SolverKind};
use hybit_matrix::Csr32Matrix;

const HYBIT_OK: c_int = 0;
const HYBIT_INVALID_ARGUMENT: c_int = 1;
const HYBIT_INVALID_MATRIX: c_int = 2;
const HYBIT_NUMERICAL_FAILURE: c_int = 3;
const HYBIT_NOT_CONVERGED: c_int = 4;
const HYBIT_PANIC: c_int = 100;

thread_local! {
    static LAST_ERROR: RefCell<CString> = RefCell::new(CString::new("no error").unwrap());
}

fn set_last_error(message: impl AsRef<str>) {
    let sanitized = message.as_ref().replace('\0', " ");
    LAST_ERROR.with(|slot| {
        *slot.borrow_mut() = CString::new(sanitized).unwrap_or_else(|_| CString::new("HyBIT error").unwrap());
    });
}

fn map_error(err: &hybit_core::HybitError) -> c_int {
    use hybit_core::HybitError::*;
    match err {
        InvalidArgument(_) | DimensionMismatch { .. } => HYBIT_INVALID_ARGUMENT,
        InvalidMatrix(_) | MissingDiagonal { .. } | ZeroDiagonal { .. } | SizeOverflow => HYBIT_INVALID_MATRIX,
        NotConverged { .. } => HYBIT_NOT_CONVERGED,
        NumericalBreakdown(_) => HYBIT_NUMERICAL_FAILURE,
    }
}

fn ffi_guard<F>(f: F) -> c_int
where
    F: FnOnce() -> Result<(), hybit_core::HybitError>,
{
    match std::panic::catch_unwind(AssertUnwindSafe(f)) {
        Ok(Ok(())) => HYBIT_OK,
        Ok(Err(err)) => {
            let code = map_error(&err);
            set_last_error(err.to_string());
            code
        }
        Err(_) => {
            set_last_error("panic crossed the HyBIT FFI boundary");
            HYBIT_PANIC
        }
    }
}

pub struct HybitSolverHandle {
    solver: HybitSolver,
}

pub struct HybitMatrixHandle {
    matrix: Csr32Matrix,
}

pub struct HybitPreparedHandle {
    prepared: HybitPreparedSystem,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct HybitSolveReport {
    pub status: c_int,
    pub solver_kind: c_int,
    pub preconditioner_kind: c_int,
    pub backend: c_int,
    pub iterations: u64,
    pub initial_residual: f64,
    pub final_residual: f64,
    pub relative_residual: f64,
    pub setup_seconds: f64,
    pub solve_seconds: f64,
    pub analysis_seconds: f64,
    pub prepare_seconds: f64,
    pub probe_seconds: f64,
    pub diagnostics_seconds: f64,
    pub local_factor_seconds: f64,
    pub restart_seconds: f64,
    pub escalations: u64,
    pub probe_iterations: u64,
    pub probe_final_residual: f64,
    pub hard_dofs: u64,
    pub local_direct_regions: u64,
    pub largest_local_region: u64,
    pub local_factor_dofs: u64,
    pub unique_local_factor_dofs: u64,
    pub local_factor_bytes: u64,
    pub overlap_layers: u64,
    pub preconditioner_reused: c_int,
    pub solve_sequence: u64,
    pub krylov_workspace_bytes: u64,
}

fn status_code(status: SolveStatus) -> c_int {
    match status {
        SolveStatus::Converged => 0,
        SolveStatus::MaxIterations => 1,
        SolveStatus::Breakdown => 2,
    }
}
fn solver_code(kind: SolverKind) -> c_int {
    match kind { SolverKind::Pcg => 1, SolverKind::Minres => 2, SolverKind::Gmres => 3, SolverKind::Bicgstab => 4, SolverKind::Hybrid => 5 }
}
fn precond_code(kind: PreconditionerKind) -> c_int {
    match kind { PreconditionerKind::None => 0, PreconditionerKind::Jacobi => 1, PreconditionerKind::BlockJacobi => 2, PreconditionerKind::LocalDirect => 3, PreconditionerKind::Hybrid => 4 }
}
fn backend_code(kind: MatrixBackend) -> c_int {
    match kind { MatrixBackend::Csr32 => 1, MatrixBackend::Abtm => 2, MatrixBackend::MatrixFree => 3 }
}

fn ffi_report(result: &hybit_core::SolveReport) -> HybitSolveReport {
    HybitSolveReport {
        status: status_code(result.status),
        solver_kind: solver_code(result.solver),
        preconditioner_kind: precond_code(result.preconditioner),
        backend: backend_code(result.backend),
        iterations: result.iterations as u64,
        initial_residual: result.initial_residual,
        final_residual: result.final_residual,
        relative_residual: result.relative_residual,
        setup_seconds: result.setup_seconds,
        solve_seconds: result.solve_seconds,
        analysis_seconds: result.analysis_seconds,
        prepare_seconds: result.prepare_seconds,
        probe_seconds: result.probe_seconds,
        diagnostics_seconds: result.diagnostics_seconds,
        local_factor_seconds: result.local_factor_seconds,
        restart_seconds: result.restart_seconds,
        escalations: result.escalations as u64,
        probe_iterations: result.probe_iterations as u64,
        probe_final_residual: result.probe_final_residual,
        hard_dofs: result.hard_dofs as u64,
        local_direct_regions: result.local_direct_regions as u64,
        largest_local_region: result.largest_local_region as u64,
        local_factor_dofs: result.local_factor_dofs as u64,
        unique_local_factor_dofs: result.unique_local_factor_dofs as u64,
        local_factor_bytes: result.local_factor_bytes as u64,
        overlap_layers: result.overlap_layers as u64,
        preconditioner_reused: if result.preconditioner_reused { 1 } else { 0 },
        solve_sequence: result.solve_sequence as u64,
        krylov_workspace_bytes: result.krylov_workspace_bytes as u64,
    }
}

#[no_mangle]
pub extern "C" fn hybit_version_major() -> u32 { 0 }
#[no_mangle]
pub extern "C" fn hybit_version_minor() -> u32 { 5 }
#[no_mangle]
pub extern "C" fn hybit_version_patch() -> u32 { 0 }

#[no_mangle]
pub unsafe extern "C" fn hybit_last_error_message(buffer: *mut c_char, capacity: usize) -> usize {
    LAST_ERROR.with(|slot| {
        let borrowed = slot.borrow();
        let bytes = borrowed.as_bytes_with_nul();
        if !buffer.is_null() && capacity > 0 {
            let n = bytes.len().min(capacity);
            ptr::copy_nonoverlapping(bytes.as_ptr() as *const c_char, buffer, n);
            if n == capacity {
                *buffer.add(capacity - 1) = 0;
            }
        }
        bytes.len()
    })
}

#[no_mangle]
pub unsafe extern "C" fn hybit_solver_create(out_solver: *mut *mut HybitSolverHandle) -> c_int {
    if out_solver.is_null() { set_last_error("out_solver is null"); return HYBIT_INVALID_ARGUMENT; }
    ffi_guard(|| {
        let handle = Box::new(HybitSolverHandle { solver: HybitSolver::new() });
        *out_solver = Box::into_raw(handle);
        Ok(())
    })
}

#[no_mangle]
pub unsafe extern "C" fn hybit_solver_destroy(solver: *mut HybitSolverHandle) {
    if !solver.is_null() { drop(Box::from_raw(solver)); }
}

#[no_mangle]
pub unsafe extern "C" fn hybit_solver_set_tolerances(
    solver: *mut HybitSolverHandle,
    relative_tolerance: f64,
    absolute_tolerance: f64,
) -> c_int {
    if solver.is_null() { set_last_error("solver is null"); return HYBIT_INVALID_ARGUMENT; }
    ffi_guard(|| {
        let handle = &mut *solver;
        let mut options = handle.solver.options();
        options.relative_tolerance = relative_tolerance;
        options.absolute_tolerance = absolute_tolerance;
        handle.solver.set_options(options)
    })
}

#[no_mangle]
pub unsafe extern "C" fn hybit_solver_set_max_iterations(solver: *mut HybitSolverHandle, max_iterations: u64) -> c_int {
    if solver.is_null() { set_last_error("solver is null"); return HYBIT_INVALID_ARGUMENT; }
    if max_iterations > usize::MAX as u64 { set_last_error("max_iterations is too large"); return HYBIT_INVALID_ARGUMENT; }
    ffi_guard(|| {
        let handle = &mut *solver;
        let mut options = handle.solver.options();
        options.max_iterations = max_iterations as usize;
        handle.solver.set_options(options)
    })
}

#[no_mangle]
pub unsafe extern "C" fn hybit_solver_set_backend(solver: *mut HybitSolverHandle, backend: c_int) -> c_int {
    if solver.is_null() { set_last_error("solver is null"); return HYBIT_INVALID_ARGUMENT; }
    let policy = match backend {
        0 => BackendPolicy::Auto,
        1 => BackendPolicy::Csr32,
        2 => BackendPolicy::Abtm,
        _ => { set_last_error("backend must be 0 (auto), 1 (CSR32), or 2 (ABTM)"); return HYBIT_INVALID_ARGUMENT; }
    };
    (*solver).solver.set_backend_policy(policy);
    HYBIT_OK
}

#[no_mangle]
pub unsafe extern "C" fn hybit_solver_set_hybrid_enabled(solver: *mut HybitSolverHandle, enabled: c_int) -> c_int {
    if solver.is_null() { set_last_error("solver is null"); return HYBIT_INVALID_ARGUMENT; }
    if enabled != 0 && enabled != 1 {
        set_last_error("enabled must be 0 or 1");
        return HYBIT_INVALID_ARGUMENT;
    }
    ffi_guard(|| {
        let handle = &mut *solver;
        let mut options = handle.solver.hybrid_options();
        options.enabled = enabled != 0;
        handle.solver.set_hybrid_options(options)
    })
}

#[no_mangle]
pub unsafe extern "C" fn hybit_solver_set_overlap_layers(solver: *mut HybitSolverHandle, layers: u64) -> c_int {
    if solver.is_null() { set_last_error("solver is null"); return HYBIT_INVALID_ARGUMENT; }
    if layers > usize::MAX as u64 { set_last_error("overlap_layers is too large"); return HYBIT_INVALID_ARGUMENT; }
    ffi_guard(|| {
        let handle = &mut *solver;
        let mut options = handle.solver.hybrid_options();
        options.overlap_layers = layers as usize;
        handle.solver.set_hybrid_options(options)
    })
}

#[no_mangle]
pub unsafe extern "C" fn hybit_matrix_create_csr_f64(
    nrows: u32,
    ncols: u32,
    nnz: u32,
    row_ptr: *const u32,
    col_idx: *const u32,
    values: *const f64,
    index_base: c_int,
    out_matrix: *mut *mut HybitMatrixHandle,
) -> c_int {
    if out_matrix.is_null() || row_ptr.is_null() || (nnz > 0 && (col_idx.is_null() || values.is_null())) {
        set_last_error("null pointer passed to hybit_matrix_create_csr_f64");
        return HYBIT_INVALID_ARGUMENT;
    }
    if index_base != 0 && index_base != 1 {
        set_last_error("index_base must be 0 or 1");
        return HYBIT_INVALID_ARGUMENT;
    }
    ffi_guard(|| {
        let rp_src = slice::from_raw_parts(row_ptr, nrows as usize + 1);
        let ci_src = slice::from_raw_parts(col_idx, nnz as usize);
        let va_src = slice::from_raw_parts(values, nnz as usize);
        let base = index_base as u32;
        let mut rp = Vec::with_capacity(rp_src.len());
        let mut ci = Vec::with_capacity(ci_src.len());
        for &v in rp_src { rp.push(v.checked_sub(base).ok_or(hybit_core::HybitError::InvalidMatrix("row_ptr contains index below index_base"))?); }
        for &v in ci_src { ci.push(v.checked_sub(base).ok_or(hybit_core::HybitError::InvalidMatrix("col_idx contains index below index_base"))?); }
        let matrix = Csr32Matrix::new(nrows as usize, ncols as usize, rp, ci, va_src.to_vec())?;
        *out_matrix = Box::into_raw(Box::new(HybitMatrixHandle { matrix }));
        Ok(())
    })
}

#[no_mangle]
pub unsafe extern "C" fn hybit_matrix_destroy(matrix: *mut HybitMatrixHandle) {
    if !matrix.is_null() { drop(Box::from_raw(matrix)); }
}

#[no_mangle]
pub unsafe extern "C" fn hybit_prepare(
    solver: *mut HybitSolverHandle,
    matrix: *const HybitMatrixHandle,
    out_prepared: *mut *mut HybitPreparedHandle,
) -> c_int {
    if solver.is_null() || matrix.is_null() || out_prepared.is_null() {
        set_last_error("null pointer passed to hybit_prepare");
        return HYBIT_INVALID_ARGUMENT;
    }
    ffi_guard(|| {
        let solver = &mut *solver;
        let matrix = &*matrix;
        let analysis = solver.solver.analyze_csr32(&matrix.matrix)?;
        let prepared = solver.solver.prepare_csr32(&matrix.matrix, &analysis)?;
        *out_prepared = Box::into_raw(Box::new(HybitPreparedHandle { prepared }));
        Ok(())
    })
}

#[no_mangle]
pub unsafe extern "C" fn hybit_prepared_destroy(prepared: *mut HybitPreparedHandle) {
    if !prepared.is_null() { drop(Box::from_raw(prepared)); }
}

#[no_mangle]
pub unsafe extern "C" fn hybit_solve_prepared(
    prepared: *mut HybitPreparedHandle,
    matrix: *const HybitMatrixHandle,
    b: *const f64,
    x: *mut f64,
    report: *mut HybitSolveReport,
) -> c_int {
    if prepared.is_null() || matrix.is_null() || b.is_null() || x.is_null() {
        set_last_error("null pointer passed to hybit_solve_prepared");
        return HYBIT_INVALID_ARGUMENT;
    }
    ffi_guard(|| {
        let prepared = &mut *prepared;
        let matrix = &*matrix;
        let nrows = matrix.matrix.nrows();
        let ncols = matrix.matrix.ncols();
        let b_slice = slice::from_raw_parts(b, nrows);
        let x_slice = slice::from_raw_parts_mut(x, ncols);
        let result = prepared.prepared.solve(&matrix.matrix, b_slice, x_slice)?;
        if !report.is_null() { *report = ffi_report(&result); }
        if result.converged() {
            Ok(())
        } else {
            Err(hybit_core::HybitError::NotConverged { iterations: result.iterations, residual: result.final_residual })
        }
    })
}

#[no_mangle]
pub unsafe extern "C" fn hybit_solve(
    solver: *mut HybitSolverHandle,
    matrix: *const HybitMatrixHandle,
    b: *const f64,
    x: *mut f64,
    report: *mut HybitSolveReport,
) -> c_int {
    if solver.is_null() || matrix.is_null() || b.is_null() || x.is_null() {
        set_last_error("null pointer passed to hybit_solve");
        return HYBIT_INVALID_ARGUMENT;
    }
    ffi_guard(|| {
        let solver = &mut *solver;
        let matrix = &*matrix;
        let nrows = matrix.matrix.nrows();
        let ncols = matrix.matrix.ncols();
        let b_slice = slice::from_raw_parts(b, nrows);
        let x_slice = slice::from_raw_parts_mut(x, ncols);
        let result = solver.solver.solve_csr32(&matrix.matrix, b_slice, x_slice)?;
        if !report.is_null() {
            *report = ffi_report(&result);
        }
        if result.converged() { Ok(()) } else { Err(hybit_core::HybitError::NotConverged { iterations: result.iterations, residual: result.final_residual }) }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ffi_end_to_end() {
        unsafe {
            let row_ptr = [0u32, 2, 5, 7];
            let col_idx = [0u32, 1, 0, 1, 2, 1, 2];
            let values = [2.0, -1.0, -1.0, 2.0, -1.0, -1.0, 2.0];
            let b = [1.0, 0.0, 1.0];
            let mut x = [0.0; 3];
            let mut solver: *mut HybitSolverHandle = ptr::null_mut();
            let mut matrix: *mut HybitMatrixHandle = ptr::null_mut();
            assert_eq!(hybit_solver_create(&mut solver), HYBIT_OK);
            assert_eq!(hybit_matrix_create_csr_f64(3, 3, 7, row_ptr.as_ptr(), col_idx.as_ptr(), values.as_ptr(), 0, &mut matrix), HYBIT_OK);
            let mut report = HybitSolveReport::default();
            assert_eq!(hybit_solve(solver, matrix, b.as_ptr(), x.as_mut_ptr(), &mut report), HYBIT_OK);
            assert_eq!(report.status, 0);

            let mut prepared: *mut HybitPreparedHandle = ptr::null_mut();
            assert_eq!(hybit_prepare(solver, matrix, &mut prepared), HYBIT_OK);
            let mut x2 = [0.0; 3];
            let mut prepared_report = HybitSolveReport::default();
            assert_eq!(hybit_solve_prepared(prepared, matrix, b.as_ptr(), x2.as_mut_ptr(), &mut prepared_report), HYBIT_OK);
            assert_eq!(prepared_report.status, 0);
            assert_eq!(prepared_report.solve_sequence, 1);
            hybit_prepared_destroy(prepared);
            hybit_matrix_destroy(matrix);
            hybit_solver_destroy(solver);
        }
    }
}
