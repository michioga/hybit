# Security and numerical reliability

HyBIT is numerical software, not a security product. The primary risks are incorrect numerical results, unsupported matrix classes, resource exhaustion, and FFI misuse.

Please do not include private or confidential matrices in public issues. Reduce a problem to a synthetic or sanitized reproducer whenever possible.

For numerical correctness issues, include the matrix properties you expect, the right-hand side, solver settings, residuals, and a reference result if available. HyBIT 0.6.0 is experimental and currently targets real SPD systems on its automatic path.
