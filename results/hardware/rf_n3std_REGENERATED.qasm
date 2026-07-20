// rf_n3std_REGENERATED.qasm — deterministic regeneration (NOT the original submission artifact).
// Source: experiments/register_fault_build.py build path, initial_layout=[0,1,2,9,10,11]
//   destructive_swap_test(ghz_state(3)) -> transpile(cepheus_coupling_map, CEPHEUS_BASIS_GATES,
//   optimization_level=3, seed_transpiler=0), emit_physical_qasm3.  Regenerated at commit e25eef9.
// Register {0,1,2,9,10,11} (Appendix C n=3), 7 CZ, zero routing, noiseless purity 1.0.
// BYTE-IDENTICAL to the committed results/hardware/hg_coll_n3.qasm, which is the circuit the
// n3_std cell actually submitted (run_register_fault.py:34). rf_n3std.qasm was never a separate file.
OPENQASM 3.0;
include "stdgates.inc";
bit[6] c;
rz(-1.5707963267948966) $0;
rx(-1.5707963267948966) $0;
rz(-1.5707963267948966) $0;
rz(1.5707963267948966) $1;
rx(1.5707963267948963) $1;
cz $0, $1;
rx(-1.5707963267948966) $1;
rz(-1.5707963267948966) $1;
rz(1.5707963267948966) $2;
rx(1.5707963267948963) $2;
cz $1, $2;
rx(-1.5707963267948966) $2;
rz(-1.5707963267948966) $2;
rz(-1.5707963267948966) $9;
rx(-1.5707963267948966) $9;
rz(-1.5707963267948966) $9;
rz(1.5707963267948966) $10;
rx(1.5707963267948963) $10;
cz $9, $10;
rx(-1.5707963267948966) $10;
rz(-1.5707963267948966) $10;
rz(1.5707963267948966) $11;
rx(1.5707963267948963) $11;
cz $10, $11;
rx(-1.5707963267948966) $11;
rz(-1.5707963267948966) $11;
barrier $0, $1, $2, $9, $10, $11;
rz(1.5707963267948966) $9;
rx(1.5707963267948963) $9;
cz $0, $9;
rz(-1.5707963267948966) $0;
rx(-1.5707963267948966) $0;
rz(-1.5707963267948966) $0;
rx(-1.5707963267948966) $9;
rz(-1.5707963267948966) $9;
rz(1.5707963267948966) $10;
rx(1.5707963267948963) $10;
cz $1, $10;
rz(-1.5707963267948966) $1;
rx(-1.5707963267948966) $1;
rz(-1.5707963267948966) $1;
rx(-1.5707963267948966) $10;
rz(-1.5707963267948966) $10;
rz(1.5707963267948966) $11;
rx(1.5707963267948963) $11;
cz $2, $11;
rz(-1.5707963267948966) $2;
rx(-1.5707963267948966) $2;
rz(-1.5707963267948966) $2;
rx(-1.5707963267948966) $11;
rz(-1.5707963267948966) $11;
c[0] = measure $0;
c[1] = measure $1;
c[2] = measure $2;
c[3] = measure $9;
c[4] = measure $10;
c[5] = measure $11;
