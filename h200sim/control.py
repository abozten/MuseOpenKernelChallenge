"""Compiler-managed dependencies: the control bits of NVIDIA SASS (Huerta et al., measured on Ampere).

Real hardware has no scoreboard. Every instruction carries
  * a **stall count**: after issuing it the warp waits this many cycles (the compiler knows the
    latency of fixed-latency results, so it spaces producer and consumer by exactly the latency);
  * up to two **dependence counters** (of six, SB0..SB5, each up to 63) it increments when it
    issues, one decremented at write-back (read-after-write / write-after-write), one when its
    source registers have been read (write-after-read);
  * a **wait mask**: the counters that must be zero before it may issue.
Variable-latency instructions (memory, shuffles, special functions, conversions, S2R, constants)
cannot be spaced statically and use the counters. Counters are shared between producers, so a
consumer waits for *all* outstanding producers on its counter, not only the one it needs.

`annotate` plays ptxas: it derives these bits from the program. Selected with Sim(dep_mode="control").
"""

FIXED_PIPES = {"alu", "int", "fp32", "fp64", "tensor", "uni"}
VARIABLE_OPS = {"S2R", "LDC", "LDC64", "ULDC", "LDC_MEM", "SHFL_XOR", "R2UR"}
RAW_COUNTERS = 5           # SB0..SB4 carry RAW / WAW; SB5 is kept for write-after-read
WAR_COUNTER = 5


def is_variable(ins) -> bool:
    return ins.pipe in ("lsu", "xu", "tma") or ins.op in VARIABLE_OPS


def annotate(prog) -> None:
    code = prog.instrs
    targets = {i.target for i in code if isinstance(i.target, int)} | \
              {i.reconv for i in code if isinstance(i.reconv, int)}
    t = 0                                   # static issue time of the previous instruction
    fixed_ready = {}                        # register -> cycle its fixed-latency result is ready
    var_writer = {}                         # register -> counter a variable-latency producer will signal on
    var_reader = {}                         # register -> WAR counter of a variable-latency instruction reading it
    used = set()                            # counters in use since the last merge point
    rr = 0
    prev = None
    for pc, ins in enumerate(code):
        need_t, wait = 0, set()
        if pc in targets:                   # control-flow merge: be conservative, wait for everything in flight
            need_t = max(fixed_ready.values(), default=0)
            wait |= used
            fixed_ready.clear()
        for r in ins.deps:                  # RAW on sources, WAW on destinations
            if r in fixed_ready:
                need_t = max(need_t, fixed_ready[r])
            if r in var_writer:
                wait.add(var_writer[r])
        for r in ins.writes:                # WAR: a pending variable-latency instruction still has to read r
            if r in var_reader:
                wait.add(var_reader[r])
        if prev is not None:
            t = max(t + 1, need_t)
            prev.stall = t - prev_t
        else:
            t = max(0, need_t)
        ins.wait = tuple(sorted(wait))
        ins.inc_raw = ins.inc_war = None
        if is_variable(ins):
            reads = [r for r in ins.deps if r not in ins.writes]
            if reads:
                ins.inc_war = WAR_COUNTER
                for r in reads:
                    var_reader[r] = WAR_COUNTER
                used.add(WAR_COUNTER)
            if ins.writes:
                ins.inc_raw = rr % RAW_COUNTERS
                rr += 1
                used.add(ins.inc_raw)
                for r in ins.writes:
                    var_writer[r] = ins.inc_raw
                    fixed_ready.pop(r, None)
        else:
            for r in ins.writes:
                fixed_ready[r] = t + int(-(-ins.lat // 1))
                var_writer.pop(r, None)
        for r in ins.writes:                # a new value in r ends any WAR obligation on its old value
            var_reader.pop(r, None)
        ins.stall = 1
        prev, prev_t = ins, t
    if prev is not None:
        prev.stall = 1
