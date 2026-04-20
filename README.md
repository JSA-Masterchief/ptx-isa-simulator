# PTX Assembler Simulator

A software simulator for NVIDIA's PTX (Parallel Thread Execution) ISA.

## What is PTX?
PTX is NVIDIA's virtual ISA used as an intermediate representation for GPU programs. The PTXAS compiler converts PTX to actual GPU machine code (SASS).

## Features
- Full PTX instruction parser
- Virtual register file per thread
- Predicated execution (`@%p`, `@!%p`)
- Special registers: `%tid.x`, `%ntid.x`, `%ctaid.x`
- Warp simulation (32 threads executing in SIMT)
- Execution statistics tracking

## Supported Instructions
```
Data Movement:   mov, ld, st, cvt
Arithmetic:      add, sub, mul, div, rem, mad, abs, neg, min, max
Comparison:      setp (lt, le, gt, ge, eq, ne)
Logic:           and, or, xor, not, shl, shr
Control Flow:    bra, ret, exit
Selection:       selp
Synchronization: bar
Atomic:          atom
```

## Build & Run
```bash
g++ -std=c++17 -O2 -o ptx_sim main.cpp
./ptx_sim
```

## Sample PTX Programs Included
1. Vector Addition kernel
2. Predicated execution
3. Loop with accumulation
4. FMA (Fused Multiply-Add)

## SIMT Execution Model
```
Warp = 32 threads executing same instruction
Each thread has:
  - Own register file
  - Own %tid (thread ID)
  - Shared %ctaid (block ID)

Divergence: threads take different paths
  → some threads masked out via predicates
```

## Concepts Covered
- GPU Execution Model (SIMT)
- Warp Scheduling Basics
- Predicated Execution
- Instruction-Level Simulation

## Future Work
- Memory hierarchy simulation
- Warp divergence handling improvements
- PTX-to-SASS style lowering
