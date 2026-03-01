# AGENT.md — Multi-Agent Coordination Protocol
# Special Relativity in Financial Modeling (SRFM)

## Anti-Collision System

Each agent owns exactly one module. No agent touches another agent's files.
Before writing any file, check this ownership table.
If a file isn't listed under your agent ID, do not touch it.

---

## Agent Ownership Map

| Agent | ID     | Owns                                              | Output Files                                              |
|-------|--------|---------------------------------------------------|-----------------------------------------------------------|
| 1     | AGT-01 | Lorentz Transform Engine                          | `src/lorentz/`, `tests/lorentz/`                         |
| 2     | AGT-02 | Spacetime Market Manifold                         | `src/manifold/`, `tests/manifold/`                        |
| 3     | AGT-03 | Momentum-Velocity Signal Processor                | `src/momentum/`, `tests/momentum/`                        |
| 4     | AGT-04 | Tensor Calculus & Covariance Engine               | `src/tensor/`, `tests/tensor/`                            |
| 5     | AGT-05 | Relativistic Backtester                           | `src/backtest/`, `tests/backtest/`                        |
| 6     | AGT-06 | Integration Layer + CLI                           | `src/core/`, `src/main.cpp`, `tests/integration/`         |

**Shared (read-only for all agents):**
- `include/srfm/types.hpp` — shared types, do not modify, open PR if change needed
- `include/srfm/constants.hpp` — physical + financial constants
- `CMakeLists.txt` — AGT-06 owns, others submit requests via `BUILD_REQUESTS.md`

---

## Lock Protocol (File-Level Mutex)

Before writing any file:
1. Check `LOCKS.md` — if your target file is listed as locked by another agent, STOP
2. Write your agent ID + filename + timestamp to `LOCKS.md` before editing
3. Remove your lock entry when done

Format:
```
AGT-02 | src/manifold/market_manifold.cpp | 2026-02-28T05:14:00Z | LOCKED
```

---

## Build Order (Dependency Chain)

Agents must ship in this sequence. Later agents depend on earlier ones.

```
AGT-01 (Lorentz)
    ↓
AGT-02 (Manifold) ← depends on Lorentz transforms
    ↓
AGT-03 (Momentum) ← depends on Manifold
AGT-04 (Tensor)   ← depends on Lorentz + Manifold (parallel with AGT-03)
    ↓
AGT-05 (Backtest) ← depends on Momentum + Tensor
    ↓
AGT-06 (Integration) ← depends on all modules
```

AGT-03 and AGT-04 may run in parallel after AGT-02 completes.

---

## Step-by-Step Ship Protocol

Each agent ships in this exact order per module:

1. **Scaffold** — create directory + empty `.hpp` header with class declaration
2. **Types** — define all structs/enums in header, no implementation yet
3. **Implementation** — write `.cpp` with full logic
4. **Unit Tests** — write tests FIRST for each function (TDD where possible)
5. **Integration stub** — expose clean public API, document in `API.md`
6. **PR signal** — append one line to `PROGRESS.md` marking completion

---

## Test Ratio Requirement

**1.5:1 test-to-production code ratio enforced.**

Count lines of substantive code (exclude blank lines + comments):
- If `src/lorentz/lorentz_transform.cpp` = 100 lines → `tests/lorentz/` must have ≥ 150 lines
- Each agent self-audits before marking step complete
- AGT-06 runs final ratio check during integration

Test framework: **Google Test (gtest)**

---

## Communication Between Agents

Agents do NOT call each other directly. All cross-agent communication is via:
- **Shared headers** in `include/srfm/` (read-only interface contracts)
- **`PROGRESS.md`** — append-only status log
- **`BUILD_REQUESTS.md`** — file change requests to AGT-06 for CMakeLists

Never edit another agent's source files. If you need a change in another module, document it in `CROSS_AGENT_REQUESTS.md` with your agent ID and wait.

---

## Directory Structure (Final State)

```
special-relativity-finance/
├── AGENT.md                    ← this file
├── CLAUDE.md                   ← Claude behavior instructions
├── LOCKS.md                    ← live lock registry
├── PROGRESS.md                 ← append-only build log
├── BUILD_REQUESTS.md           ← CMake change requests
├── CROSS_AGENT_REQUESTS.md     ← inter-agent change requests
├── API.md                      ← public API documentation
├── CMakeLists.txt              ← AGT-06 owns
├── include/
│   └── srfm/
│       ├── types.hpp           ← shared types (read-only)
│       └── constants.hpp       ← shared constants (read-only)
├── src/
│   ├── lorentz/                ← AGT-01
│   ├── manifold/               ← AGT-02
│   ├── momentum/               ← AGT-03
│   ├── tensor/                 ← AGT-04
│   ├── backtest/               ← AGT-05
│   ├── core/                   ← AGT-06
│   └── main.cpp                ← AGT-06
└── tests/
    ├── lorentz/                ← AGT-01
    ├── manifold/               ← AGT-02
    ├── momentum/               ← AGT-03
    ├── tensor/                 ← AGT-04
    ├── backtest/               ← AGT-05
    └── integration/            ← AGT-06
```
