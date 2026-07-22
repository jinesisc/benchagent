# BenchAgent — Dev Log

Working notes, one entry per session. Newest at the top.
Rule: 5 minutes max per entry. Bullet fragments are fine.

---

## 2026-07-18 · Session 1 — MockBench core

**Goal:** Scaffold project and begin building

**Did:**
- Scaffolding, set up Claude Code, developed `measure_vpp()` helper method

**Broke / surprised me:**
- Syntax issues 

**Learned:**
- Reviewed Python syntax, figured out that mock noise voltage is giving negative values (unrealistic)

**Next:** Finish developing `mock.py`

---

## 2026-07-17 · Session 2 — Mock.py

**Goal:** Finish mock.py and begin testing

**Did:**
- Finished implementing `mock.py` and began on test module `test_mock_physics.py`

**Broke / surprised me:**
- PyTests, method and syntax

**Learned:**
- How to use PyTests and dataclasses

**Next:** Finish developing `test_mock_physics.py` and begin on Tools aspect

---

## 2026-07-23 · Session 3 — Finalize mock physics test

**Goal:** Finish developing tests three and four

**Did:**
- Finished implementing `test_mock_physics.py` and learned more about testing noise

**Broke / surprised me:**
- Testing noise required more knowledge about statistics

**Learned:**
- How to deal with noisy samples, and calibration

**Next:** Begin on Tools aspect
