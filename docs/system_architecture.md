# System Architecture & Generation Pipeline

The UTDTB v5.0 generator is a multi-stage simulation engine that orchestrates thermodynamic, mechanical, and environmental models.

## 1. Generation Logic Flow
The following diagram illustrates the sequence from high-level split orchestration to the cycle-level physics loop.

```text
┌─────────────────────────────────────────────────────────────┐
│                  UTDTBGenerator.generate()                  │
└───────────────────────────┬─────────────────────────────────┘
                            │
     ┌──────────────────────┼──────────────────────┐
     ▼                      ▼                       ▼
[TRAIN split]          [VAL split]           [TEST split]
     │                      │                       │
     └──────────────────────┼───────────────────────┘
                            │
                 ┌──────────▼──────────┐
                 │   EngineSimulator   │
                 └──────────┬──────────┘
                            │
       ┌────────────────────┼─────────────────────┐
       ▼                    ▼                      ▼
 [Stage 1-3]           [Stage 4-6]            [Stage 7-9]
  Init & Context        Env & FADEC            Physics & Noise
 (MFG, Fleet, Bias)    (Phase, Alt, Surge)    (Brayton, Degrad, HDF5)
