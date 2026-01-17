# Dependency Graph

```mermaid
flowchart TB
    subgraph M0["M0: Specification"]
        S[Spec Documents]
    end
    
    subgraph M1["M1: Scaffolding"]
        R[Repo Structure]
        P[Packaging]
        CI[CI Pipeline]
    end
    
    subgraph M2["M2: Schema"]
        SC[Step/Episode]
        SP[DatasetSpec]
        IF[Interfaces]
    end
    
    subgraph M3["M3: Ingestion"]
        AD[TFDS Adapters]
        RL[RLDS Parsing]
    end
    
    subgraph M4["M4: Transforms"]
        TR[Transforms]
        VA[Validators]
        CO[Compiler]
    end
    
    subgraph M5["M5: LeRobot Writer"]
        LW[LeRobot v3]
    end
    
    subgraph M6["M6: RLDS Export"]
        RE[RLDS/TFDS]
    end
    
    subgraph M7["M7: Indexing"]
        IX[Index]
        QE[Query]
    end
    
    subgraph M8["M8: Training/Eval"]
        TD[Dataset]
        EV[Eval]
    end
    
    S --> R
    R --> P --> CI
    CI --> SC --> SP --> IF
    IF --> AD --> RL
    IF --> TR --> VA --> CO
    RL --> CO
    CO --> LW
    CO --> RE
    LW --> IX --> QE
    LW --> TD
    TD --> EV
```

## Critical Path

1. **M0 → M1**: Spec before scaffolding
2. **M1 → M2**: Package before schema
3. **M2 → M3**: Interfaces before adapters
4. **M2 → M4**: Interfaces before transforms
5. **M3 + M4 → M5/M6**: Ingestion + transforms before writers
6. **M5 → M7**: LeRobot writer before indexing
7. **M5 → M8**: Writer before training utilities

## Parallelizable

- M5 (LeRobot) and M6 (RLDS) can proceed in parallel
- M3 (Ingestion) and M4 (Transforms) after M2
- Sub-tasks within each milestone
