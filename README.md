# Bayes-optimal best-arm identification

Code accompanying *Suboptimal Frequentist Performance of the Flat-Prior Generalized Bayes Optimal Algorithm in Best Arm Identification*.

## Lean proofs

The [Lean Proofs](Lean%20Proofs/README.md) directory contains one Lean source file, a guide to the paper's results, and the pinned configuration needed to check it. Start with its [installation and verification instructions](Lean%20Proofs/README.md#run-the-verification), then use the [paper-to-proof map](Lean%20Proofs/PAPER_MAP.md) to find individual statements and their scope.

## Simulations

To reproduce the simulation of the paper, run:

```sh
python3 bayesoptimal.py > results.txt
```

A simulation run yields two figures (PDFs) of the paper. The simulation is separate from the Lean proof verification.
