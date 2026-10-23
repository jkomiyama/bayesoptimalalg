JUNPEI PAPER — LEAN SUBMISSION ARTIFACT

Companion to the supplied manuscript:
  Suboptimal Frequentist Performance of the Flat-Prior Generalized
  Bayes Optimal Algorithm in Best Arm Identification
  Junpei Komiyama; manuscript dated July 13, 2026.

Start here. This is a standalone Lean project containing the formal proof
sources, the exact manuscript being referenced, a paper-to-code guide, and
reproducible verification commands. No other project directory is required.

WHAT THIS ARTIFACT ESTABLISHES

The formal counterparts of all 18 numbered proved results are included:
Theorems 1–2 and Lemmas 3–8 and 10–19. The main result includes the route from
initialization-only Bayes-risk minimization to the original frequentist regret
law, together with the supporting policy constructions and probability bounds.
There are 156 proof modules and 1,517 public theorem/lemma declarations.

Revision 2 closes the first release's Theorem 1 correspondence gap. The first
release proved general finite-arm optimality of a constructed policy, while
the implication from arbitrary actual-risk optimality to the stated decision
rules was restricted to three arms and integer counts. The new endpoint
Junpei.GeneralOptimality.FiniteHorizon.theorem1 proves that implication for
every finite nonempty arm set, positive real precision vector, and finite
collection of Borel policy rules defined on proper Gaussian states. Its
premise compares actual risk against all concrete Borel history competitors.
REVISION_NOTES.txt describes the release changes.

Revision 3 retains all 155 revision 2 proof modules and adds the corrected
Footnote 14 characterization. For every nonnegative sequence d(T), Lean proves
that P[N(0,T^-3) >= d(T)] >= c*T^(-p) for some positive real constants c,p and
all integer T>=T0, with a fixed T0 independent of T, if and only if
d(T)=O(T^(-3/2)*sqrt(log T)). The theorem uses the actual Gaussian measure and
Lean's IsBigO relation. It assumes no monotonicity of d. This new theorem and
the original footnote counterexample have separate entries below.

Important manuscript status: Footnote 14, PDF page 38, is false in the
polynomial-probability-lower-bound sense used in the discussion. This artifact
proves its counterexample. It does not prove the false footnote. Conjecture 9
remains open. Some contextual general-prior and cited-literature statements
are outside the formalization. Read ALL_ARGUMENTS_STATUS.txt for exact limits.
The supplied PDF is unchanged. A successful build concerns the explicit Lean
statements, including the counterexample; it does not certify every PDF sentence.

QUICK START

Prerequisites: Git, Lean's elan version manager (providing lake), and Python
3.10 or later. Install Lean using the official instructions:
https://lean-lang.org/install/manual/

Open a terminal in THIS directory, the one containing lakefile.toml.
The lean-toolchain file selects Lean 4.24.0 automatically through elan.
Do not replace lean-toolchain or regenerate lake-manifest.json for this release.

1. Obtain the pinned dependencies and their precompiled library cache:

     lake exe cache get

   This step needs internet access on first setup. Git revisions are locked in
   lake-manifest.json. Downloading the mathlib cache avoids rebuilding mathlib.
   Reference: https://leanprover-community.github.io/install/project.html

2. Run the complete verification, rebuilding the local proofs:

   Windows PowerShell:
     python -X utf8 verify.py --fresh

   Linux / macOS:
     python3 -X utf8 verify.py --fresh

   --fresh cleans only this project's compiled outputs. Dependency caches are
   retained. Omit --fresh for an incremental build followed by the same audits.
   Python launches Lean, inventories sources, and records evidence; all
   mathematical proof checking is performed by Lean.

3. Read REPORT.txt and verification_results.json after the command finishes.
   A successful run returns exit code 0, reports verification_ok: true, and
   has no audit issues. Raw Lean outputs are in logs/.

Run `python verify.py --help` (or python3) for options. --lake accepts the path
to a Lake executable if lake is not on PATH. --jobs controls concurrent audit
processes after the build; it does not change Lake's build scheduler.

HOW TO FIND A PAPER PROOF

  docs/PROOF_MAP.html       Offline searchable paper-to-proof guide, with PDF
                           page links and links to the corresponding sources.
  docs/PROOF_MAP.txt        Readable map of theorems, lemmas, definitions,
                           all numbered equations, remarks, and extra arguments.
  docs/proof_map.tsv        The same mapping in spreadsheet-friendly form.
  docs/DECLARATIONS.html    Offline searchable browser for all 1,517 public
                           theorem/lemma declarations, with source-file links.
  docs/declarations.tsv     Full declaration names, source paths, and line numbers.
  docs/MODULES.txt          Index of all 156 proof modules.
  docs/MAIN_PROOF_ROUTE.txt Suggested reading order for the end-to-end main proof.
  COVERAGE.txt             Detailed hypotheses and scope for each numbered result.
  ALL_ARGUMENTS_STATUS.txt Unnumbered arguments, Footnote 14, and remaining limits.
  ADDITIONAL_PROOFS.txt    Mock policies, embeddings, and shared-sample couplings.

Open either HTML guide in a browser; both work without a server or network.
For interactive Lean inspection, open this directory in VS Code with the
official Lean 4 extension, then open a .lean file. The InfoView shows theorem
statements, assumptions, goals, and compiler messages. The source line numbers
in the index refer to this exact release; a plain browser opens the source file,
while an editor can navigate to the listed line.

KEY ENTRY POINTS

  Junpei/RealPrecisionMain.lean
    Recommended main-theorem endpoint. The policy has jointly Borel real-
    precision posterior states. Its premise is actual risk minimization only
    after initialization, against concrete Borel full-history competitors.
    Start with Junpei.RealPrecisionMain.theorem2_paper_exponent and then read
    Junpei.RealPrecisionMain.InitialHistoryRiskOptimal.

  Junpei/InitialHistoryMain.lean
    The same main endpoints for the integer-count posterior-state policy type.

  Junpei/GeneralOptimalityFinite.lean
    Theorem 1 for any sequentially posterior-optimal finite policy on proper
    Gaussian states. Start with Junpei.GeneralOptimality.FiniteHorizon.theorem1
    and its SequentiallyOptimal definition. The conclusions are actual risk
    equal to Bellman loss, EBI-maximizing sampling, and maximum-mean terminal
    recommendation. Decisions satisfy the rules almost surely in the fresh
    randomizer at every proper state. A concrete minimizing policy is proved
    to exist in the same finite policy class.

  Junpei/GeneralOptimality.lean and GeneralOptimalityHistory.lean
    Derive those decision rules from actual-risk minimization at a single
    state, using the explicit state-policy-to-history compiler. The Proper
    and Finite modules prove exact execution and risk identities for the
    paper's proper-state domain and finite collection of policy rules.

  Junpei/FinitePosterior.lean
    Gaussian posterior-loss identities and the earlier constructed-policy
    optimality results used by the general Theorem 1 proof chain.

  Junpei/Main.lean and Junpei/MainExactRate.lean
    Core regret bound, actual frequentist regret definition, event argument,
    and the paper's exponent. These core endpoints use local optimality;
    the RealPrecisionMain route derives the required behavior from initial
    actual-risk minimization through the intervening proved reductions.

  Junpei/Footnote14Scale.lean
    Direct formal negation of the footnote's claimed necessary gap scale.

  Junpei/Footnote14Characterization.lean
    The corrected equivalence for arbitrary nonnegative threshold sequences:
    Junpei.Footnote14Characterization.corrected_footnote14_iff.
    The constants c>0, p>0 are real and T0 is natural; all are chosen once for
    the sequence, independently of T. The older counterexample is retained.
    The supplied manuscript has not been rewritten by this proof addition.

FOCUSED INSPECTION COMMANDS

After building, these commands print formal signatures and axiom dependencies:

  lake env lean ClaimsAudit.lean
  lake env lean MainTheoremAudit.lean
  lake env lean AdditionalProofAudit.lean
  lake env lean EndToEndAudit.lean
  lake env lean BundleAudit.lean
  lake env lean checks/ProseArgumentAudit.lean
  lake env lean checks/GeneralOptimalityAudit.lean
  lake env lean checks/Footnote14CharacterizationAudit.lean

`lake build` builds all imported proof modules, but does not itself execute
every standalone audit. Use verify.py for the complete recorded audit run.
The verifier compiles all twelve checks/*.lean files in addition to the root audits.

FILE ORGANIZATION

  Draft0/                  Required Gaussian/Bellman foundation (34 modules).
  Junpei/                  Revised-paper proofs and extensions (122 modules).
  JunpeiVersionLean.lean    Imports every proof module.
  *Audit.lean, checks/      Signature and transitive axiom inspections.
  BundleAudit.lean          Aggregate reference to all public theorem/lemma terms.
  declarations.json        Names included in the aggregate audit.
  lean-toolchain           Exact Lean release.
  lakefile.toml            Library/build configuration; warnings treated as errors.
  lake-manifest.json       Exact dependency Git revisions.
  verify.py                Portable full verification driver, Python stdlib only.
  manuscript/              Exact PDF and extracted text used by the proof map.
  docs/                    Navigation and reading guides.
  logs/, REPORT.txt,
  verification_results.json Evidence from verification of this packaged project.
  evidence/                Provenance and the prior original-project run.
  tools/                   Administrative index and package-integrity utilities.

Keep Draft0/: despite its historical name, these are required imported proof
foundations, not a second manuscript or an optional earlier proof bundle.
No Lean installation, dependency checkout, .lake cache, or compiled local proof
object is needed in the distributed archive; first setup obtains dependencies.

INTERPRETING VERIFICATION

Lean checks explicit propositions under their displayed hypotheses. The paper
map explains how those propositions represent the manuscript. The axiom audit
permits only propext, Classical.choice, and Quot.sound. It rejects sorryAx and
other unproved axiom dependencies. The source scan also rejects sorry/admit
placeholders and native_decide shortcuts. This artifact retains the standard
Lean/mathlib foundations used by the original project.

For Theorem 2, the positive gap is arbitrary; the polynomial lower bound is for
all sufficiently large ODD horizons, uniformly over the stated policy class.
The full-horizon consequence is a zero limit-inferior exponential rate.
Read the exact Lean signatures before interpreting either as an even-horizon
bound or a claim for unrestricted history-dependent main policies.

The packaged release includes raw successful compiler evidence. To check the
supplied files before a rerun, run:

  python tools/check_package.py

The release checksums identify the supplied files; they are not mathematical
verification or a cryptographic signature from an independent authority.
REPORT.txt, verification_results.json, and logs/ are regenerated on a rerun,
so their release checksums can then differ legitimately. Proof/input changes
must be reverified with Lean. The verification JSON records the inputs used.

To regenerate navigation after an intentional source edit, run
`python tools/build_index.py`. To create a new ZIP and checksums after verifying
and reviewing your changes, run `python tools/make_archive.py`; use --name
NAME.zip for a distinct archive, or --replace to replace an existing ZIP.
The archive excludes .lake
and Python caches and checks every archived file against its on-disk source.

ENVIRONMENT AND TROUBLESHOOTING

The packaged project is tested with the pinned Windows Lean toolchain. The
driver uses portable Python and Lake commands; Linux/macOS commands are provided
but are not claimed to have been independently tested on those systems.

If lake is not found, install elan and reopen the terminal. If Python is named
python3 or py on your machine, use that launcher. If dependencies are missing,
run lake exe cache get from this directory. Building without its cache is much
slower. A fresh local build can take several minutes; logs/build.log records
progress. No assertion of success should be based solely on a pre-existing log.

DEPENDENCY INFORMATION

Lean: leanprover/lean4:v4.24.0
mathlib: v4.24.0, commit f897ebcf72cd16f89ab4577d0c826cd14afaafc7
All transitive revisions and upstream URLs: lake-manifest.json.
Dependency sources/binaries are downloaded separately from their upstreams;
their respective upstream licenses apply. No new license for the paper or its
original proof sources is assigned by this packaging step.
