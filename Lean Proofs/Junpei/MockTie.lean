import Junpei.MockRisk

set_option linter.unusedSectionVars false
set_option maxHeartbeats 1000000

open MeasureTheory ProbabilityTheory
open scoped NNReal ENNReal

namespace Junpei.MockTie

open Junpei.FiniteGaussian Junpei.FinitePosterior Junpei.FiniteModelExclusion
open Junpei.FiniteModelMeasurable Junpei.EmbeddingRaw Junpei.EmbeddingExecution
open Junpei.MockPolicy Junpei.MockMeasurable Junpei.MockRisk

variable {A : Type*} [Fintype A] [Nonempty A] [DecidableEq A] [LinearOrder A]
variable [MeasurableSpace A] [MeasurableSingletonClass A]

noncomputable section

def maximizers (m : Means A) : Finset A := Finset.univ.filter (fun a => m a = maxAll m)

theorem maximizers_nonempty (m : Means A) : (maximizers m).Nonempty := by
  obtain ⟨a, ha⟩ := maxAll_attained m
  exact ⟨a, by simp [maximizers, ha]⟩

def minimumMaximizer (m : Means A) : A := (maximizers m).min' (maximizers_nonempty m)

theorem minimumMaximizer_attains (m : Means A) : m (minimumMaximizer m) = maxAll m := by
  exact (Finset.mem_filter.mp (Finset.min'_mem (maximizers m) (maximizers_nonempty m))).2

theorem minimumMaximizer_le (m : Means A) (a : A) (ha : m a = maxAll m) :
    minimumMaximizer m ≤ a := by
  exact Finset.min'_le (maximizers m) a (by simp [maximizers, ha])

theorem minimumMaximizer_eq_iff (m : Means A) (a : A) :
    minimumMaximizer m = a ↔ m a = maxAll m ∧ ∀ b, b < a → m b < maxAll m := by
  constructor
  · intro h
    constructor
    · rw [← h]; exact minimumMaximizer_attains m
    · intro b hb
      apply lt_of_le_of_ne (coordinate_le_maxAll m b)
      intro hEq
      have hle := minimumMaximizer_le m b hEq
      rw [h] at hle
      exact (not_le_of_gt hb) hle
  · rintro ⟨ha, hb⟩
    apply le_antisymm (minimumMaximizer_le m a ha)
    apply le_of_not_gt
    intro h
    have hh := hb (minimumMaximizer m) h
    rw [minimumMaximizer_attains m] at hh
    exact lt_irrefl _ hh

/-- The literal minimum-index argmax convention is Borel on every finite
linearly ordered arm set, in particular Fin K and its retained subtype. -/
theorem minimumMaximizer_measurable : Measurable (minimumMaximizer (A := A)) := by
  apply measurable_to_countable'
  intro a
  have he : minimumMaximizer ⁻¹' {a} =
      {m : Means A | m a = maxAll m} ∩
        ⋂ b : A, {m : Means A | b < a → m b < maxAll m} := by
    ext m
    simp only [Set.mem_preimage, Set.mem_singleton_iff, Set.mem_inter_iff,
      Set.mem_iInter, Set.mem_setOf_eq]
    exact minimumMaximizer_eq_iff m a
  rw [he]
  apply MeasurableSet.inter
  · exact measurableSet_eq_fun (measurable_pi_apply a) maxAll_lipschitz.continuous.measurable
  · apply MeasurableSet.iInter
    intro b
    by_cases h : b < a
    · simp only [h, true_implies]
      exact measurableSet_lt (measurable_pi_apply b) maxAll_lipschitz.continuous.measurable
    · simp only [h, false_implies]
      exact MeasurableSet.univ

section Removed
variable (i : A) [Nonempty {a : A // a ≠ i}]

def minimumRetained (m : Means A) : A :=
  ((minimumMaximizer (A := {a : A // a ≠ i}) (fun a => m a) : {a : A // a ≠ i}) : A)

theorem minimumRetained_excludes (m : Means A) : minimumRetained i m ≠ i :=
  (minimumMaximizer (A := {a : A // a ≠ i}) (fun a => m a)).property

theorem minimumRetained_attains (m : Means A) :
    m (minimumRetained i m) = otherMaximum i m :=
  minimumMaximizer_attains (A := {a : A // a ≠ i}) (fun a => m a)

theorem minimumRetained_le (m : Means A) (a : A) (ha : a ≠ i)
    (hm : m a = otherMaximum i m) : minimumRetained i m ≤ a :=
  minimumMaximizer_le (A := {a : A // a ≠ i}) (fun a => m a) ⟨a, ha⟩ hm

theorem minimumRetained_measurable : Measurable (minimumRetained i) :=
  measurable_subtype_coe.comp (minimumMaximizer_measurable.comp
    (measurable_pi_lambda _ (fun a => measurable_pi_apply (a : A))))

/-- This equality is of actual conditional posterior regret integrals, not
merely equality of maximized means. The literal minimum selector may change
the arm selected on a tie, while its conditional risk remains identical. -/
theorem minimumRetained_terminalRegret (c : Precisions A) (m : Means A) (a : A)
    (ha : m a = maxAll m) :
    terminalRegret c m (minimumRetained i m) = terminalRegret c m (terminalChoice i m a) := by
  rw [terminalRegret_eq, terminalRegret_eq, minimumRetained_attains i m,
    terminalChoice_posterior_best i m a ha]

def literalPolicy : (H : ℕ) → Precisions A → Means A → Strategy (A := A) H
  | 0, _, m => .stop (minimumRetained i m)
  | H + 1, c, m =>
      let a := chooseFirst H c m
      .sample a (fun z => literalPolicy H (advance c a) (shift m a z))

def legacyPolicy : (H : ℕ) → Precisions A → Means A → Strategy (A := A) H
  | 0, _, m => .stop (terminalChoice i m (firstMax m))
  | H + 1, c, m =>
      let a := chooseFirst H c m
      .sample a (fun z => legacyPolicy H (advance c a) (shift m a z))

theorem policies_utility_eq (H : ℕ) (c : Precisions A) (m : Means A) :
    utility (literalPolicy i H c m) c m = utility (legacyPolicy i H c m) c m := by
  induction H generalizing c m with
  | zero =>
    simp only [literalPolicy, legacyPolicy, utility, minimumRetained_attains,
      terminalChoice_posterior_best i m (firstMax m) (firstMax_attains m)]
  | succ H ih => simp only [literalPolicy, legacyPolicy, utility, ih]

theorem policies_expectedRegret_eq (H : ℕ) (c : Precisions A) (m : Means A) :
    expectedRegret (literalPolicy i H c m) c m = expectedRegret (legacyPolicy i H c m) c m := by
  induction H generalizing c m with
  | zero => exact minimumRetained_terminalRegret i c m (firstMax m) (firstMax_attains m)
  | succ H ih => simp only [literalPolicy, legacyPolicy, expectedRegret, ih]

theorem raw_legacy_eq_terminalModified (H : ℕ) (c : Precisions A) (m : Means A) :
    rawOfInnovation (legacyPolicy i H c m) c m =
      terminalModified i (rawOfInnovation (optimalFirst H c m) c m) c m := by
  induction H generalizing c m with
  | zero => rfl
  | succ H ih =>
    simp only [legacyPolicy, optimalFirst, rawOfInnovation, terminalModified]
    congr 1
    funext y
    exact ih _ _

theorem literal_raw_run_measurable (H : ℕ) (c : Precisions A) :
    Measurable (fun p : Means A × RawStream A H =>
      run (rawOfInnovation (literalPolicy i H c p.1) c p.1) p.2) := by
  induction H generalizing c with
  | zero => exact (minimumRetained_measurable i).comp measurable_fst
  | succ H ih =>
    let g : (Means A × RawStream A (H + 1)) × A → A := fun p =>
      run (rawOfInnovation (literalPolicy i H (advance c p.2)
        (shift p.1.1 p.2 ((p.1.2 p.2 ⟨0, Nat.zero_lt_succ H⟩ - p.1.1 p.2) /
          ((c p.2 : ℝ) + 1)))) (advance c p.2)
        (shift p.1.1 p.2 ((p.1.2 p.2 ⟨0, Nat.zero_lt_succ H⟩ - p.1.1 p.2) /
          ((c p.2 : ℝ) + 1)))) (tail p.1.2)
    have hg : Measurable g := by
      apply measurable_from_prod_countable_left
      intro a
      have ht : Measurable (fun p : Means A × RawStream A (H + 1) =>
          (shift p.1 a ((p.2 a ⟨0, Nat.zero_lt_succ H⟩ - p.1 a) / ((c a : ℝ) + 1)),
            tail p.2)) := by
        apply Measurable.prodMk
        · apply measurable_pi_lambda
          intro b
          simp only [shift]
          split_ifs <;> fun_prop
        · unfold tail; fun_prop
      exact (ih (advance c a)).comp ht
    exact hg.comp (measurable_id.prodMk ((chooseFirst_measurable H c).comp measurable_fst))

theorem literal_raw_regular (H : ℕ) (c : Precisions A) (m : Means A) :
    Regular (rawOfInnovation (literalPolicy i H c m) c m) := by
  induction H generalizing c m with
  | zero => trivial
  | succ H ih =>
    constructor
    · intro y; exact ih _ _
    · have ht : Measurable (fun p : ℝ × RawStream A H =>
          (shift m (chooseFirst H c m)
            ((p.1 - m (chooseFirst H c m)) / ((c (chooseFirst H c m) : ℝ) + 1)), p.2)) := by
        apply Measurable.prodMk
        · apply measurable_pi_lambda
          intro a
          simp only [shift]
          split_ifs <;> fun_prop
        · exact measurable_snd
      exact (literal_raw_run_measurable i H (advance c (chooseFirst H c m))).comp ht

theorem legacy_raw_regular (H : ℕ) (c : Precisions A) (m : Means A) :
    Regular (rawOfInnovation (legacyPolicy i H c m) c m) := by
  induction H generalizing c m with
  | zero => trivial
  | succ H ih =>
    constructor
    · intro y; exact ih _ _
    · have ht : Measurable (fun p : ℝ × RawStream A H =>
          (shift m (chooseFirst H c m)
            ((p.1 - m (chooseFirst H c m)) / ((c (chooseFirst H c m) : ℝ) + 1)), p.2)) := by
        apply Measurable.prodMk
        · apply measurable_pi_lambda
          intro a
          simp only [shift]
          split_ifs <;> fun_prop
        · exact measurable_snd
      have hmeas := modifiedOptimal_run_measurable i H (advance c (chooseFirst H c m))
      simp_rw [← raw_legacy_eq_terminalModified i] at hmeas
      exact hmeas.comp ht

/-- Terminal reties preserve actual full latent-Gaussian execution risks at
every finite horizon; sampling choices and their raw observation paths agree. -/
theorem literal_rawRisk_eq_legacy (H : ℕ) (c : Precisions A) (m : Means A)
    (hc : ∀ a, 0 < c a) :
    rawRisk (rawOfInnovation (literalPolicy i H c m) c m) c m =
      rawRisk (rawOfInnovation (legacyPolicy i H c m) c m) c m := by
  rw [rawOfInnovation_risk _ _ _ hc (literal_raw_regular i H c m),
    rawOfInnovation_risk _ _ _ hc (legacy_raw_regular i H c m), policies_expectedRegret_eq]

theorem literal_raw_run_excludes (H : ℕ) (c : Precisions A) (m : Means A)
    (real : RawStream A H) : run (rawOfInnovation (literalPolicy i H c m) c m) real ≠ i := by
  induction H generalizing c m with
  | zero => exact minimumRetained_excludes i m
  | succ H ih => exact ih _ _ _

theorem terminalModified_literal_raw (H : ℕ) (c : Precisions A) (m : Means A) :
    terminalModified i (rawOfInnovation (literalPolicy i H c m) c m) c m =
      rawOfInnovation (literalPolicy i H c m) c m := by
  induction H generalizing c m with
  | zero => simp [literalPolicy, rawOfInnovation, terminalModified, terminalChoice,
      minimumRetained_excludes i m]
  | succ H ih =>
    simp only [literalPolicy, rawOfInnovation, terminalModified]
    congr 1
    funext y
    exact ih _ _

def literalForced (H : ℕ) (c : Precisions A) (m : Means A) : Strategy (A := A) (H + 1) :=
  .sample i (fun y =>
    let z := (y - m i) / ((c i : ℝ) + 1)
    rawOfInnovation (literalPolicy i H (advance c i) (shift m i z))
      (advance c i) (shift m i z))

def legacyForced (H : ℕ) (c : Precisions A) (m : Means A) : Strategy (A := A) (H + 1) :=
  .sample i (fun y =>
    let z := (y - m i) / ((c i : ℝ) + 1)
    rawOfInnovation (legacyPolicy i H (advance c i) (shift m i z))
      (advance c i) (shift m i z))

theorem legacyForced_eq_modified_forced (H : ℕ) (c : Precisions A) (m : Means A) :
    legacyForced i H c m = terminalModified i (forcedPolicy i H c m) c m := by
  unfold legacyForced forcedPolicy
  simp only [terminalModified, continuation]
  congr 1
  funext y
  exact raw_legacy_eq_terminalModified i H _ _

theorem literalForced_run_measurable (H : ℕ) (c : Precisions A) (m : Means A) :
    Measurable (run (literalForced i H c m)) := by
  have ht : Measurable (fun p : RawStream A (H + 1) =>
      (shift m i ((p i ⟨0, Nat.zero_lt_succ H⟩ - m i) / ((c i : ℝ) + 1)), tail p)) := by
    apply Measurable.prodMk
    · apply measurable_pi_lambda
      intro a
      simp only [shift]
      split_ifs <;> fun_prop
    · unfold tail; fun_prop
  exact (literal_raw_run_measurable i H (advance c i)).comp ht

theorem literalForced_run_excludes (H : ℕ) (c : Precisions A) (m : Means A)
    (real : RawStream A (H + 1)) : run (literalForced i H c m) real ≠ i :=
  literal_raw_run_excludes i H _ _ _

theorem literalForced_rawRisk_eq_legacy (H : ℕ) (c : Precisions A) (m : Means A)
    (hc : ∀ a, 0 < c a) : rawRisk (literalForced i H c m) c m = rawRisk (legacyForced i H c m) c m := by
  have ht : Measurable (fun p : ℝ × RawStream A H =>
      (shift m i ((p.1 - m i) / ((c i : ℝ) + 1)), p.2)) := by
    apply Measurable.prodMk
    · apply measurable_pi_lambda
      intro a
      simp only [shift]
      split_ifs <;> fun_prop
    · exact measurable_snd
  have hLit := (literal_raw_run_measurable i H (advance c i)).comp ht
  have hLegacy := modifiedOptimal_run_measurable i H (advance c i)
  simp_rw [← raw_legacy_eq_terminalModified i] at hLegacy
  have hLeg := hLegacy.comp ht
  unfold literalForced legacyForced
  rw [rawRisk_succ _ _ _ _ (hc i) hLit, rawRisk_succ _ _ _ _ (hc i) hLeg]
  apply integral_congr_ae
  apply Filter.Eventually.of_forall
  intro z
  simp only [recovered_innovation]
  exact literal_rawRisk_eq_legacy i H (advance c i) (shift m i z) (advance_positive c i hc)

/-- The exact numerical-minimum terminal convention is used unconditionally,
including ties where the old policy already recommended another retained arm. -/
def literalMockPolicy (H : ℕ) (c : Precisions A) (m : Means A)
    (auxiliary : Fin (H + 1) → ℝ) : Strategy (A := A) H :=
  let z := (auxiliary ⟨0, Nat.zero_lt_succ H⟩ - m i) / ((c i : ℝ) + 1)
  replay i (rawOfInnovation (literalPolicy i H (advance c i) (shift m i z))
    (advance c i) (shift m i z)) (advance c i) (shift m i z) (fun t => auxiliary t.succ)

theorem literalMockPolicy_budget (H : ℕ) (c : Precisions A) (m : Means A)
    (auxiliary : Fin (H + 1) → ℝ) (real : RawStream A H) :
    (trace (literalMockPolicy i H c m auxiliary) real).length = H := trace_budget _ _

theorem literalMockPolicy_run_eq_full (H : ℕ) (c : Precisions A) (m : Means A)
    (auxiliary : Fin (H + 1) → ℝ) (real : RawStream A (H + 1)) :
    run (literalMockPolicy i H c m auxiliary) (tail real) =
      run (literalForced i H c m) (substitute i real auxiliary) := by
  simp only [literalMockPolicy, literalForced, run, substitute, if_true]
  rw [replay_run_eq, terminalModified_literal_raw]
  rfl

theorem literalMockPolicy_leaf (c : Precisions A) (m : Means A) :
    replay i (rawOfInnovation (literalPolicy i 0 c m) c m) c m Fin.elim0 =
      .stop (minimumRetained i m) := by
  simp [literalPolicy, rawOfInnovation, replay, terminalChoice, minimumRetained_excludes i m]

theorem literalMockPolicy_run_measurable (H : ℕ) (c : Precisions A) (m : Means A) :
    Measurable (fun p : (Fin (H + 1) → ℝ) × RawStream A H =>
      run (literalMockPolicy i H c m p.1) p.2) := by
  have ht : Measurable (fun p : (Fin (H + 1) → ℝ) × RawStream A H =>
      substitute i (pad p.2) p.1) := by
    apply measurable_pi_lambda
    intro a
    apply measurable_pi_lambda
    intro t
    by_cases ha : a = i
    · simp only [substitute, ha, if_true]
      fun_prop
    · simp only [substitute, if_neg ha, pad]
      refine Fin.cases ?_ (fun r => ?_) t
      · exact measurable_const
      · simp only [Fin.cases_succ]
        fun_prop
  convert (literalForced_run_measurable i H c m).comp ht using 1
  funext p
  have h := literalMockPolicy_run_eq_full i H c m p.1 (pad p.2)
  simpa only [tail_pad] using h

def terminalMeans : (H : ℕ) → Precisions A → Means A → RawStream A H → Means A
  | 0, _, m, _ => m
  | H + 1, c, m, real =>
      let a := chooseFirst H c m
      let z := (real a ⟨0, Nat.zero_lt_succ H⟩ - m a) / ((c a : ℝ) + 1)
      terminalMeans H (advance c a) (shift m a z) (tail real)

/-- Exact raw execution bookkeeping, including the posterior at the terminal
node. This identifies the selected index, not just its maximized mean. -/
theorem literal_raw_run_minimum (H : ℕ) (c : Precisions A) (m : Means A)
    (real : RawStream A H) :
    run (rawOfInnovation (literalPolicy i H c m) c m) real =
      minimumRetained i (terminalMeans H c m real) := by
  induction H generalizing c m with
  | zero => rfl
  | succ H ih => exact ih _ _ _

def mockTerminalMeans (H : ℕ) (c : Precisions A) (m : Means A)
    (auxiliary : Fin (H + 1) → ℝ) (real : RawStream A H) : Means A :=
  let z := (auxiliary ⟨0, Nat.zero_lt_succ H⟩ - m i) / ((c i : ℝ) + 1)
  terminalMeans H (advance c i) (shift m i z)
    (substitute i real (fun t => auxiliary t.succ))

theorem literalMockPolicy_recommends_minimum (H : ℕ) (c : Precisions A) (m : Means A)
    (auxiliary : Fin (H + 1) → ℝ) (real : RawStream A H) :
    run (literalMockPolicy i H c m auxiliary) real =
      minimumRetained i (mockTerminalMeans i H c m auxiliary real) := by
  unfold literalMockPolicy mockTerminalMeans
  rw [replay_run_eq, terminalModified_literal_raw, literal_raw_run_minimum]

/-- The returned arm is exactly the minimum index among all retained
posterior maximizers at the mock algorithm's actual terminal internal state. -/
theorem literalMockPolicy_terminal_spec (H : ℕ) (c : Precisions A) (m : Means A)
    (auxiliary : Fin (H + 1) → ℝ) (real : RawStream A H) :
    let u := mockTerminalMeans i H c m auxiliary real
    let a := run (literalMockPolicy i H c m auxiliary) real
    a ≠ i ∧ u a = otherMaximum i u ∧
      ∀ b, b ≠ i → u b = otherMaximum i u → a ≤ b := by
  dsimp only
  rw [literalMockPolicy_recommends_minimum]
  exact ⟨minimumRetained_excludes i _, minimumRetained_attains i _,
    fun b hb hm => minimumRetained_le i _ b hb hm⟩

theorem replay_literal_run_measurable (H : ℕ) (c : Precisions A) (imaginary : Fin H → ℝ) :
    Measurable (fun p : Means A × RawStream A H =>
      run (replay i (rawOfInnovation (literalPolicy i H c p.1) c p.1) c p.1 imaginary) p.2) := by
  have ht : Measurable (fun p : Means A × RawStream A H =>
      (p.1, substitute i p.2 imaginary)) := by
    apply measurable_fst.prodMk
    apply measurable_pi_lambda
    intro a
    apply measurable_pi_lambda
    intro t
    simp only [substitute]
    split_ifs <;> fun_prop
  convert (literal_raw_run_measurable i H c).comp ht using 1
  funext p
  rw [replay_run_eq, terminalModified_literal_raw]
  rfl

theorem replay_literal_regular (H : ℕ) (c : Precisions A) (m : Means A)
    (imaginary : Fin H → ℝ) :
    Regular (replay i (rawOfInnovation (literalPolicy i H c m) c m) c m imaginary) := by
  induction H generalizing c m with
  | zero => trivial
  | succ H ih =>
    let a := chooseFirst H c m
    by_cases ha : a = i
    · simp only [literalPolicy, rawOfInnovation, replay]
      rw [if_pos ha]
      constructor
      · intro _; exact ih _ _ _
      · exact (replay_literal_run_measurable i H (advance c a) (fun t => imaginary t.succ)).comp
          (measurable_const.prodMk measurable_snd)
    · simp only [literalPolicy, rawOfInnovation, replay]
      rw [if_neg ha]
      constructor
      · intro _; exact ih _ _ _
      · have ht : Measurable (fun p : ℝ × RawStream A H =>
            (shift m a ((p.1 - m a) / ((c a : ℝ) + 1)), p.2)) := by
          apply Measurable.prodMk
          · apply measurable_pi_lambda
            intro b
            simp only [shift]
            split_ifs <;> fun_prop
          · exact measurable_snd
        exact (replay_literal_run_measurable i H (advance c a)
          (fun t => imaginary t.succ)).comp ht

theorem literalMockPolicy_regular (H : ℕ) (c : Precisions A) (m : Means A)
    (auxiliary : Fin (H + 1) → ℝ) : Regular (literalMockPolicy i H c m auxiliary) :=
  replay_literal_regular i H _ _ _

def literalMockLoss (H : ℕ) (c : Precisions A) (m : Means A) : ℝ :=
  ∫ w : MockStream.World A (H + 1) i,
    maxAll (MockStream.theta (H + 1) m i w) -
      MockStream.theta (H + 1) m i w
        (run (literalMockPolicy i H c m (auxiliaryOfBlock i H m w.1.2))
          (tail (MockStream.realStream (H + 1) m i w))) ∂MockStream.law (H + 1) c i

theorem literalMockLoss_eq_mock_risk (H : ℕ) (c : Precisions A) (m : Means A) :
    literalMockLoss i H c m = Junpei.MockSample.risk (H + 1) c m i
      (MockStream.mockStream (H + 1) m i) (run (literalForced i H c m)) := by
  apply integral_congr_ae
  apply Filter.Eventually.of_forall
  intro w
  have he := literalMockPolicy_run_eq_full i H c m (auxiliaryOfBlock i H m w.1.2)
    (MockStream.realStream (H + 1) m i w)
  rw [substitute_real_mock] at he
  dsimp only
  rw [he]

/-- The literal minimum-index mock algorithm has exactly the same actual
latent-world risk as the prior certificate, even when tied arm indices differ. -/
theorem literalMockLoss_eq_original (H : ℕ) (c : Precisions A) (m : Means A)
    (hc : ∀ a, 0 < c a) : literalMockLoss i H c m = mockLoss i H c m := by
  rw [literalMockLoss_eq_mock_risk,
    Junpei.MockSample.risk_coupling (H + 1) c m i _
      (literalForced_run_measurable i H c m) (literalForced_run_excludes i H c m),
    real_risk_eq_rawRisk i (H + 1) c m _ (literalForced_run_measurable i H c m),
    literalForced_rawRisk_eq_legacy i H c m hc, legacyForced_eq_modified_forced,
    mockLoss_eq_modified_real_risk,
    real_risk_eq_rawRisk i (H + 1) c m _ (forcedPolicy_modified_run_measurable i H c m)]

/-- Definition 5's displayed sandwich now holds for the literal minimum
retained argmax terminal rule, with all other sampling actions unchanged. -/
theorem literal_mock_loss_chain (H : ℕ) (c : Precisions A) (m : Means A)
    (hc : ∀ a, 0 < c a) :
    loss H c m ≤ literalMockLoss i H c m ∧
      literalMockLoss i H c m ≤ actionLoss H c m i + exclusion i c m := by
  rw [literalMockLoss_eq_original i H c m hc]
  exact mock_loss_chain i H c m hc

end Removed
end
end Junpei.MockTie
