import Mathlib.Analysis.Complex.ExponentialBounds
import Mathlib.Data.Real.Sqrt
import Mathlib.Tactic.FinCases
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.NormNum
import Mathlib.Tactic.Positivity
import Draft0.Rates

/-!
Corrected deterministic induction and terminal recommendation step in Theorem 2.
The action admissibility premises are explicit. Establishing them for the actual
Bayes optimal policy still requires stochastic Bellman comparison lemmas.
In particular the terminal premise is separate, never inferred from t < T.
-/

namespace Draft0.MainBridge

abbrev Arm := Fin 3
abbrev ArmCounts := Arm → ℕ

def advance (c : ArmCounts) (a : Arm) : ArmCounts :=
  fun i => c i + if i = a then 1 else 0

def invariant (q s : ℕ) (c : ArmCounts) : Prop :=
  c 0 ≤ q ∧ c 1 ≤ q ∧ c 2 = 1 ∧ c 0 + c 1 + c 2 = s

def admissible (q : ℕ) (c : ArmCounts) (a : Arm) : Prop :=
  a ≠ 2 ∧ (a = 0 → c 0 < q) ∧ (a = 1 → c 1 < q)

theorem advance_invariant (q s : ℕ) (c : ArmCounts) (a : Arm)
    (hc : invariant q s c) (ha : admissible q c a) :
    invariant q (s + 1) (advance c a) := by
  fin_cases a <;> simp [invariant, admissible, advance] at * <;> omega

/-- The full round induction, including an independent last-round premise. -/
theorem invariant_all_rounds (q : ℕ) (hq : 2 ≤ q)
    (counts : ℕ → ArmCounts) (action : ℕ → Arm)
    (hstart : ∀ i, counts 3 i = 1)
    (hupdate : ∀ s, 3 ≤ s → s < 2 * q + 1 →
      counts (s + 1) = advance (counts s) (action (s + 1)))
    (hpreterminal : ∀ s, 3 ≤ s → s < 2 * q →
      invariant q s (counts s) → admissible q (counts s) (action (s + 1)))
    (hterminal : invariant q (2 * q) (counts (2 * q)) →
      admissible q (counts (2 * q)) (action (2 * q + 1))) :
    ∀ s, 3 ≤ s → s ≤ 2 * q + 1 → invariant q s (counts s) := by
  intro s
  induction s using Nat.strong_induction_on with
  | h s ih =>
    intro hlo hhi
    by_cases hbase : s = 3
    · subst s
      change counts 3 0 ≤ q ∧ counts 3 1 ≤ q ∧ counts 3 2 = 1 ∧
        counts 3 0 + counts 3 1 + counts 3 2 = 3
      rw [hstart 0, hstart 1, hstart 2]
      omega
    · have hprev : 3 ≤ s - 1 := by omega
      have hs : s - 1 + 1 = s := by omega
      have hc := ih (s - 1) (by omega) hprev (by omega)
      have ha : admissible q (counts (s - 1)) (action s) := by
        by_cases hlast : s = 2 * q + 1
        · have heq : s - 1 = 2 * q := by omega
          rw [heq] at hc ⊢
          rw [hlast]
          exact hterminal hc
        · have ha' := hpreterminal (s - 1) hprev (by omega) hc
          simpa only [hs] using ha'
      have hu := hupdate (s - 1) hprev (by omega)
      simp only [hs] at hu
      rw [hu]
      simpa only [hs] using advance_invariant q (s - 1) (counts (s - 1)) (action s) hc ha

theorem invariant_final_counts (q : ℕ) (c : ArmCounts)
    (hc : invariant q (2 * q + 1) c) :
    c 0 = q ∧ c 1 = q ∧ c 2 = 1 := by
  unfold invariant at hc
  omega

/-- Safe numerical separation from the printed X and final Y events. -/
theorem terminal_underestimation_separation (T : ℕ) (hT : 5 ≤ T)
    (CU mean₁ mean₃ : ℝ) (hCU : 30 ≤ CU)
    (hY : |mean₁| ≤ 1 / (T : ℝ) ^ 2)
    (hX : mean₃ ≤ -CU * Real.sqrt (Real.log (T : ℝ))) : mean₃ < mean₁ := by
  have hTr : 1 ≤ (T : ℝ) := by exact_mod_cast (show 1 ≤ T by omega)
  have hTtwo : (2 : ℝ) ≤ (T : ℝ) := by exact_mod_cast (show 2 ≤ T by omega)
  have hlog : (1 / 4 : ℝ) < Real.log (T : ℝ) := by
    have htwo : (1 / 4 : ℝ) < Real.log 2 := by linarith [Real.log_two_gt_d9]
    exact htwo.trans_le (Real.log_le_log (by norm_num) hTtwo)
  have hsqrt : (1 / 2 : ℝ) < Real.sqrt (Real.log (T : ℝ)) := by
    apply (Real.lt_sqrt (by norm_num)).mpr
    nlinarith
  have hsmall : 1 / (T : ℝ) ^ 2 ≤ 1 := by
    apply (div_le_one (by positivity)).mpr
    nlinarith
  have hmean := (abs_le.mp hY).1
  have hprod : 1 < CU * Real.sqrt (Real.log (T : ℝ)) := by
    nlinarith
  linarith

theorem greedy_recommendation_wrong (means : Arm → ℝ) (chosen : Arm)
    (hgreedy : ∀ i, means i ≤ means chosen) (hsep : means 2 < means 0) :
    chosen ≠ 2 := by
  intro hc
  have h := hgreedy 0
  rw [hc] at h
  exact not_le_of_gt hsep h

theorem final_recommendation_wrong (T : ℕ) (hT : 5 ≤ T) (CU : ℝ) (hCU : 30 ≤ CU)
    (means : Arm → ℝ) (chosen : Arm)
    (hgreedy : ∀ i, means i ≤ means chosen)
    (hY : |means 0| ≤ 1 / (T : ℝ) ^ 2)
    (hX : means 2 ≤ -CU * Real.sqrt (Real.log (T : ℝ))) : chosen ≠ 2 := by
  exact greedy_recommendation_wrong means chosen hgreedy
    (terminal_underestimation_separation T hT CU (means 0) (means 2) hCU hY hX)

/-- All deterministic conditions needed by the repaired main argument on one
bad sample path. The terminal action condition is deliberately a separate field. -/
structure BadPathConditions (q : ℕ) (counts : ℕ → ArmCounts) (action : ℕ → Arm)
    (means : Arm → ℕ → ℝ) (chosen : Arm) (CU : ℝ) : Prop where
  initial : ∀ i, counts 3 i = 1
  update : ∀ s, 3 ≤ s → s < 2 * q + 1 →
    counts (s + 1) = advance (counts s) (action (s + 1))
  beforeTerminal : ∀ s, 3 ≤ s → s < 2 * q →
    invariant q s (counts s) → admissible q (counts s) (action (s + 1))
  atTerminal : invariant q (2 * q) (counts (2 * q)) →
    admissible q (counts (2 * q)) (action (2 * q + 1))
  firstArmClose : |means 0 q| ≤ 1 / ((2 * q + 1 : ℕ) : ℝ) ^ 2
  thirdArmLow : means 2 1 ≤ -CU * Real.sqrt (Real.log ((2 * q + 1 : ℕ) : ℝ))
  greedy : ∀ i, means i (counts (2 * q + 1) i) ≤
    means chosen (counts (2 * q + 1) chosen)

theorem bad_path_final_counts_and_wrong (q : ℕ) (hq : 2 ≤ q)
    (counts : ℕ → ArmCounts) (action : ℕ → Arm) (means : Arm → ℕ → ℝ)
    (chosen : Arm) (CU : ℝ) (hCU : 30 ≤ CU)
    (h : BadPathConditions q counts action means chosen CU) :
    counts (2 * q + 1) 0 = q ∧ counts (2 * q + 1) 1 = q ∧
      counts (2 * q + 1) 2 = 1 ∧ chosen ≠ 2 := by
  have hinv := invariant_all_rounds q hq counts action h.initial h.update
    h.beforeTerminal h.atTerminal (2 * q + 1) (by omega) (by omega)
  have hc := invariant_final_counts q (counts (2 * q + 1)) hinv
  refine ⟨hc.1, hc.2.1, hc.2.2, ?_⟩
  apply final_recommendation_wrong (2 * q + 1) (by omega) CU hCU
    (fun i => means i (counts (2 * q + 1) i)) chosen h.greedy
  · simpa only [hc.1] using h.firstArmClose
  · simpa only [hc.2.2] using h.thirdArmLow

/-- A genuine expected-regret consequence. The unproved policy-comparison
obligations are the explicit BadPathConditions hypothesis, not hidden axioms. -/
theorem bad_event_regret_lower_bound {Ω : Type*} [MeasurableSpace Ω]
    (μ : MeasureTheory.Measure Ω) [MeasureTheory.IsProbabilityMeasure μ]
    (q : ℕ) (hq : 2 ≤ q) (CU gap : ℝ) (hCU : 30 ≤ CU) (hgap : 0 ≤ gap)
    (event : Set Ω) (counts : Ω → ℕ → ArmCounts) (action : Ω → ℕ → Arm)
    (means : Ω → Arm → ℕ → ℝ) (recommend : Ω → Arm)
    (hwrongMeas : MeasurableSet {ω | recommend ω ≠ 2})
    (hpaths : ∀ ω ∈ event,
      BadPathConditions q (counts ω) (action ω) (means ω) (recommend ω) CU) :
    gap * μ.real event ≤ Draft0.Rates.fixedGapRegret μ {ω | recommend ω ≠ 2} gap := by
  apply Draft0.Rates.fixedGapRegret_event_lower_bound μ hwrongMeas ?_ hgap
  intro ω hω
  exact (bad_path_final_counts_and_wrong q hq (counts ω) (action ω) (means ω)
    (recommend ω) CU hCU (hpaths ω hω)).2.2.2

end Draft0.MainBridge
