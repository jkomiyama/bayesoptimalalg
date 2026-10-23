import Junpei.Exclusion

open MeasureTheory ProbabilityTheory
open scoped NNReal ENNReal

namespace Junpei.ExclusionLoss

open Draft0.BayesianModel Draft0.PosteriorLaw Draft0.PosteriorRegret
open Draft0.ThreeArmComparison Junpei.Exclusion

noncomputable section

def twoTerminal (m : Means) : ℝ := max (m 0) (m 1)

theorem twoTerminal_lipschitz : LipschitzWith 1 twoTerminal := by
  have h (i : Arm) : LipschitzWith 1 (fun m : Means => m i) := LipschitzWith.eval i
  simpa only [max_self] using (h 0).max (h 1)

/-- Actual posterior expectation of the best latent mean among arms1 and2. -/
def twoPotential (c : Counts) (m : Means) : ℝ :=
  ∫ u : Means, twoTerminal u ∂posteriorLaw c m

theorem twoTerminal_integrable (c : Counts) (m : Means) :
    Integrable twoTerminal (posteriorLaw c m) := by
  have h := (posterior_coordinate_integrable c m 1).add (pair_integrable c m 0 1)
  convert h using 1
  funext u
  dsimp [twoTerminal]
  by_cases hu : u 0 ≤ u 1
  · rw [max_eq_right hu, max_eq_right (by linarith : u 0 - u 1 ≤ 0), add_zero]
  · rw [max_eq_left (le_of_not_ge hu), max_eq_left (by linarith : 0 ≤ u 0 - u 1)]
    ring

theorem twoPotential_eq_smooth (c : Counts) :
    twoPotential c = smooth (smooth twoTerminal 1 (c 1 : ℝ≥0)⁻¹) 0 (c 0 : ℝ≥0)⁻¹ := by
  funext m
  unfold twoPotential posteriorLaw
  rw [integral_map (tripleShift_measurable m).aemeasurable
    twoTerminal_lipschitz.continuous.measurable.aestronglyMeasurable]
  have hi := triple_shift_integrable twoTerminal_lipschitz m c
  simp [twoTerminal, tripleShift, shift, Fin.ext_iff] at hi ⊢
  unfold centeredTripleLaw at hi ⊢
  rw [integral_prod _ hi]
  dsimp only [Prod.fst, Prod.snd]
  have hf (x : ℝ) : (∫ y : ℝ × ℝ, max (m 0 + x) (m 1 + y.1)
      ∂(gaussianReal 0 (c 1 : ℝ≥0)⁻¹).prod (gaussianReal 0 (c 2 : ℝ≥0)⁻¹)) =
      ∫ y : ℝ, max (m 0 + x) (m 1 + y) ∂gaussianReal 0 (c 1 : ℝ≥0)⁻¹ := by
    simpa using integral_fun_fst (μ := gaussianReal 0 (c 1 : ℝ≥0)⁻¹)
      (ν := gaussianReal 0 (c 2 : ℝ≥0)⁻¹) (fun y : ℝ => max (m 0 + x) (m 1 + y))
  simp_rw [hf]
  simp [smooth, twoTerminal, shift]

theorem twoPotential_lipschitz (c : Counts) : LipschitzWith 1 (twoPotential c) := by
  rw [twoPotential_eq_smooth]
  exact smooth_lipschitz (smooth_lipschitz twoTerminal_lipschitz _ _) _ _

theorem twoPotential_tower (c : Counts) (i : Arm) (hi : i ≠ 2) (hc : 0 < c i) :
    smooth (twoPotential (advance c i)) i (innovation c i) = twoPotential c := by
  have hsplit := count_variance_split (c i) hc
  fin_cases i
  · change smooth (twoPotential (advance c 0)) 0 (innovation c 0) = twoPotential c
    rw [twoPotential_eq_smooth, twoPotential_eq_smooth]
    norm_num [advance, innovation] at hsplit ⊢
    rw [smooth_add (smooth_lipschitz twoTerminal_lipschitz _ _), ← hsplit]
  · change smooth (twoPotential (advance c 1)) 1 (innovation c 1) = twoPotential c
    rw [twoPotential_eq_smooth, twoPotential_eq_smooth]
    norm_num [advance, innovation] at hsplit ⊢
    rw [smooth_comm (smooth_lipschitz twoTerminal_lipschitz _ _) 1 0]
    congr 1
    rw [smooth_add twoTerminal_lipschitz, ← hsplit]
  · exact False.elim (hi rfl)

def twoLoss (k : ℕ) (c : Counts) (m : Means) : ℝ := twoPotential c m - twoValue k c m

def twoActionLoss (k : ℕ) (c : Counts) (m : Means) (i : Arm) : ℝ :=
  ∫ z : ℝ, twoLoss k (advance c i) (shift m i z) ∂gaussianReal 0 (innovation c i)

theorem twoActionLoss_eq (k : ℕ) (c : Counts) (m : Means) (i : Arm)
    (hi : i ≠ 2) (hc : 0 < c i) :
    twoActionLoss k c m i = twoPotential c m - twoAction k c m i := by
  unfold twoActionLoss twoLoss twoAction
  rw [integral_sub (shifted_integrable (twoPotential_lipschitz _) _ _ _)
    (twoAction_integrable k c m i)]
  have ht := congrFun (twoPotential_tower c i hi hc) m
  exact congrArg (fun x : ℝ => x - ∫ z : ℝ, twoValue k (advance c i) (shift m i z)
    ∂gaussianReal 0 (innovation c i)) ht

theorem potential_difference_bounds (c : Counts) (m : Means) :
    0 ≤ potential c m - twoPotential c m ∧
      potential c m - twoPotential c m ≤ exclusion c m 2 := by
  have hlo := integral_mono (twoTerminal_integrable c m) (posterior_terminal_integrable c m)
    (fun u => max_le_max le_rfl (le_max_left _ _))
  have hup := integral_mono (posterior_terminal_integrable c m)
    ((twoTerminal_integrable c m).add (pair_integrable c m 2 0)) (fun u => by
      change max (u 0) (max (u 1) (u 2)) ≤ max (u 0) (u 1) + max (u 2 - u 0) 0
      have h0 := le_max_left (u 0) (u 1)
      have h1 := le_max_right (u 0) (u 1)
      have hd := le_max_left (u 2 - u 0) 0
      have hn := le_max_right (u 2 - u 0) 0
      apply max_le
      · linarith
      · apply max_le <;> linarith)
  simp only [Pi.add_apply] at hup
  rw [integral_add (twoTerminal_integrable c m) (pair_integrable c m 2 0)] at hup
  rw [← potential_eq_posterior_integral] at hlo hup
  change twoPotential c m ≤ potential c m at hlo
  change potential c m ≤ twoPotential c m + pairIntegral c m 2 0 at hup
  have hs := pairIntegral_le_exclusion c m 2 0 (by decide)
  constructor <;> linarith

/-- Revised Lemma13's optimal-loss comparison, with the actual latent posterior potentials. -/
theorem lemma13_loss (k : ℕ) (c : Counts) (m : Means) (hc : ∀ i, 0 < c i) :
    |loss k c m - twoLoss k c m| ≤ exclusion c m 2 := by
  have hd := potential_difference_bounds c m
  have hv := value_comparison k c m hc
  have hs := stateSlack_le_exclusion c m
  rw [abs_le]
  unfold loss twoLoss
  constructor <;> linarith

/-- Revised Lemma13's forced-action-loss comparison for either remaining arm. -/
theorem lemma13_actionLoss (k : ℕ) (c : Counts) (m : Means) (i : Arm)
    (hi : i ≠ 2) (hc : ∀ j, 0 < c j) :
    |actionLoss k c m i - twoActionLoss k c m i| ≤ exclusion c m 2 := by
  rw [actionLoss_eq k c m i (hc i), twoActionLoss_eq k c m i hi (hc i), abs_le]
  have hd := potential_difference_bounds c m
  have ha := action_comparison k c m i hc
  have hs := stateSlack_le_exclusion c m
  constructor <;> linarith

end

end Junpei.ExclusionLoss
