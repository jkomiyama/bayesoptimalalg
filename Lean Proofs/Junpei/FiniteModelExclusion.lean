import Junpei.FinitePosterior
import Draft0.ThreeArmComparison

open MeasureTheory ProbabilityTheory
open scoped NNReal ENNReal

namespace Junpei.FiniteModelExclusion

open Junpei.FiniteGaussian Junpei.FinitePosterior Draft0.Gaussian

variable {A : Type*} [Fintype A] [Nonempty A] [DecidableEq A]

noncomputable section

omit [Nonempty A] [DecidableEq A] in
theorem posteriorLaw_eq_pi (c : Precisions A) (m : Means A) :
    posteriorLaw c m = Measure.pi (fun a => gaussianReal (m a) (c a)⁻¹) := by
  unfold posteriorLaw centeredLaw
  change (Measure.pi (fun a => gaussianReal 0 (c a)⁻¹)).map
    (fun x a => m a + x a) = _
  rw [Measure.pi_map_pi (fun a => (by fun_prop :
    AEMeasurable (fun x : ℝ => m a + x) (gaussianReal 0 (c a)⁻¹)))]
  simp only [gaussianReal_map_const_add, zero_add]

def pairSlack (c : Precisions A) (m : Means A) (i j : A) : ℝ :=
  hingeLoss (m j - m i) ((c i)⁻¹ + (c j)⁻¹)

omit [Fintype A] [Nonempty A] [DecidableEq A] in
theorem pairSlack_nonnegative (c : Precisions A) (m : Means A) (i j : A) :
    0 ≤ pairSlack c m i j := hingeLoss_nonnegative _ _

omit [Nonempty A] [DecidableEq A] in
theorem pairSlack_eq_integral (c : Precisions A) (m : Means A) (i j : A) (hij : i ≠ j) :
    pairSlack c m i j = ∫ u, max (u i - u j) 0 ∂posteriorLaw c m := by
  have hiInd : iIndepFun (fun a (u : Means A) => u a) (posteriorLaw c m) := by
    rw [posteriorLaw_eq_pi]
    exact iIndepFun_pi (fun _ => aemeasurable_id)
  have hInd := (hiInd.indepFun hij).comp measurable_id measurable_neg
  have hneg : (posteriorLaw c m).map (fun u => -u j) = gaussianReal (-m j) (c j)⁻¹ := by
    have h := gaussianReal_map_neg (μ := m j) (v := (c j)⁻¹)
    rw [← posterior_coordinate_law c m j] at h
    rw [Measure.map_map (f := fun u : Means A => u j) (g := fun x : ℝ => -x)
      (by fun_prop) (measurable_pi_apply j)] at h
    exact h
  have hlaw := gaussianReal_add_gaussianReal_of_indepFun hInd
    (posterior_coordinate_law c m i) hneg
  have hdist : (posteriorLaw c m).map (fun u => u i - u j) =
      gaussianReal (m i - m j) ((c i)⁻¹ + (c j)⁻¹) := by
    simpa only [Function.comp_def, id_eq, Pi.add_apply, sub_eq_add_neg] using hlaw
  have hmap := gaussianReal_map_const_add (μ := (0 : ℝ))
    (v := (c i)⁻¹ + (c j)⁻¹) (m i - m j)
  simp only [zero_add] at hmap
  have hpos : Integrable (fun x : ℝ => max x 0)
      (gaussianReal (m i - m j) ((c i)⁻¹ + (c j)⁻¹)) :=
    (gaussian_id_integrable _ _).pos_part
  rw [← hdist] at hpos
  rw [← integral_map (by fun_prop) hpos.aestronglyMeasurable, hdist,
    ← hmap, integral_map (by fun_prop) (by fun_prop)]
  unfold pairSlack hingeLoss
  congr 1
  funext x
  congr 1
  ring

section Removed
variable (i : A) [Nonempty {a : A // a ≠ i}]

def otherMaximum (m : Means A) : ℝ := maxAll (fun a : {a : A // a ≠ i} => m a)

def otherMinimum (m : Means A) : ℝ := -maxAll (fun a : {a : A // a ≠ i} => -m a)

omit [Nonempty A] [Nonempty {a : A // a ≠ i}] in
theorem restriction_lipschitz :
    LipschitzWith 1 (fun m : Means A => fun a : {a : A // a ≠ i} => m a) := by
  apply LipschitzWith.of_dist_le_mul
  intro m n
  simp only [NNReal.coe_one, one_mul, dist_eq_norm]
  apply (pi_norm_le_iff_of_nonneg (norm_nonneg (m - n))).mpr
  intro a
  have h := (LipschitzWith.eval (a : A) : LipschitzWith 1 (fun m : Means A => m a)).dist_le_mul m n
  simpa only [dist_eq_norm, NNReal.coe_one, one_mul, Pi.sub_apply] using h

omit [Nonempty A] in
theorem otherMaximum_lipschitz : LipschitzWith 1 (otherMaximum i) := by
  simpa only [one_mul] using maxAll_lipschitz.comp (restriction_lipschitz i)

omit [Nonempty A] in
theorem otherMinimum_lipschitz : LipschitzWith 1 (otherMinimum i) := by
  have hn : LipschitzWith 1 (fun m : Means A => fun a : {a : A // a ≠ i} => -m a) :=
    (restriction_lipschitz i).neg
  simpa only [one_mul] using (maxAll_lipschitz.comp hn).neg

def exclusion (c : Precisions A) (m : Means A) : ℝ :=
  ∫ u, max (u i - otherMinimum i u) 0 ∂posteriorLaw c m

omit [Nonempty A] in
theorem exclusion_integrable (c : Precisions A) (m : Means A) :
    Integrable (fun u => max (u i - otherMinimum i u) 0) (posteriorLaw c m) :=
  ((posterior_coordinate_integrable c m i).sub
    (posterior_integrable (otherMinimum_lipschitz i) c m)).pos_part

omit [Nonempty A] in
theorem pairSlack_le_exclusion (c : Precisions A) (m : Means A) (j : A) (hji : j ≠ i) :
    pairSlack c m i j ≤ exclusion i c m := by
  rw [pairSlack_eq_integral c m i j hji.symm]
  apply integral_mono ((posterior_coordinate_integrable c m i).sub
    (posterior_coordinate_integrable c m j)).pos_part (exclusion_integrable i c m)
  intro u
  have hmin : otherMinimum i u ≤ u j := by
    have h := coordinate_le_maxAll (fun a : {a : A // a ≠ i} => -u a) ⟨j, hji⟩
    unfold otherMinimum
    linarith
  exact max_le_max (sub_le_sub_left hmin _) le_rfl

def reducedValue : ℕ → Precisions A → Means A → ℝ
  | 0, _, m => otherMaximum i m
  | k + 1, c, m => maxAll (fun a =>
      ∫ z : ℝ, reducedValue k (advance c a) (shift m a z)
        ∂gaussianReal 0 (innovation c a))

def reducedAction (k : ℕ) (c : Precisions A) (m : Means A) (a : A) : ℝ :=
  ∫ z : ℝ, reducedValue i k (advance c a) (shift m a z)
    ∂gaussianReal 0 (innovation c a)

theorem reducedValue_lipschitz (k : ℕ) (c : Precisions A) :
    LipschitzWith 1 (reducedValue i k c) := by
  induction k generalizing c with
  | zero => exact otherMaximum_lipschitz i
  | succ k ih => exact maxAll_family_lipschitz (fun a =>
      integral_shift_lipschitz (ih (advance c a)) a (innovation c a))

theorem reducedAction_integrable (k : ℕ) (c : Precisions A) (m : Means A) (a : A) :
    Integrable (fun z => reducedValue i k (advance c a) (shift m a z))
      (gaussianReal 0 (innovation c a)) :=
  shifted_integrable (reducedValue_lipschitz i k (advance c a)) m a (innovation c a)

omit [Nonempty A] [Fintype A] in
theorem advance_comm (c : Precisions A) (a b : A) :
    advance (advance c a) b = advance (advance c b) a := by
  ext j
  simp only [advance, add_right_comm]

omit [Nonempty A] [Fintype A] in
theorem shift_comm (m : Means A) (a b : A) (x y : ℝ) :
    shift (shift m a x) b y = shift (shift m b y) a x := by
  ext j
  simp only [shift]
  ring

theorem reducedValue_removed_irrelevant (k : ℕ) (c : Precisions A) (m : Means A) (z : ℝ) :
    reducedValue i k (advance c i) (shift m i z) = reducedValue i k c m := by
  induction k generalizing c m z with
  | zero =>
    unfold reducedValue otherMaximum
    congr 1
    funext a
    simp [shift, a.property]
  | succ k ih =>
    unfold reducedValue
    congr 1
    funext a
    by_cases hai : a = i
    · subst a
      simp_rw [ih]
      simp
    · have hv : innovation (advance c i) a = innovation c a := by
        simp [innovation, advance, hai]
      rw [hv]
      congr 1
      funext x
      rw [advance_comm c i a, shift_comm m i a z x, ih]

theorem reducedAction_removed (k : ℕ) (c : Precisions A) (m : Means A) :
    reducedAction i k c m i = reducedValue i k c m := by
  unfold reducedAction
  simp_rw [reducedValue_removed_irrelevant]
  simp

omit [Fintype A] [Nonempty A] [Nonempty {a : A // a ≠ i}] in
theorem pairSlack_draw_integrable (c : Precisions A) (m : Means A) (j a : A)
    (hji : j ≠ i) :
    Integrable (fun z => pairSlack (advance c a) (shift m a z) i j)
      (gaussianReal 0 (innovation c a)) := by
  by_cases hai : a = i
  · subst a
    simpa [pairSlack, advance, shift, hji, innovation, sub_sub] using
      Draft0.ThreeArmComparison.hingeLoss_shift_integrable (m j - m i)
        (updateVariance (c i)) ((c i + 1)⁻¹ + (c j)⁻¹)
  · by_cases haj : a = j
    · subst a
      simpa [pairSlack, advance, shift, hji, hji.symm, innovation, sub_eq_add_neg,
        add_assoc, add_left_comm, add_comm] using
        Draft0.ThreeArmComparison.hingeLoss_add_integrable (m j - m i)
          (updateVariance (c j)) ((c i)⁻¹ + (c j + 1)⁻¹)
    · simp [pairSlack, advance, shift, Ne.symm hai, Ne.symm haj]

omit [Fintype A] [Nonempty A] [Nonempty {a : A // a ≠ i}] in
theorem pairSlack_tower (c : Precisions A) (m : Means A) (j a : A)
    (hji : j ≠ i) (hc : ∀ b, 0 < c b) :
    (∫ z, pairSlack (advance c a) (shift m a z) i j
      ∂gaussianReal 0 (innovation c a)) = pairSlack c m i j := by
  by_cases hai : a = i
  · subst a
    have h := Draft0.ThreeArmComparison.hingeLoss_shift_tower (m j - m i)
      (updateVariance (c i)) ((c i + 1)⁻¹ + (c j)⁻¹)
    have hv := posteriorVariance_after_draw (c i) (c j) (hc i)
    unfold posteriorVariance at hv
    rw [← hv] at h
    simpa [pairSlack, advance, shift, hji, innovation, posteriorVariance, sub_sub] using h
  · by_cases haj : a = j
    · subst a
      have h := Draft0.ThreeArmComparison.hingeLoss_add_tower (m j - m i)
        (updateVariance (c j)) ((c i)⁻¹ + (c j + 1)⁻¹)
      have hv := posteriorVariance_after_draw (c j) (c i) (hc j)
      unfold posteriorVariance at hv
      rw [add_comm (c i)⁻¹ (c j + 1)⁻¹, ← hv,
        add_comm (c j)⁻¹ (c i)⁻¹] at h
      simpa [pairSlack, advance, shift, hji, hji.symm, innovation, sub_eq_add_neg,
        add_assoc, add_left_comm, add_comm] using h
    · simp [pairSlack, advance, shift, Ne.symm hai, Ne.symm haj]

theorem value_comparison (k : ℕ) (c : Precisions A) (m : Means A) (j : A)
    (hji : j ≠ i) (hc : ∀ a, 0 < c a) :
    reducedValue i k c m ≤ value k c m ∧
      value k c m ≤ reducedValue i k c m + pairSlack c m i j := by
  induction k generalizing c m with
  | zero =>
    constructor
    · apply maxAll_le
      intro a
      exact coordinate_le_maxAll m a
    · apply maxAll_le
      intro a
      by_cases hai : a = i
      · subst a
        have hm := coordinate_le_maxAll (fun b : {a : A // a ≠ i} => m b) ⟨j, hji⟩
        have hp := Draft0.ThreeArmComparison.positivePart_le_hingeLoss (m j - m i)
          ((c i)⁻¹ + (c j)⁻¹)
        have hmi := le_max_left (-(m j - m i)) (0 : ℝ)
        change m i ≤ otherMaximum i m + pairSlack c m i j
        unfold otherMaximum pairSlack
        linarith
      · have hm := coordinate_le_maxAll (fun b : {a : A // a ≠ i} => m b) ⟨a, hai⟩
        have hp := pairSlack_nonnegative c m i j
        change m a ≤ otherMaximum i m + pairSlack c m i j
        unfold otherMaximum
        linarith
  | succ k ih =>
    constructor
    · apply maxAll_le
      intro a
      exact (integral_mono (reducedAction_integrable i k c m a)
        (action_integrable k c m a) (fun z =>
          (ih (advance c a) (shift m a z) (advance_positive c a hc)).1)).trans
        (action_le_value k c m a)
    · apply maxAll_le
      intro a
      have hupper := integral_mono (action_integrable k c m a)
        ((reducedAction_integrable i k c m a).add (pairSlack_draw_integrable i c m j a hji))
        (fun z => (ih (advance c a) (shift m a z) (advance_positive c a hc)).2)
      dsimp only [Pi.add_apply] at hupper
      rw [integral_add (reducedAction_integrable i k c m a)
        (pairSlack_draw_integrable i c m j a hji), pairSlack_tower i c m j a hji hc] at hupper
      exact hupper.trans (add_le_add_right (coordinate_le_maxAll
        (reducedAction i k c m) a) _)

/-- Lemma12 for arbitrary finite arm sets, real positive posterior precisions,
every arm, and every finite remaining horizon. The bound is the actual
posterior expectation of the arm's advantage over the minimum other mean. -/
theorem lemma12_finite (k : ℕ) (c : Precisions A) (m : Means A) (j : A)
    (hji : j ≠ i) (hc : ∀ a, 0 < c a) :
    loss k c m - actionLoss k c m i ≤ exclusion i c m := by
  rw [actionLoss_eq k c m i (hc i)]
  unfold loss
  have hupper := integral_mono (action_integrable k c m i)
    ((reducedAction_integrable i k c m i).add (pairSlack_draw_integrable i c m j i hji))
    (fun z => (value_comparison i k (advance c i) (shift m i z) j hji
      (advance_positive c i hc)).2)
  dsimp only [Pi.add_apply] at hupper
  rw [integral_add (reducedAction_integrable i k c m i)
    (pairSlack_draw_integrable i c m j i hji), pairSlack_tower i c m j i hji hc] at hupper
  change action k c m i ≤ reducedAction i k c m i + pairSlack c m i j at hupper
  rw [reducedAction_removed] at hupper
  have hlower := (value_comparison i k c m j hji hc).1
  have hR := pairSlack_le_exclusion i c m j hji
  linarith

end Removed

instance others_nonempty [Nontrivial A] (i : A) : Nonempty {a : A // a ≠ i} := by
  obtain ⟨j, hji⟩ := exists_ne i
  exact ⟨⟨j, hji⟩⟩

/-- Nontriviality of a finite arm type is exactly the requirement of at least
two arms. Every removed arm is covered, without fixing an anchor in the bound. -/
theorem lemma12_any_finite_arm [Nontrivial A] (k : ℕ) (c : Precisions A)
    (m : Means A) (i : A) (hc : ∀ a, 0 < c a) :
    loss k c m - actionLoss k c m i ≤ exclusion i c m := by
  obtain ⟨j, hji⟩ := exists_ne i
  exact lemma12_finite i k c m j hji hc

end
end Junpei.FiniteModelExclusion
