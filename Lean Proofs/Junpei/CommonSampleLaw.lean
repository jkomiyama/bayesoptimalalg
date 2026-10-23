import Junpei.Coupling19Saturated
import Mathlib.Probability.HasLaw

open MeasureTheory ProbabilityTheory Set
open scoped NNReal ENNReal

namespace Junpei.CommonSample

open Draft0.Counts Draft0.Gaussian Draft0.TwoArmBellman Draft0.TwoArmGain
open Junpei.TwoArmOptimal

noncomputable section

theorem centered_pair_difference_law (c : Counts) :
    HasLaw (fun x : Fin 2 → ℝ => x 0 - x 1) (gaussianReal 0 (variance c))
      (FinitePosterior.centeredLaw (pairPrecisions c.n1 c.n2)) := by
  let μ := FinitePosterior.centeredLaw (pairPrecisions c.n1 c.n2)
  have h0 : HasLaw (fun x : Fin 2 → ℝ => x 0)
      (gaussianReal 0 (c.n1 : ℝ≥0)⁻¹) μ := by
    exact (measurePreserving_eval (fun a : Fin 2 => gaussianReal 0
      ((pairPrecisions c.n1 c.n2) a)⁻¹) 0).hasLaw
  have h1 : HasLaw (fun x : Fin 2 → ℝ => -x 1)
      (gaussianReal 0 (c.n2 : ℝ≥0)⁻¹) μ := by
    have he : HasLaw (fun x : Fin 2 → ℝ => x 1)
        (gaussianReal 0 (c.n2 : ℝ≥0)⁻¹) μ :=
      (measurePreserving_eval (fun a : Fin 2 => gaussianReal 0
        ((pairPrecisions c.n1 c.n2) a)⁻¹) 1).hasLaw
    have hn : HasLaw (fun x : ℝ => -x) (gaussianReal 0 (c.n2 : ℝ≥0)⁻¹)
        (gaussianReal 0 (c.n2 : ℝ≥0)⁻¹) := by
      constructor
      · exact (by fun_prop : AEMeasurable (fun x : ℝ => -x) (gaussianReal 0 (c.n2 : ℝ≥0)⁻¹))
      · simpa using gaussianReal_map_neg (μ := (0 : ℝ)) (v := (c.n2 : ℝ≥0)⁻¹)
    exact hn.fun_comp he
  have hind : iIndepFun (fun a : Fin 2 => fun x : Fin 2 → ℝ => x a) μ :=
    iIndepFun_pi (fun _ => measurable_id.aemeasurable)
  have hInd : IndepFun (fun x : Fin 2 → ℝ => x 0) (fun x : Fin 2 → ℝ => -x 1) μ :=
    (hind.indepFun (by decide : (0 : Fin 2) ≠ 1)).comp measurable_id (by fun_prop)
  have h := hInd.hasLaw_fun_add h0 h1
  rw [gaussianReal_conv_gaussianReal, zero_add] at h
  simpa only [sub_eq_add_neg, Draft0.TwoArmBellman.variance, posteriorVariance] using h

theorem potential_eq_gaussian (c : Counts) (m1 m2 : ℝ) :
    potential c m1 m2 = m2 + hingeLoss (m2 - m1) (variance c) := by
  let cp := pairPrecisions c.n1 c.n2
  let μ := FinitePosterior.centeredLaw cp
  letI : IsProbabilityMeasure μ := by dsimp [μ]; infer_instance
  have hi1 : Integrable (fun x : Fin 2 → ℝ => x 1) μ :=
    FinitePosterior.centered_coordinate_integrable cp 1
  have hi0 : Integrable (fun x : Fin 2 → ℝ => x 0) μ :=
    FinitePosterior.centered_coordinate_integrable cp 0
  have hip : Integrable (fun x : Fin 2 → ℝ => max (x 0 - x 1 - (m2 - m1)) 0) μ :=
    ((hi0.sub hi1).sub (integrable_const (m2 - m1))).pos_part
  have hf (x : Fin 2 → ℝ) : FiniteGaussian.maxAll (pairMeans m1 m2 + x) =
      (m2 + x 1) + max (x 0 - x 1 - (m2 - m1)) 0 := by
    have he : pairMeans m1 m2 + x = pairMeans (m1 + x 0) (m2 + x 1) := by
      ext a
      fin_cases a <;> simp [pairMeans]
    rw [he, maxAll_pair]
    by_cases h : m1 + x 0 ≤ m2 + x 1
    · rw [max_eq_right h, max_eq_right (by linarith : x 0 - x 1 - (m2 - m1) ≤ 0), add_zero]
    · rw [max_eq_left (le_of_not_ge h), max_eq_left (by linarith : 0 ≤ x 0 - x 1 - (m2 - m1))]
      ring
  unfold potential FinitePosterior.potential FinitePosterior.expectation
  simp_rw [hf]
  change (∫ x : Fin 2 → ℝ, (m2 + x 1) + max (x 0 - x 1 - (m2 - m1)) 0 ∂μ) = _
  have him : Integrable (fun x : Fin 2 → ℝ => m2 + x 1) μ := (integrable_const m2).add hi1
  rw [integral_add him hip, integral_add (integrable_const m2) hi1, integral_const]
  have hm1 : (∫ x : Fin 2 → ℝ, x 1 ∂μ) = 0 := by
    have hcoord : HasLaw (fun x : Fin 2 → ℝ => x 1) (gaussianReal 0 (cp 1)⁻¹) μ :=
      (measurePreserving_eval (fun a => gaussianReal 0 (cp a)⁻¹) 1).hasLaw
    exact hcoord.integral_eq.trans integral_id_gaussianReal
  rw [hm1]
  simp only [measureReal_univ_eq_one, one_smul, add_zero]
  congr 1
  have h := (centered_pair_difference_law c).integral_comp
    (f := fun x : ℝ => max (x - (m2 - m1)) 0) (by fun_prop)
  exact h

theorem terminalRisk_eq_posteriorLoss (c : Counts) (m1 m2 : ℝ) :
    terminalRisk c m1 m2 = posteriorLoss (m1 - m2) (variance c) := by
  unfold terminalRisk
  rw [potential_eq_gaussian]
  have h := hingeLoss_sub_abs (m2 - m1) (variance c)
  change m2 + hingeLoss (m2 - m1) (variance c) - max m1 m2 = hingeLoss |m1 - m2| (variance c)
  rw [abs_sub_comm m1 m2]
  by_cases he : m1 ≤ m2
  · rw [max_eq_right he] at ⊢
    rw [max_eq_right (by linarith : -(m2 - m1) ≤ 0)] at h
    linarith only [h]
  · rw [max_eq_left (le_of_not_ge he)]
    rw [max_eq_left (by linarith : 0 ≤ -(m2 - m1))] at h
    linarith only [h]

def blockSum {M : Type*} [AddCommMonoid M] (N n b : ℕ) (f : Fin N → M) : M :=
  ∑ j : Fin N, if n ≤ j.val ∧ j.val < b then f j else 0

theorem blockSum_empty {M : Type*} [AddCommMonoid M] (N n : ℕ) (f : Fin N → M) :
    blockSum N n n f = 0 := by
  apply Finset.sum_eq_zero
  intro j _
  rw [if_neg (by omega : ¬ (n ≤ j.val ∧ j.val < n))]

theorem blockSum_head {M : Type*} [AddCommMonoid M] (N n b : ℕ) (f : Fin N → M)
    (hn : n < b) (hb : b ≤ N) :
    blockSum N n b f = f ⟨n, by omega⟩ + blockSum N (n + 1) b f := by
  have hsplit (j : Fin N) : (if n ≤ j.val ∧ j.val < b then f j else 0) =
      (if j = ⟨n, by omega⟩ then f j else 0) +
      (if n + 1 ≤ j.val ∧ j.val < b then f j else 0) := by
    by_cases he : j.val = n
    · have he' : j = ⟨n, by omega⟩ := Fin.ext he
      rw [he']
      simp [hn]
    · have he' : j ≠ ⟨n, by omega⟩ := fun h => he (congrArg Fin.val h)
      have heq : (n ≤ j.val ∧ j.val < b) ↔ (n + 1 ≤ j.val ∧ j.val < b) := by omega
      simp only [if_neg he', zero_add, heq]
  unfold blockSum
  simp_rw [hsplit]
  rw [Finset.sum_add_distrib]
  simp

theorem finalCounts_monotone (s : List Bool) (c : Counts) :
    c.n1 ≤ (finalCounts s c).n1 ∧ c.n2 ≤ (finalCounts s c).n2 := by
  induction s generalizing c with
  | nil => exact ⟨le_rfl, le_rfl⟩
  | cons i s ih =>
    have hh := ih (step i c)
    cases i <;> simp only [finalCounts, step, Bool.false_eq_true, ↓reduceIte, leftStep, rightStep] at * <;> omega

theorem terminalMeans_eq_blockSum (N : ℕ) (s : List Bool) (c : Counts)
    (m : ℝ × ℝ) (x : Bool × Fin N → ℝ)
    (hN1 : (finalCounts s c).n1 ≤ N) (hN2 : (finalCounts s c).n2 ≤ N) :
    terminalMeans s c m (finiteStream N x) =
      (m.1 + blockSum N c.n1 (finalCounts s c).n1 (fun j => x (true, j)),
       m.2 + blockSum N c.n2 (finalCounts s c).n2 (fun j => x (false, j))) := by
  induction s generalizing c m with
  | nil => simp [terminalMeans, finalCounts, blockSum_empty]
  | cons i s ih =>
    cases i
    · change (finalCounts s (rightStep c)).n1 ≤ N at hN1
      change (finalCounts s (rightStep c)).n2 ≤ N at hN2
      simp only [terminalMeans, finalCounts, step, commonUpdate, Bool.false_eq_true, ↓reduceIte]
      rw [ih (rightStep c) _ hN1 hN2]
      have hm := finalCounts_monotone s (rightStep c)
      have hn : c.n2 < (finalCounts s (rightStep c)).n2 := by
        change c.n1 ≤ (finalCounts s (rightStep c)).n1 ∧ c.n2 + 1 ≤ (finalCounts s (rightStep c)).n2 at hm
        omega
      have hcn : c.n2 < N := by omega
      change (m.1 + blockSum N c.n1 (finalCounts s (rightStep c)).n1 _,
        (m.2 + finiteStream N x false c.n2) + blockSum N (c.n2 + 1) (finalCounts s (rightStep c)).n2 _) =
        (m.1 + blockSum N c.n1 (finalCounts s (rightStep c)).n1 _,
        m.2 + blockSum N c.n2 (finalCounts s (rightStep c)).n2 _)
      simp only [finiteStream, dif_pos hcn]
      rw [blockSum_head N c.n2 (finalCounts s (rightStep c)).n2 _ hn hN2]
      congr 1
      ring
    · change (finalCounts s (leftStep c)).n1 ≤ N at hN1
      change (finalCounts s (leftStep c)).n2 ≤ N at hN2
      simp only [terminalMeans, finalCounts, step, commonUpdate, ↓reduceIte]
      rw [ih (leftStep c) _ hN1 hN2]
      have hm := finalCounts_monotone s (leftStep c)
      have hn : c.n1 < (finalCounts s (leftStep c)).n1 := by
        change c.n1 + 1 ≤ (finalCounts s (leftStep c)).n1 ∧ c.n2 ≤ (finalCounts s (leftStep c)).n2 at hm
        omega
      have hcn : c.n1 < N := by omega
      change ((m.1 + finiteStream N x true c.n1) + blockSum N (c.n1 + 1) (finalCounts s (leftStep c)).n1 _,
        m.2 + blockSum N c.n2 (finalCounts s (leftStep c)).n2 _) =
        (m.1 + blockSum N c.n1 (finalCounts s (leftStep c)).n1 _,
        m.2 + blockSum N c.n2 (finalCounts s (leftStep c)).n2 _)
      simp only [finiteStream, dif_pos hcn]
      rw [blockSum_head N c.n1 (finalCounts s (leftStep c)).n1 _ hn hN1]
      congr 1
      ring

end
end Junpei.CommonSample
