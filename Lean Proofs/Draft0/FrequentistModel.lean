import Draft0.BadEvent
import Draft0.PolicyPaths

open MeasureTheory ProbabilityTheory
open scoped BigOperators

namespace Draft0.FrequentistModel

open BayesianModel BadEvent

noncomputable section

/-- The instance used in the main theorem has true means (0,0,1). -/
def trueMean (i : Arm) : ℝ := if i = 2 then 1 else 0

def observation {T : ℕ} (ω : RawArrays T) (i : Arm) (n : Fin T) : ℝ :=
  ω i n + trueMean i

/-- Every statistic supplied to the Bayes policy is computed from the actual
independent frequentist reward array. -/
def empiricalPath {T : ℕ} (ω : RawArrays T) : PolicyPaths.Path :=
  fun i n => sampleMean n (ω i) + trueMean i

theorem true_instance_gap (i : Arm) : 1 - trueMean i = if i = 2 then 0 else 1 := by
  unfold trueMean
  split_ifs <;> norm_num

theorem empiricalPath_first {T : ℕ} (ω : RawArrays T) (n : ℕ) :
    empiricalPath ω 0 n = fullMeanPath ω 1 n := by simp [empiricalPath, trueMean, fullMeanPath]

theorem empiricalPath_second {T : ℕ} (ω : RawArrays T) (n : ℕ) :
    empiricalPath ω 1 n = fullMeanPath ω 2 n := by simp [empiricalPath, trueMean, fullMeanPath]

theorem empiricalPath_third {T : ℕ} (ω : RawArrays T) (n : ℕ) :
    empiricalPath ω 2 n = fullMeanPath ω 3 n := by simp [empiricalPath, trueMean, fullMeanPath]

theorem sampleMean_continuous (T n : ℕ) : Continuous (sampleMean (q := T) n) := by
  unfold sampleMean weightedSum
  fun_prop

theorem sampleMean_mul_count (T n : ℕ) (ω : Fin T → ℝ) (hn : 0 < n) :
    (n : ℝ) * sampleMean n ω = ∑ i : Fin T, if (i : ℕ) < n then ω i else 0 := by
  have hn0 : (n : ℝ) ≠ 0 := by exact_mod_cast Nat.ne_of_gt hn
  unfold sampleMean weightedSum
  rw [Finset.mul_sum]
  apply Finset.sum_congr rfl
  intro i _hi
  unfold meanCoefficient
  split_ifs
  · field_simp
  · simp

theorem rawSum_succ (T n : ℕ) (ω : Fin T → ℝ) (hnT : n < T) :
    (∑ i : Fin T, if (i : ℕ) < n + 1 then ω i else 0) =
      (∑ i : Fin T, if (i : ℕ) < n then ω i else 0) + ω ⟨n, hnT⟩ := by
  classical
  have hpoint : ∀ i : Fin T,
      (if (i : ℕ) < n + 1 then ω i else 0) =
        (if (i : ℕ) < n then ω i else 0) + (if i = ⟨n, hnT⟩ then ω i else 0) := by
    intro i
    by_cases hi : (i : ℕ) < n
    · have hin1 : (i : ℕ) < n + 1 := by omega
      have hine : i ≠ ⟨n, hnT⟩ := by
        intro h
        have hv : (i : ℕ) = n := congrArg Fin.val h
        omega
      simp [hi, hin1, hine]
    · by_cases hieq : (i : ℕ) = n
      · have hieq' : i = ⟨n, hnT⟩ := Fin.ext hieq
        simp [hieq']
      · have hin1 : ¬ ((i : ℕ) < n + 1) := by omega
        have hine : i ≠ ⟨n, hnT⟩ := by intro h; exact hieq (congrArg Fin.val h)
        simp [hi, hin1, hine]
  simp_rw [hpoint]
  rw [Finset.sum_add_distrib]
  simp

/-- The empirical statistic of the actual raw reward array follows Eq. (3). -/
theorem sampleMean_update (T n : ℕ) (ω : Fin T → ℝ) (hn : 0 < n) (hnT : n < T) :
    sampleMean (n + 1) ω =
      Gaussian.posteriorMeanUpdate (sampleMean n ω) (n : ℝ) (ω ⟨n, hnT⟩) := by
  have hsum := rawSum_succ T n ω hnT
  rw [← sampleMean_mul_count T (n + 1) ω (by omega),
    ← sampleMean_mul_count T n ω hn] at hsum
  have hn1 : (n : ℝ) + 1 ≠ 0 := by positivity
  unfold Gaussian.posteriorMeanUpdate
  apply (eq_div_iff hn1).mpr
  push_cast at hsum
  nlinarith only [hsum]

theorem empiricalPath_update {T : ℕ} (ω : RawArrays T) (i : Arm) (n : ℕ)
    (hn : 0 < n) (hnT : n < T) :
    empiricalPath ω i (n + 1) =
      Gaussian.posteriorMeanUpdate (empiricalPath ω i n) (n : ℝ)
        (observation ω i ⟨n, hnT⟩) := by
  have h := sampleMean_update T n (ω i) hn hnT
  have hn1 : (n : ℝ) + 1 ≠ 0 := by positivity
  simp only [empiricalPath, observation, h, Gaussian.posteriorMeanUpdate]
  field_simp
  ring

theorem empiricalPath_measurable (T : ℕ) : Measurable (empiricalPath (T := T)) := by
  apply measurable_pi_lambda
  intro i
  apply measurable_pi_lambda
  intro n
  exact (((sampleMean_continuous T n).measurable.comp (measurable_pi_apply i)).add_const
    (trueMean i))

def actualRecommendation (T : ℕ) (ω : RawArrays T) : Arm :=
  PolicyPaths.recommendation T (empiricalPath ω)

theorem actualRecommendation_measurable (T : ℕ) : Measurable (actualRecommendation T) :=
  (PolicyPaths.recommendation_measurable T).comp (empiricalPath_measurable T)

def wrongEvent (T : ℕ) : Set (RawArrays T) := {ω | actualRecommendation T ω ≠ 2}

theorem wrongEvent_measurable (T : ℕ) : MeasurableSet (wrongEvent T) :=
  ((actualRecommendation_measurable T) (measurableSet_singleton 2)).compl

/-- Frequentist expected simple regret of the explicit posterior Bellman
policy. The integrand is the true gap of its actual recommendation. -/
def regret (T : ℕ) : ℝ :=
  ∫ ω : RawArrays T, 1 - trueMean (actualRecommendation T ω) ∂rawArraysLaw T

theorem regret_eq_wrong_probability (T : ℕ) :
    regret T = (rawArraysLaw T).real (wrongEvent T) := by
  have hi : (fun ω : RawArrays T => 1 - trueMean (actualRecommendation T ω)) =
      (wrongEvent T).indicator (fun _ => (1 : ℝ)) := by
    funext ω
    rw [true_instance_gap]
    by_cases h : actualRecommendation T ω = 2 <;> simp [wrongEvent, h]
  unfold regret
  rw [hi, integral_indicator (wrongEvent_measurable T)]
  simp

theorem badEvent_regret_lower_bound (T q : ℕ) (hT : T = 2 * q + 1) (hq : 2 ≤ q)
    (hwrong : rawBadEvent T q (by omega) (by omega) ⊆ wrongEvent T) :
    (underestimationConstant / 64) * Rates.invPower 886 T ≤ regret T := by
  rw [regret_eq_wrong_probability]
  exact (rawBadEvent_probability_lower T q hT hq).trans (measureReal_mono hwrong)

end

end Draft0.FrequentistModel
