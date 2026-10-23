import Junpei.ProbabilityModel
import Junpei.PathInduction
import Draft0.FrequentistModel

open MeasureTheory ProbabilityTheory
open scoped NNReal

namespace Junpei.Empirical

open Draft0.BayesianModel Draft0.BadEvent

noncomputable section

def trueMean (Δ : ℝ) (i : Arm) : ℝ := if i = 2 then Δ else 0

def observation {T : ℕ} (Δ : ℝ) (ω : RawArrays T) (i : Arm) (n : Fin T) : ℝ :=
  ω i n + trueMean Δ i

def empiricalPath {T : ℕ} (Δ : ℝ) (ω : RawArrays T) : Draft0.PolicyPaths.Path :=
  fun i n => sampleMean n (ω i) + trueMean Δ i

theorem paperPath_empirical {T : ℕ} (Δ : ℝ) (ω : RawArrays T) :
    Draft0.PathInduction.paperPath (empiricalPath Δ ω) = Junpei.Probability.rawMeanPath Δ ω := by
  ext i n
  simp only [Draft0.PathInduction.paperPath, empiricalPath, trueMean, Junpei.Probability.rawMeanPath]
  split_ifs <;> simp_all

theorem empiricalPath_measurable (T : ℕ) (Δ : ℝ) :
    Measurable (empiricalPath (T := T) Δ) := by
  apply measurable_pi_lambda
  intro i
  apply measurable_pi_lambda
  intro n
  exact (((Draft0.FrequentistModel.sampleMean_continuous T n).measurable.comp
    (measurable_pi_apply i)).add_const (trueMean Δ i))

theorem observation_hasLaw (T : ℕ) (Δ : ℝ) (i : Arm) (n : Fin T) :
    HasLaw (fun ω : RawArrays T => observation Δ ω i n)
      (gaussianReal (trueMean Δ i) 1) (rawArraysLaw T) := by
  have h1 := measurePreserving_eval (fun _ : Arm => arrayLaw T) i
  have h2 := measurePreserving_eval (fun _ : Fin T => gaussianReal 0 1) n
  have h3 : MeasurePreserving (fun x : ℝ => x + trueMean Δ i)
      (gaussianReal 0 1) (gaussianReal (trueMean Δ i) 1) := by
    constructor
    · fun_prop
    · simpa using gaussianReal_map_add_const (μ := (0 : ℝ)) (v := (1 : ℝ≥0)) (trueMean Δ i)
  exact (h3.comp (h2.comp h1)).hasLaw

theorem empiricalPath_update {T : ℕ} (Δ : ℝ) (ω : RawArrays T) (i : Arm) (n : ℕ)
    (hn : 0 < n) (hnT : n < T) :
    empiricalPath Δ ω i (n+1) = Draft0.Gaussian.posteriorMeanUpdate
      (empiricalPath Δ ω i n) (n : ℝ) (observation Δ ω i ⟨n, hnT⟩) := by
  have h := Draft0.FrequentistModel.sampleMean_update T n (ω i) hn hnT
  have hn1 : (n : ℝ)+1 ≠ 0 := by positivity
  simp only [empiricalPath, observation, h, Draft0.Gaussian.posteriorMeanUpdate]
  field_simp
  ring

theorem badEvent_goodPath (T q : ℕ) (Δ CU : ℝ) (hqT : q ≤ T) (hT : 0 < T)
    (ω : RawArrays T) (hω : ω ∈ Junpei.Probability.rawBadEvent T q Δ CU hqT hT) :
    Junpei.PathInduction.GoodPath T q CU (empiricalPath Δ ω) := by
  have h := Junpei.Probability.rawBadEvent_implies_newW T q Δ CU hqT hT ω hω
    (⟨0, 0, 0⟩ : Draft0.Counts.Counts) (by norm_num) (by norm_num) (by norm_num)
  rw [← paperPath_empirical] at h
  exact ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1⟩

theorem true_gap (Δ : ℝ) (i : Arm) : Δ - trueMean Δ i = if i=2 then 0 else Δ := by
  simp only [trueMean]
  split_ifs <;> ring

end
end Junpei.Empirical
