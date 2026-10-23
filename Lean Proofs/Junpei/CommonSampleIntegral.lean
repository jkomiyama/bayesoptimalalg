import Junpei.CommonSampleJoint

open MeasureTheory ProbabilityTheory Set
open scoped NNReal ENNReal

namespace Junpei.CommonSample

open Draft0.Counts Draft0.Gaussian Draft0.TwoArmBellman Draft0.TwoArmGain

noncomputable section

def streamTerminalRisk (N : ℕ) (s : List Bool) (c : Counts) (m1 m2 : ℝ)
    (x : Bool × Fin N → ℝ) : ℝ :=
  terminalRisk (finalCounts s c)
    (terminalMeans s c (m1, m2) (finiteStream N x)).1
    (terminalMeans s c (m1, m2) (finiteStream N x)).2

theorem streamTerminalRisk_integrable (N : ℕ) (s : List Bool) (c : Counts) (m1 m2 : ℝ)
    (hN1 : (finalCounts s c).n1 ≤ N) (hN2 : (finalCounts s c).n2 ≤ N) :
    Integrable (streamTerminalRisk N s c m1 m2) (streamLaw N) := by
  let g : (Bool × Fin N → ℝ) → ℝ := fun x =>
    (terminalMeans s c (m1, m2) (finiteStream N x)).1 -
      (terminalMeans s c (m1, m2) (finiteStream N x)).2
  have hlaw : HasLaw g (gaussianReal (m1 - m2) (allocationVariance s c)) (streamLaw N) :=
    terminal_gap_law N s c m1 m2 hN1 hN2
  have hi : Integrable (fun d : ℝ => hingeLoss |d|
      (Draft0.TwoArmBellman.variance (finalCounts s c))) ((streamLaw N).map g) := by
    rw [hlaw.map_eq]
    exact absolute_hinge_gaussian_integrable _ _ _
  have h := (integrable_map_measure hi.aestronglyMeasurable hlaw.aemeasurable).mp hi
  convert h using 1
  funext x
  exact terminalRisk_eq_posteriorLoss (finalCounts s c) _ _

/-- Identification of the SINGLE joint per-arm Gaussian common-sample space
with the recursively integrated actual posterior-predictive regret. No
distributional or coupling hypothesis is assumed. -/
theorem stream_integral_eq_recursive_risk (N : ℕ) (s : List Bool) (c : Counts)
    (m1 m2 : ℝ) (h1 : 0 < c.n1) (h2 : 0 < c.n2)
    (hN1 : (finalCounts s c).n1 ≤ N) (hN2 : (finalCounts s c).n2 ≤ N) :
    (∫ x : Bool × Fin N → ℝ, streamTerminalRisk N s c m1 m2 x ∂streamLaw N) =
      risk s c m1 m2 := by
  have hlaw := terminal_gap_law N s c m1 m2 hN1 hN2
  have hi := absolute_hinge_gaussian_integrable (m1 - m2) (allocationVariance s c)
    (Draft0.TwoArmBellman.variance (finalCounts s c))
  have hmap := hlaw.integral_comp hi.aestronglyMeasurable
  have hterminal : (fun x : Bool × Fin N → ℝ => streamTerminalRisk N s c m1 m2 x) =
      (fun x : Bool × Fin N → ℝ => hingeLoss
        |(terminalMeans s c (m1, m2) (finiteStream N x)).1 -
          (terminalMeans s c (m1, m2) (finiteStream N x)).2|
        (Draft0.TwoArmBellman.variance (finalCounts s c))) := by
    funext x
    exact terminalRisk_eq_posteriorLoss (finalCounts s c) _ _
  rw [hterminal]
  change (∫ x : Bool × Fin N → ℝ, ((fun d : ℝ => hingeLoss |d|
    (Draft0.TwoArmBellman.variance (finalCounts s c))) ∘
    (fun x : Bool × Fin N → ℝ => (terminalMeans s c (m1, m2) (finiteStream N x)).1 -
      (terminalMeans s c (m1, m2) (finiteStream N x)).2)) x ∂streamLaw N) = _
  rw [hmap]
  rw [risk_eq s c m1 m2 h1 h2, utility_eq, potential_eq_gaussian]
  have hs := allocationVariance_decomposition s c h1 h2
  rw [← hs]
  have hg := hinge_variance_increment_integral (m2 - m1) (allocationVariance s c)
    (Draft0.TwoArmBellman.variance (finalCounts s c))
  rw [show -(m2 - m1) = m1 - m2 by ring] at hg
  linarith only [hg]

theorem streamTerminalRisk_permutation (N : ℕ) (s u : List Bool) (h : s.Perm u)
    (c : Counts) (m1 m2 : ℝ) (x : Bool × Fin N → ℝ) :
    streamTerminalRisk N s c m1 m2 x = streamTerminalRisk N u c m1 m2 x := by
  unfold streamTerminalRisk
  rw [finalCounts_permutation s u h c, terminalMeans_permutation s u h c (m1, m2) (finiteStream N x)]

/-- The raw shared-sample terminal-risk integrals are equal pointwise before
integration under the SAME actual Gaussian product measure. -/
theorem stream_integral_permutation (N : ℕ) (s u : List Bool) (h : s.Perm u)
    (c : Counts) (m1 m2 : ℝ) :
    (∫ x : Bool × Fin N → ℝ, streamTerminalRisk N s c m1 m2 x ∂streamLaw N) =
      ∫ x : Bool × Fin N → ℝ, streamTerminalRisk N u c m1 m2 x ∂streamLaw N := by
  congr 1
  funext x
  exact streamTerminalRisk_permutation N s u h c m1 m2 x

end
end Junpei.CommonSample
