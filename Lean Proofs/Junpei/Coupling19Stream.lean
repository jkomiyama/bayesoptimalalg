import Junpei.CommonSampleIntegral

open MeasureTheory ProbabilityTheory Set
open scoped NNReal ENNReal

namespace Junpei.Coupling19

open Draft0.Counts Draft0.Gaussian Draft0.TwoArmBellman Draft0.TwoArmGain
open Junpei.CommonSample Junpei.Forecast

noncomputable section

/-- Actual common-product-space loss reduction. The extra LEFT observation
is placed LAST; this equality identifies its integral with the forced-first
Bellman loss reduction and the conditional one-step-gain average. -/
theorem left_common_product_gain (N q k : ℕ) (c : Counts) (m1 m2 : ℝ)
    (hq : 0 < q) (h1 : 0 < c.n1) (h2 : 0 < c.n2)
    (hcap1 : c.n1 ≤ q) (hcap2 : c.n2 ≤ q)
    (hs : c.n1 + c.n2 + k = 2 * q - 1)
    (hN : q + 1 ≤ N) :
    (∫ x : Bool × Fin N → ℝ,
      (streamTerminalRisk N (schedule rightTie k c) c m1 m2 x -
       streamTerminalRisk N (schedule rightTie k c ++ [true]) c m1 m2 x) ∂streamLaw N) =
      leftAction k m1 m2 c - payoff k m1 m2 c := by
  have hskip := oriented_skip_left_counts q k c hq hcap1 hcap2 hs
  have hskip1 : (finalCounts (schedule rightTie k c) c).n1 ≤ N := by
    rw [schedule_final, hskip.1]
    split_ifs <;> omega
  have hskip2 : (finalCounts (schedule rightTie k c) c).n2 ≤ N := by
    rw [schedule_final, hskip.2]
    split_ifs <;> omega
  have hforce1 : (finalCounts (schedule rightTie k c ++ [true]) c).n1 ≤ N := by
    rw [finalCounts_append]
    change (finalCounts (schedule rightTie k c) c).n1 + 1 ≤ N
    rw [schedule_final, hskip.1]
    split_ifs <;> omega
  have hforce2 : (finalCounts (schedule rightTie k c ++ [true]) c).n2 ≤ N := by
    rw [finalCounts_append]
    exact hskip2
  rw [integral_sub (streamTerminalRisk_integrable N _ c m1 m2 hskip1 hskip2)
    (streamTerminalRisk_integrable N _ c m1 m2 hforce1 hforce2),
    stream_integral_eq_recursive_risk N _ c m1 m2 h1 h2 hskip1 hskip2,
    stream_integral_eq_recursive_risk N _ c m1 m2 h1 h2 hforce1 hforce2]
  have hcoup := risk_permutation _ _
    (forced_left_is_common_then_last q k c hq hcap1 hcap2 hs) c m1 m2 h1 h2
  rw [forced_left_risk_eq k c m1 m2 h1 h2] at hcoup
  rw [skip_risk_eq rightTie rightTie_smaller k c m1 m2 h1 h2, ← hcoup]
  ring

/-- The symmetric common-product-space loss reduction for arm 2. -/
theorem right_common_product_gain (N q k : ℕ) (c : Counts) (m1 m2 : ℝ)
    (hq : 0 < q) (h1 : 0 < c.n1) (h2 : 0 < c.n2)
    (hcap1 : c.n1 ≤ q) (hcap2 : c.n2 ≤ q)
    (hs : c.n1 + c.n2 + k = 2 * q - 1)
    (hN : q + 1 ≤ N) :
    (∫ x : Bool × Fin N → ℝ,
      (streamTerminalRisk N (schedule leftTie k c) c m1 m2 x -
       streamTerminalRisk N (schedule leftTie k c ++ [false]) c m1 m2 x) ∂streamLaw N) =
      rightAction k m1 m2 c - payoff k m1 m2 c := by
  have hskip := equation22_terminal_counts q k c hq hcap1 hcap2 hs
  have hskip1 : (finalCounts (schedule leftTie k c) c).n1 ≤ N := by
    rw [schedule_final, leftTie_run, hskip.1]
    split_ifs <;> omega
  have hskip2 : (finalCounts (schedule leftTie k c) c).n2 ≤ N := by
    rw [schedule_final, leftTie_run, hskip.2]
    split_ifs <;> omega
  have hforce1 : (finalCounts (schedule leftTie k c ++ [false]) c).n1 ≤ N := by
    rw [finalCounts_append]
    exact hskip1
  have hforce2 : (finalCounts (schedule leftTie k c ++ [false]) c).n2 ≤ N := by
    rw [finalCounts_append]
    change (finalCounts (schedule leftTie k c) c).n2 + 1 ≤ N
    rw [schedule_final, leftTie_run, hskip.2]
    split_ifs <;> omega
  rw [integral_sub (streamTerminalRisk_integrable N _ c m1 m2 hskip1 hskip2)
    (streamTerminalRisk_integrable N _ c m1 m2 hforce1 hforce2),
    stream_integral_eq_recursive_risk N _ c m1 m2 h1 h2 hskip1 hskip2,
    stream_integral_eq_recursive_risk N _ c m1 m2 h1 h2 hforce1 hforce2]
  have hcoup := risk_permutation _ _
    (forced_right_is_common_then_last q k c hq hcap1 hcap2 hs) c m1 m2 h1 h2
  rw [forced_right_risk_eq k c m1 m2 h1 h2] at hcoup
  rw [skip_risk_eq leftTie leftTie_smaller k c m1 m2 h1 h2, ← hcoup]
  ring

end
end Junpei.Coupling19
