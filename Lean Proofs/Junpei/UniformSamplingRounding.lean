import Junpei.UniformSamplingConclusion

set_option linter.unusedSectionVars false

open MeasureTheory ProbabilityTheory Set Filter
open scoped NNReal ENNReal Topology

namespace Junpei.UniformSampling

noncomputable section

variable {A : Type*} [Fintype A] [Nonempty A]

/-- Exact balanced allocation: each arm gets floor(T/K), and the first
T mod K arms in the fixed enumeration get one additional observation. -/
def allocation (T : ℕ) (a : A) : ℕ :=
  T / Fintype.card A +
    if (Fintype.equivFin A a).val < T % Fintype.card A then 1 else 0

theorem allocation_bounds (T : ℕ) (a : A) :
    T / Fintype.card A ≤ allocation T a ∧
      allocation T a ≤ T / Fintype.card A + 1 := by
  unfold allocation
  split_ifs <;> omega

theorem allocation_sum (T : ℕ) : ∑ a : A, allocation T a = T := by
  classical
  let K := Fintype.card A
  have hK : 0 < K := Fintype.card_pos
  have hrem : T % K ≤ K := (Nat.mod_lt T hK).le
  have hcount : (∑ a : Fin K, if a.val < T % K then (1 : ℕ) else 0) = T % K := by
    have hc : (Finset.univ.filter (fun a : Fin K => a.val < T % K)).card = T % K := by
      simpa only [Fintype.card_subtype] using Fintype.card_fin_lt_of_le hrem
    simpa only [Finset.sum_ite, Finset.sum_const, nsmul_eq_mul, mul_one, mul_zero,
      add_zero] using hc
  have he : (∑ a : A, if (Fintype.equivFin A a).val < T % K then (1 : ℕ) else 0) =
      T % K := by
    exact (Equiv.sum_comp (Fintype.equivFin A)
      (fun a : Fin K => if a.val < T % K then (1 : ℕ) else 0)).trans hcount
  simp only [allocation, Finset.sum_add_distrib, Finset.sum_const, Finset.card_univ,
    nsmul_eq_mul]
  rw [he]
  exact Nat.div_add_mod T (Fintype.card A)

theorem allocation_positive (T : ℕ) (hT : Fintype.card A ≤ T) (a : A) :
    0 < allocation T a := by
  have hK : 0 < Fintype.card A := Fintype.card_pos
  exact (Nat.div_pos hT hK).trans_le (allocation_bounds T a).1

theorem allocation_le_budget (T : ℕ) (a : A) : allocation T a ≤ T := by
  calc
    allocation T a ≤ ∑ b : A, allocation T b :=
      Finset.single_le_sum (fun b _ => Nat.zero_le (allocation T b)) (Finset.mem_univ a)
    _ = T := allocation_sum T

theorem allocation_proportion (a : A) :
    Tendsto (fun T : ℕ => (allocation T a : ℝ) / (T : ℝ)) atTop
      (𝓝 (1 / (Fintype.card A : ℝ))) := by
  let K : ℝ := Fintype.card A
  have hK : 0 < K := by
    dsimp [K]
    exact_mod_cast (Fintype.card_pos : 0 < Fintype.card A)
  have hlo : Tendsto (fun T : ℕ => 1/K - 1/(T : ℝ)) atTop (𝓝 (1/K)) := by
    simpa using tendsto_const_nhds.sub (tendsto_const_div_atTop_nhds_zero_nat (1 : ℝ))
  have hhi : Tendsto (fun T : ℕ => 1/K + 1/(T : ℝ)) atTop (𝓝 (1/K)) := by
    simpa using tendsto_const_nhds.add (tendsto_const_div_atTop_nhds_zero_nat (1 : ℝ))
  have hbounds (T : ℕ) : (T : ℝ) - K ≤ (allocation T a : ℝ) * K ∧
      (allocation T a : ℝ) * K ≤ (T : ℝ) + K := by
    have he : (T : ℝ) = (T / Fintype.card A : ℕ) * K +
        (T % Fintype.card A : ℕ) := by
      dsimp [K]
      exact_mod_cast (show T = (T / Fintype.card A) * Fintype.card A +
        T % Fintype.card A by
          simpa only [Nat.mul_comm] using (Nat.div_add_mod T (Fintype.card A)).symm)
    have hr : (T % Fintype.card A : ℕ) < K := by
      dsimp [K]
      exact_mod_cast Nat.mod_lt T (Fintype.card_pos : 0 < Fintype.card A)
    have hr0 : (0 : ℝ) ≤ (T % Fintype.card A : ℕ) := by positivity
    have hl : (T / Fintype.card A : ℕ) ≤ (allocation T a : ℝ) := by
      exact_mod_cast (allocation_bounds T a).1
    have hu : (allocation T a : ℝ) ≤ (T / Fintype.card A : ℕ) + 1 := by
      exact_mod_cast (allocation_bounds T a).2
    constructor <;> nlinarith
  apply tendsto_of_tendsto_of_tendsto_of_le_of_le' hlo hhi
  · filter_upwards [eventually_ge_atTop (1 : ℕ)] with T hT
    have hTp : 0 < (T : ℝ) := by exact_mod_cast (by omega : 0 < T)
    apply (le_div_iff₀ hTp).mpr
    have hb := (hbounds T).1
    field_simp [hK.ne', hTp.ne']
    nlinarith
  · filter_upwards [eventually_ge_atTop (1 : ℕ)] with T hT
    have hTp : 0 < (T : ℝ) := by exact_mod_cast (by omega : 0 < T)
    apply (div_le_iff₀ hTp).mpr
    have hb := (hbounds T).2
    field_simp [hK.ne', hTp.ne']
    nlinarith

end
end Junpei.UniformSampling
