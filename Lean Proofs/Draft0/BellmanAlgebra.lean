import Mathlib.Data.Real.Basic
import Mathlib.Tactic.Linarith

/-!
Algebraic Bellman steps with every probabilistic comparison premise explicit.
These proofs establish no mock-sample coupling or full-policy optimality.
-/

namespace Draft0.BellmanAlgebra

noncomputable def ebi (baseline actionLoss : ℝ) : ℝ := baseline - actionLoss

theorem ebi_order (baseline loss₁ loss₂ : ℝ) :
    ebi baseline loss₁ ≤ ebi baseline loss₂ ↔ loss₂ ≤ loss₁ := by
  unfold ebi
  constructor <;> intro h <;> linarith

theorem argmax_ebi_iff_argmin_loss {ι : Type*} (baseline : ℝ)
    (loss : ι → ℝ) (chosen : ι) :
    (∀ i, ebi baseline (loss i) ≤ ebi baseline (loss chosen)) ↔
      (∀ i, loss chosen ≤ loss i) := by
  constructor
  · intro h i
    exact (ebi_order baseline (loss i) (loss chosen)).mp (h i)
  · intro h i
    exact (ebi_order baseline (loss i) (loss chosen)).mpr (h i)

theorem lemma11_from_mock_comparisons (baseline actionLoss mockLoss contribution : ℝ)
    (hoptimal : baseline ≤ mockLoss)
    (hmock : mockLoss ≤ actionLoss + contribution) :
    ebi baseline actionLoss ≤ contribution := by
  unfold ebi
  linarith

theorem lemma12_from_comparisons (loss₃ loss₂ contribution : ℝ)
    (h₁ : loss₃ - loss₂ ≤ contribution) (h₂ : loss₂ - loss₃ ≤ contribution) :
    |loss₃ - loss₂| ≤ contribution := by
  apply abs_le.mpr
  constructor <;> linarith

theorem lemma7_transfer (baseline₃ action₃ baseline₂ action₂ contribution bound : ℝ)
    (hbase : |baseline₃ - baseline₂| ≤ contribution)
    (hact : |action₃ - action₂| ≤ contribution)
    (hbound : bound ≤ ebi baseline₂ action₂) :
    bound - 2 * contribution ≤ ebi baseline₃ action₃ := by
  rcases abs_le.mp hbase with ⟨hbase₁, hbase₂⟩
  rcases abs_le.mp hact with ⟨hact₁, hact₂⟩
  unfold ebi at *
  linarith

theorem lemma7_gap_transfer (b₁ b₂ c₁ c₂ contribution bound : ℝ)
    (h₁ : |b₁ - c₁| ≤ 2 * contribution)
    (h₂ : |b₂ - c₂| ≤ 2 * contribution)
    (hgap : bound ≤ c₁ - c₂) : bound - 4 * contribution ≤ b₁ - b₂ := by
  rcases abs_le.mp h₁ with ⟨h₁a, h₁b⟩
  rcases abs_le.mp h₂ with ⟨h₂a, h₂b⟩
  linarith

end Draft0.BellmanAlgebra
