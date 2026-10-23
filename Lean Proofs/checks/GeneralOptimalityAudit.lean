import Junpei.GeneralOptimalityFinite

-- Generic finite arm type, positive real precisions, and one finite budget.
-- No local Bellman condition is assumed by the actual-risk predicate.
#print Junpei.GeneralOptimality.MinimizesHistoryRisk
#print Junpei.GeneralOptimality.SequentiallyOptimalUpTo
#check Junpei.GeneralOptimality.toRaw_compile
#check Junpei.GeneralOptimality.actualRisk_compile
#check Junpei.GeneralOptimality.minimizing_actualRisk_eq_loss
#check Junpei.GeneralOptimality.minimizesHistoryRisk_iff_actualRisk_eq_loss
#check Junpei.GeneralOptimality.minimizing_rawRisk_eq_loss_ae
#check Junpei.GeneralOptimality.minimizing_draw_ae
#check Junpei.GeneralOptimality.minimizing_draw_EBI_ae
#check Junpei.GeneralOptimality.minimizing_recommend_ae
#check Junpei.GeneralOptimality.theorem1
#check Junpei.GeneralOptimality.exists_sequentiallyOptimalUpTo

#print axioms Junpei.GeneralOptimality.toRaw_compile
#print axioms Junpei.GeneralOptimality.actualRisk_compile
#print axioms Junpei.GeneralOptimality.minimizing_actualRisk_eq_loss
#print axioms Junpei.GeneralOptimality.minimizing_draw_EBI_ae
#print axioms Junpei.GeneralOptimality.minimizing_recommend_ae
#print axioms Junpei.GeneralOptimality.theorem1
#print axioms Junpei.GeneralOptimality.exists_sequentiallyOptimalUpTo

-- Exact proper Gaussian domain: no rule at zero precision is assumed.
#print Junpei.GeneralOptimality.ProperDomain.State
#print Junpei.GeneralOptimality.ProperDomain.Policy
#check Junpei.GeneralOptimality.ProperDomain.rawTree_extend
#check Junpei.GeneralOptimality.ProperDomain.actualRisk_extend
#check Junpei.GeneralOptimality.ProperDomain.theorem1
#print axioms Junpei.GeneralOptimality.ProperDomain.rawTree_extend
#print axioms Junpei.GeneralOptimality.ProperDomain.actualRisk_extend
#print axioms Junpei.GeneralOptimality.ProperDomain.theorem1

-- Literal finite policy: precisely N sampling maps and one terminal map.
#print Junpei.GeneralOptimality.FiniteHorizon.Policy
#print Junpei.GeneralOptimality.FiniteHorizon.actualRisk
#print Junpei.GeneralOptimality.FiniteHorizon.MinimizesHistoryRisk
#print Junpei.GeneralOptimality.FiniteHorizon.SequentiallyOptimal
#check Junpei.GeneralOptimality.FiniteHorizon.rawTree_extend
#check Junpei.GeneralOptimality.FiniteHorizon.actualRisk_extend
#check Junpei.GeneralOptimality.FiniteHorizon.theorem1
#check Junpei.GeneralOptimality.FiniteHorizon.canonical_actualRisk
#check Junpei.GeneralOptimality.FiniteHorizon.exists_sequentiallyOptimal
#print axioms Junpei.GeneralOptimality.FiniteHorizon.rawTree_extend
#print axioms Junpei.GeneralOptimality.FiniteHorizon.actualRisk_extend
#print axioms Junpei.GeneralOptimality.FiniteHorizon.theorem1
#print axioms Junpei.GeneralOptimality.FiniteHorizon.canonical_sequentiallyOptimal
#print axioms Junpei.GeneralOptimality.FiniteHorizon.exists_sequentiallyOptimal
