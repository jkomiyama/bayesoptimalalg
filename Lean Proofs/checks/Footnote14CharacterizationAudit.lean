import Junpei.Footnote14Characterization

-- Inspect the actual Gaussian law, existing scale definitions, and the
-- intermediate natural-exponent predicate used in the proof.
#print Junpei.ProseCorollaries.Footnote14.standardDeviation
#print Junpei.ProseCorollaries.Footnote14.logarithmicGap
#print Junpei.Footnote14Characterization.EventuallyPolynomialTail
#check Junpei.Footnote14Scale.standardDeviation_eq_paper_scale
#check Junpei.Footnote14Characterization.tail_standardize
#check Junpei.Footnote14Characterization.eventuallyPolynomialTail_of_isBigO
#check Junpei.Footnote14Characterization.isBigO_of_eventuallyPolynomialTail
#check Junpei.Footnote14Characterization.eventuallyPolynomialTail_iff_isBigO

-- The final signature has real c>0 and p>0 and an explicit natural T0.
-- Its only hypothesis is nonnegativity of the arbitrary real sequence d.
#check Junpei.Footnote14Characterization.realExponentPolynomialTail_iff_eventually
#check Junpei.Footnote14Characterization.corrected_footnote14_iff
#print axioms Junpei.Footnote14Characterization.tail_standardize
#print axioms Junpei.Footnote14Characterization.eventuallyPolynomialTail_of_isBigO
#print axioms Junpei.Footnote14Characterization.isBigO_of_eventuallyPolynomialTail
#print axioms Junpei.Footnote14Characterization.realExponentPolynomialTail_iff_eventually
#print axioms Junpei.Footnote14Characterization.corrected_footnote14_iff

-- The prior counterexample to the smaller claimed necessary scale is retained.
#check Junpei.ProseCorollaries.Footnote14.counterexample_to_necessary_scale
#check Junpei.Footnote14Scale.polynomial_tail_does_not_require_paper_scale
#print axioms Junpei.Footnote14Scale.polynomial_tail_does_not_require_paper_scale
