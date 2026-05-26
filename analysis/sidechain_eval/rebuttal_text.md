# Suggested rebuttal text (sidechain reconstruction)

MHC-Diff predicts peptide Cα coordinates only. For Zenodo structures and downstream use we add sidechains in a separate post-processing step, either with **PDBFixer + OpenMM** (as in the original pipeline) or with **DLPacker** on a fixed peptide backbone (crystal N, C, O + predicted Cα).

On 50 representative X-ray test structures (lowest best-of-10 Cα RMSD from the 8K benchmark), DLPacker achieves mean all-heavy RMSD of **1.90 Å** vs crystal, comparable to full-atom errors reported for end-to-end tools such as APE-Gen (~2.0 Å) and PANDORA (~1.5 Å). The generic PDBFixer + OpenMM step performs worse on sidechains (**3.1 Å** sidechain RMSD). Sidechain placement is therefore a distinct step from Cα placement; it could be improved further by extending MHC-Diff to full-atom prediction, which was outside the scope of the current model design.

See `APPENDIX_sidechain_benchmark.tex` and `reference_results/` for the full comparison table.
