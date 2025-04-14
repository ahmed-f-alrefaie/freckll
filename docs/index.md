# FRECKLL

[![Release](https://img.shields.io/github/v/release/ahmed-f-alrefaie/FRECKLL)](https://img.shields.io/github/v/release/ahmed-f-alrefaie/FRECKLL)
[![Build status](https://img.shields.io/github/actions/workflow/status/ahmed-f-alrefaie/FRECKLL/main.yml?branch=main)](https://github.com/ahmed-f-alrefaie/FRECKLL/actions/workflows/main.yml?query=branch%3Amain)
[![Commit activity](https://img.shields.io/github/commit-activity/m/ahmed-f-alrefaie/FRECKLL)](https://img.shields.io/github/commit-activity/m/ahmed-f-alrefaie/FRECKLL)
[![License](https://img.shields.io/github/license/ahmed-f-alrefaie/FRECKLL)](https://img.shields.io/github/license/ahmed-f-alrefaie/FRECKLL)

**F**ast and **R**educed **E**xoplanet **C**hemical **K**inetics disti**LL**ed (**FRECKLL**) is a chemical kinetic solver in Python that
employes numerical distillation and the [K-fold summation technique](https://epubs.siam.org/doi/10.1137/030601818) to solve chemical networks
rapidly and accurately. The full [Venot Methanol network](https://ui.adsabs.harvard.edu/abs/2020A%26A...634A..78V/abstract)
with photochemistry can be solved to steady state in as little as 42 seconds and the reduced network in only 7 seconds!

![HD 209458 b](assets/full_hd209_light.gif#only-light)
![HD 209458 b](assets/full_hd209_dark.gif#only-dark)

Head to the [Getting Started](gettingstarted.md) to get started or try the [Colab]() notebook from your web-browser!

## References

- [FRECKLL paper](https://iopscience.iop.org/article/10.3847/1538-4357/ad3dee)
- [Methanol Network](https://ui.adsabs.harvard.edu/abs/2020A%26A...634A..78V/abstract)
- [K-fold method](https://epubs.siam.org/doi/10.1137/030601818)
