# Experiment_Quantum_Continual_Learning

In our work [Quantum continual learning on a programmable superconducting processor](https://arxiv.org/abs/2409.09729), we report an experimental demonstration of quantum continual learning on a fully programmable superconducting processor. In particular, we sequentially train a quantum classifier with three tasks, two about identifying real-life images and the other on classifying quantum states, and demonstrate its catastrophic forgetting through experimentally observed rapid performance drops for prior tasks. To overcome this dilemma, we exploit the elastic weight consolidation strategy and show that the quantum classifier can incrementally learn and retain knowledge across the three distinct tasks.

Here, we provide the codes for numerical simulations, data for experimental results and numerical results.

## Contents

- [Numerical Simulations](Numerical_Simulations)
- [Experimental Results](Experimental_Results)

## The numerical simulations are built With

* [Yao Quantum](https://yaoquantum.org/) - An open-source quantum simulation framework in Julia language

Detailed installation instructions and tutorials of Julia and Yao.jl can be found at [julialang.org](https://julialang.org/) and [yaoquantum.org/documentation](https://docs.yaoquantum.org/dev/).

## Funding

We acknowledge support from the Quantum Science and Technology-National Science and Technology Major Project (Grant Nos. 2021ZD0300200 and 2021ZD0302203), the National Natural Science Foundation of China (Grant Nos. 12174342, 92365301, 12274367, 12322414, 12274368, 12075128, and T2225008), the National Key R&D ProgramofChina (GrantNo. 2023YFB4502600), and the ZhejiangProvincial Natural Science Foundation of China (Grant Nos. LDQ23A040001, LR24A040002). Z.L., W.L., W.J., Z.-Z.S., and D.-L.D. are supported in addition by Tsinghua University Dushi Program, and the Shanghai Qi Zhi Institute Innovation Program (Grant No. SQZ202318). C.S. is supported by the Xiaomi Young Scholars Program. P.-X.S. acknowledges support from the European Union's Horizon Europe research and innovation programme under the Marie Skłodowska-Curie Grant Agreement No. 101180589 (SymPhysAI), the National Science Centre (Poland) OPUS Grant No. 2021/41/B/ST3/04475, and the Foundation for Polish Science project MagTop (No. FENG.02.01-IP.05-0028/23) co-financed by the European Union from the funds of Priority 2 of the European Funds for a Smart Economy Program 2021–2027 (FENG). Views and opinions expressed are however those of the author(s) only and do not necessarily reflect those of the European Union or the European Research Executive Agency. Neither the European Union nor the granting authority can be held responsible for them.

## License

Released under [MIT License](https://github.com/luzd19/Experiment_Quantum_Continual_Learning/blob/main/LICENSE) .
