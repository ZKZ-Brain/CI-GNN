<h2 align="center">

CI-GNN: A Granger Causality-Inspired Graph Neural Network 🔥

</h2>

## The Architecture of CI-GNN

![](framework.png)

**Figure 1 ﻿Architecture of our proposed CI-GNN.**  The model consists of four modules: GraphVAE, causal effect estimator, causal subgraph generator and a basic classifier $\varphi$. Given an input Graph $G=\{(A,X)\}$, GraphVAE learns (disentangled) latent factors $Z=[\alpha;\beta]$. The causal effect estimator ensures that only $\alpha$ is causally related to label $Y$ by the conditional mutual information (CMI) regularization $I\left ( \alpha; Y|\beta \right )$. Based on $\alpha$, we introduce another linear decoder $\theta_2$ to generate causal subgraph $\mathcal{G}_{\text{sub}}$, which can then be used for graph classification by classifier $\varphi$.


## Questions, Suggestions, and Collaborations

If you have any questions, suggestions, or would like to collaborate us on relevant topics, please feel free to contact us by [yusj9011@gmail.com](mailto:yusj9011@gmail.com) (Shujian Yu), kzzheng@stu.xjtu.edu.cn (Kaizhong Zheng).

## Reference
```
@article{zheng2024ci,
  title={Ci-gnn: A granger causality-inspired graph neural network for interpretable brain network-based psychiatric diagnosis},
  author={Zheng, Kaizhong and Yu, Shujian and Chen, Badong},
  journal={Neural Networks},
  pages={106147},
  year={2024},
  publisher={Elsevier}
}
```
