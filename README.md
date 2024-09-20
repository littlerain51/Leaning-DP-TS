# Leaning-DP-TS
在差分隐私中是不是噪声加入得越少对学习效果影响越小？

阅读这篇文章的一些笔记：
Near-Optimal Thompson Sampling-based Algorithms for Differentially Private Stochastic Bandits
原文链接：https://proceedings.mlr.press/v180/hu22a.html
<img width="879" alt="image" src="https://github.com/user-attachments/assets/befe8e14-4d8a-415a-bcab-e6cc6b075026">

与这篇文章最紧密联系的文章：
Near-Optimal Algorithms for Differentially Private Online Learning in a Stochastic Environment
原文链接：http://arxiv.org/abs/2102.07929

学习中的两个难点：
- binary mechanism几乎没有介绍，这一部分加噪声机制不明确
- 推导过程想不出来证明思路
 
学习笔记中详细注释了附录中的推导过程

文中提到modified logarithmic  mechanism是为了减少加噪声次数，所以产生两点疑问：
- modified logarithmic  mechanism确实在logarithmic  mechanism部分少加了噪声，但是结合随之而变化的binary mechanism部分加的噪声，是否总噪声减少了？
- 加的噪声越少，学习策略的效果就会越好吗？

为了解决这两个问题，按照算法流程编写了代码，实验结果展示modified加的总噪声一般会更多，对于不同数量级的差分隐私的水平（$\epsilon$)，确实噪声越少，累积regret就会更少，但是在差分隐私的水平（$\epsilon$)固定的情况下，modified logarithmic  mechanism和original logarithmic  mechanism对应的少量噪声的差异并不能决定哪一个的学习策略一定好。


<img width="576" alt="截屏2024-09-20 22 34 38" src="https://github.com/user-attachments/assets/1a0d5026-64b9-4f97-935e-4299e4586ab7">

以上纯属个人观点，供同样初学这篇文章的小伙伴参考，欢迎各位指正。
