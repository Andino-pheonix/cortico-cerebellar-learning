# 文献与实现关系

1. Nguyen KP, Person AL. Cerebellar circuit computations for predictive motor control. *Nature Reviews Neuroscience*. 2025;26:538–553. [原文链接](https://doi.org/10.1038/s41583-025-00936-z)。提供小脑预测控制与回路机制背景；本模型未实现PC、CF等细胞。
2. Israely S, Ninou H, Rajchert O, et al. Cerebellar output shapes cortical preparatory activity during motor adaptation. *Nature Communications*. 2025;16:2574. [原文链接](https://doi.org/10.1038/s41467-025-57832-4)。启发小脑回传影响皮层准备的连接关系；本模型不复现其猴实验或完整计算模型。
3. Tseng YW, Diedrichsen J, Krakauer JW, Shadmehr R, Bastian AJ. Sensory prediction errors drive cerebellum-dependent adaptation of reaching. *Journal of Neurophysiology*. 2007;98:54–62. [原文链接](https://doi.org/10.1152/jn.00266.2007)。帮助区分当前纠正动作与后续适应；本实现使用期望轨迹跟踪误差，未建立独立感觉结果预测器。
4. Feulner B, Perich MG, Miller LE, Clopath C, Gallego JA. A neural implementation model of feedback-based motor learning. *Nature Communications*. 2025;16:1805. [原文链接](https://doi.org/10.1038/s41467-024-54738-5)。启发保存早期活动与晚到误差配对的学习思路；本模型使用读出层更新，未复现原文400单元RNN的循环连接学习规则。

仓库图示由本实验代码或独立示意整理生成；没有复制上述论文原图、PDF、数据或作者代码。
