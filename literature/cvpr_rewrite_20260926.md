# CVPR 方法论文阅读与本项目改写依据

阅读日期：2026-09-26。范围：以下两篇官方 PDF 的正文第 1–8 页，
未把附录视为已完成审查。两篇均属于方法论文，而非发现型证据论文。
元数据由 CVF 官方 BibTeX 获取，并与 arXiv 记录交叉核对。

## FlowIE — CVPR 2024

- Zhu et al., *FlowIE: Efficient Image Enhancement via Rectified Flow*, pp.13–22.
- [官方论文](https://openaccess.thecvf.com/content/CVPR2024/papers/Zhu_FlowIE_Efficient_Image_Enhancement_via_Rectified_Flow_CVPR_2024_paper.pdf)
- [独立元数据](https://arxiv.org/abs/2406.00508)
- 一句话：把预训练扩散先验转为条件流，研究少步图像增强。
- 引言逻辑：应用困难 → 现有方法的推理成本 → 条件传输设计 → 可验证的质量/效率主张。
- 方法：预备定义后依次说明条件、路径和采样；不是先堆模块名称。
- 实验：§4.1 设置，§4.2 主比较，§4.3 设计消融，§4.4 扩展；
  Table 1/2 同时提供质量与 FPS，Table 3 对应方法设计。
- 启发：质量与速度是两个证据轴，不能仅凭步数推断速度。
- 不照搬：该文 §3.3 描述使用少量 test data 选择采样点；本项目不能
  借此为测试集选阈值辩护。其 t=0 控制也不自动证明本项目 FM 的必要性。

## ResFlow — CVPR 2025

- Qin et al., *Reversing Flow for Image Restoration*, pp.7545–7558.
- [官方论文](https://openaccess.thecvf.com/content/CVPR2025/papers/Qin_Reversing_Flow_for_Image_Restoration_CVPR_2025_paper.pdf)
- [独立元数据](https://arxiv.org/abs/2506.16961)
- 一句话：从不可逆退化的歧义出发，引入辅助状态构造恢复流。
- 引言/图 1 先交代物理与建模矛盾，图 2 再展示模型；§3 解释
  建模、参数化、优化与推断，§4 分主结果和组件消融。
- Table 5 消融辅助变量、日程和注入方式；Fig.5 检查中间恢复过程。
- 启发：系统假设必须先于方法，组件必须有对应证据。
- 边界：该文的增强状态可逆性论述不是本项目恢复丢失信道信息的证明。
  不能照搬为“虚拟重传创造信息”或宣称每图改善。

## 本稿采用与拒绝的写法

采用“问题—设计—预测—对照”结构；贡献收束为带共同物理回退的接收机
计算动作。先定义发送、侧信息、反馈，再给出修复和决策。
实验顺序：固定协议 → 总体效果 → 配对决策机制 → 风险/开销 → 外部定位。
把失败边界纳入主论证，避免写成逐次实验日志。

不复制原文，不套用顶会的 SOTA 语气，不把重写当成新结果。
不新增 seed sweep；不把 PSNR 提升写成 LPIPS/MS-SSIM 改善；
不把 K=4 写成二阶积分、不把潜变量掩码写成空间无损或稀疏加速。
继续用 IEEEtran 和通信资源度量，不移植 CVPR 页数要求。

## 仍缺的决定性证据

1. 相同风险约束下的有效 ACK 覆盖，而非只比较不同风险的 NACK。
2. 相同信道/前向率/反馈假设下的已发表架构比较。
3. 端到端处理时延与物理重传成本；现有微基准不能替代。

这些是实验缺口，不是增加文字可弥补的写作缺口。
