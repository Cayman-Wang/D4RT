# 新对话启动模板（D4RT 仓库专用）

> 用法：当任务明确只针对 `D4RT/` 仓库时，把本文件内容整体粘贴为第一条消息，再补你的具体任务。

```text
请先读取以下计划文件，再回答我的问题：

workspace_path=<workspace_path>
main_plan=<main_plan>
active_plan=<active_plan>

执行顺序要求：
1) 先读取 active_plan。
2) 再读取 main_plan。
3) 再按 active_plan 中 must_read 列表读取其余必读文件。

在你的第一条回复中，先输出 Intent Digest（三行）：
Goal: <从计划中提炼的一句话目标>
Current Milestone: <当前里程碑>
Next Action: <下一步可执行动作>

如果任何必读文件缺失、路径无效或计划冲突，请先明确报错并给出修复建议，不要直接开始实现。
```

## 默认变量示例
- `workspace_path=/home/grasp/Desktop/wym-project/4d-gaussgym/D4RT`
- `main_plan=/home/grasp/Desktop/wym-project/4d-gaussgym/D4RT/research/plans/d4rt_static_dynamic_separation/master_plan_zh.md`
- `active_plan=/home/grasp/Desktop/wym-project/4d-gaussgym/D4RT/research/plans/ACTIVE_PLAN.md`

## 说明
- 本工作区只服务于 D4RT repo-specific 任务。
- 如果任务涉及跨模型路线、Route A 主线或 Gaussian-first 备忘，请回到根目录 `research/`。
