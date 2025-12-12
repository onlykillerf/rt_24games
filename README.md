# ToT 24 Games Solver
基于论文 **Tree of Thoughts: Deliberate Problem Solving with Large Language Models** 的思路，使用 GPT-4o-mini 模型实现的 24 点游戏求解器。相比传统的暴力或 DFS/BFS 搜索，该实现通过 **树状思维（Tree of Thoughts, ToT）** 对中间状态进行评分与筛选，能够找到更多样的合法解法。

---

## 🌟 功能亮点

- **Tree of Thoughts 推理**：LLM 负责为中间状态评分，选择更可能抵达 24 的路径。
- **Beam Search 剪枝**：默认 `beam_width=100`，在搜索空间和时间开销之间取得平衡。
- **算式规范化 & 去重**：加减乘除均做了格式化处理，自动避免重复解。
- **异步并发评估**：利用 `AsyncOpenAI` 并发调用 LLM，对候选节点批量评分。
- **多解输出**：同一组数字可返回多种不同的解法链路，便于分析。

---

## 📁 目录结构

```
24games/
├── README.md          # 当前文档
├── ToT_24games.py     # 主程序（ToT 推理 + 测试入口）
└── .env               # 环境变量（需自行创建）
```

---

## ⚙️ 环境准备

1. **Python 版本**：推荐 ≥ 3.10
2. **安装依赖**：
   ```bash
   pip install openai python-dotenv
   ```
3. **环境变量（.env）**：在 `24games` 目录下创建 `.env` 文件，内容示例：
   ```ini
   OPENAI_API_KEY=sk-xxx
   OPENAI_BASE_URL=https://api.openai.com/v1
   LLM_MODEL_ID=gpt-4o-mini
   ```

若已有全局环境变量，也可省略 `.env`。

---

## 🚀 快速开始

```bash
cd 24games
python ToT_24games.py
```

脚本末尾的 `test_case` 用于控制要解的四个数字，修改即可测试不同组合：

```python
if __name__ == "__main__":
    async def main():
        agent = ToTAgent(model="gpt-4o-mini", beam_width=100)
        test_case = [3, 3, 8, 8]
        solutions = await agent.solve_all(test_case)
```

### 可能输出

```
🤖 ToTAgent 正在寻找最佳解法: [3, 3, 8, 8]
--- 步骤 1 ---
生成了 24 种候选数字组.
已经收集 24 种途径
...
✨ 发现 5 种不同的解法 [3, 3, 8, 8]:
解法 1: 3 + 8 = 11  ->  11 + 8 = 19  ->  19 + 3 = 22 ...
```

---

## Tips
1.最好使用gpt-4o-mini以上的模型，负责很难完成评分的过程
2.目前只能手动输入测试用例，后续可能有时间的话，可以做一个答案准确率测试，并且部署到前端。
3.思路开阔：利用ToT思想可以和Plan and solve, ReAct,Reflexion结合

## 🔧 配置参数说明

| 参数 | 位置 | 默认值 | 说明 |
|------|------|--------|------|
| `model` | `ToTAgent` 初始化 | `gpt-4o-mini` | 使用的 LLM 模型 ID |
| `beam_width` | `ToTAgent` 初始化 | `100` | 每一层保留的最大节点数 |
| `temperature` | `GeneralAgent.think` | `0.7` | 评分阶段的随机性，略高可探索更多方案 |

> **提示**：`beam_width` 越大越容易找到更多解，但调用次数随之增加，成本也会提升。

---

## 🧠 树状思维流程速览

1. **初始化**：将四个数字作为根节点。 
2. **拓展候选**：对每一节点枚举所有合法的二元运算（加减乘除），规范化表达式，生成新节点。
3. **节点评估**：
   - 剩余 2 个数字 → 程序直接计算是否能得 24；
   - 剩余 ≥3 个数字 → 调用 LLM，请其输出 `Sure / Likely / Impossible` 评分。
4. **Beam 选择**：按评分降序，截取前 `beam_width` 个节点进入下一步。
5. **终局收集**：若节点只剩一个数字且等于 24，记录其历史算式并去重。

---

## 🛠️ 常见问题

| 问题 | 可能原因 | 解决方案 |
|------|----------|----------|
| `❌ LLM Error: ...` | API Key 或 Base URL 配置错误 | 检查 `.env` 或系统环境变量 |
| 长时间无响应 | `beam_width` 过大、网络延迟 | 减小 `beam_width` 或改用本地代理 |
| 找不到解 | 本身无解 / LLM 评分偏低 | 增大 `beam_width`、调整 `temperature`、多跑几次 |

---

## 📚 参考资料

- [Tree of Thoughts: Deliberate Problem Solving with Large Language Models](https://arxiv.org/abs/2305.10601)
- OpenAI 官方文档：<https://platform.openai.com/docs>

---

## 📜 许可证

延续仓库主项目的开源许可证（MIT）。如需在科研或商业项目中使用，请遵循相应条款并注明引用。

---

欢迎提交 Issue / PR 改进求解策略，或扩展到更多算术游戏 👋
