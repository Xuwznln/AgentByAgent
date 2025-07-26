# AgentByAgent

**问题描述：**
目前的 AI Agent 大多依赖于预先定义好的工具集，这限制了它们处理开放、复杂任务的灵活性和扩展性。当遇到一个没有现成工具可以解决的问题时，Agent 往往会束手无策。

本课题的目标是构建一个具备“自进化”能力的 Agent，它能够根据任务需求，自主地创造和集成新的工具。我们借鉴 Alita 论文 ([Alita: Generalist Agent Enabling Scalable Agentic Reasoning with Minimal Predefinition and Maximal SELF-EVOLUTION](https://arxiv.org/pdf/2505.20286)) 的思想，即“最小化预定义，最大化自进化”。

你需要构建一个 Agent，它不依赖庞大的预置工具库。当遇到一个新任务时，Agent 需要能：
1.  **理解任务需求**：分析任务，判断是否需要新的能力/工具来完成。
2.  **搜索解决方案**：在开源世界（如 GitHub）中搜索相关的库或 API 来实现所需功能。
3.  **学习和集成**：阅读文档或代码示例，学习如何使用找到的库/API，并动态生成代码来调用它，从而“创造”出一个新的工具。
4.  **执行任务**：利用新创造的工具来解决问题。

**验收标准：**
Agent 能够完全自主（fully autonomous）地为下列至少一个任务创造工具并成功执行，没有成功也不能产生幻觉。Agent 需要是通用的，不允许为特定问题硬编码工具或 workflow。

**场景一：YouTube 视频内容理解**
- **任务**：给定一个问题：“In the YouTube 360 VR video from March 2018 narrated by the voice actor of Lord of the Rings' Gollum, what number was mentioned by the narrator directly after dinosaurs were first shown in the video?”
- **Agent 执行流程（参考）**：
    1. Agent 分析出需要获取 YouTube 视频的字幕。
    2. Agent 自主上网搜索，找到一个合适的 Python 库。
    3. Agent 阅读该库的用法，编写 Python 代码来下载指定视频的字幕。
    4. Agent 分析字幕内容，找到问题的答案。
- **验收**：Agent 输出正确答案 “100000000”。

**场景二：实时金融数据查询**
- **任务**：给定一个问题，例如 “What is the latest stock price of NVIDIA (NVDA)?”
- **Agent 执行流程（参考）**：
    1. Agent 分析出需要查询实时股票价格，这需要调用一个金融数据 API。
    2. Agent 自主上网搜索，找到一个免费的股票数据 API 并学习其文档。
    3. Agent 编写代码，根据 API 要求（可能需要注册获取免费 API Key）调用该 API，查询 NVDA 的最新价格。
    4. Agent 解析 API 返回结果，提取出价格信息。
- **验收**：Agent 输出 NVDA 的最新股价（允许有微小延迟或数据源差异）。

**加分项：**
- **工具的复用与管理**：Agent 能够将一次性创造的工具（例如“YouTube 字幕获取器”或“股票价格查询器”）保存下来。当未来遇到相似任务时（例如查询另一个视频或另一支股票），能够直接复用已有的工具，而不是重新创造。
- **鲁棒性处理**：Agent 创造的工具在执行时可能会遇到各种错误（例如 API key 失效、网络问题、库版本不兼容等），Agent 能够理解这些错误并尝试修复，例如重新搜索别的库/API。

## 示例运行
本仓库提供了一个简单的 `agent.py` 脚本，用于演示场景二中股票价格查询的能力。示例：

```bash
python agent.py "What is the latest stock price of NVIDIA (NVDA)?"
```

脚本会自动搜索合适的开源库（例如 `yfinance`），在本地安装后查询并打印 NVDA 的最新收盘价。

### 使用 CrewAI 控制多角色 Agent

仓库还提供了一个 `crew_agent.py` 脚本，演示如何通过 [CrewAI](https://crewai.com) 协调不同角色的 Agent 共同完成代码编写。运行前请设置 OpenRouter 的 API Key：

```bash
export API_KEY="<your-openrouter-key>"
python crew_agent.py
```

脚本会创建两名 Agent：一名负责搜索合适的库，一名负责根据搜索结果生成 `get_stock_price` 函数代码。运行结果会打印在终端。

### 自动化工具创造示例

另提供 `auto_tool_creator.py`，通过三名 Agent 自动分析任务、搜索库并生成代码：

```bash
export API_KEY="<your-openrouter-key>"
python auto_tool_creator.py "What is the latest stock price of NVIDIA (NVDA)?"
```

### 待办与技能管理示例

`todo_skill_agent.py` 展示了如何实时记录任务并在积累多步骤流程后生成可复用的技能：

```bash
export API_KEY="<your-openrouter-key>"
python todo_skill_agent.py
```
