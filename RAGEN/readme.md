### 配置

位置：RAGEN/config/base.yaml

说明：base.yaml里面引入了其他的yaml配置文件（ppo_trainer.yaml和envs.yaml）

### 环境Env（以webshop为例）

**配置路径**

基础设置：RAGEN/ragen/env/webshop

注册环境：RAGEN/config/envs.yaml

训练时引入环境：RAGEN/config/base.yaml

**实现机制**

位置：RAGEN/external/webshop-minimal/webshop_minimal/env.py中的step

返回值说明：`state` 是一个包含了历史轨迹的字符串。它的结构大致为"前一步执行的动作 [SEP] 前一步的观测 [SEP] 刚执行的动作 [SEP] 当前的观测"。`status` 字典主要包含`reward`、`done`和`info`三个key：`reward`在除了“购买”之外的所有步骤（如搜索、点击商品、翻页等），它的值都保持为 0.0；`done`的值会一直保持 False，直到用户执行了购买操作；`info`来自于 `get_reward` 函数返回的 `info` 对象。这个 `info` 字典详细记录了奖励计算的细节，比如哪些商品属性匹配了任务目标，哪些没有匹配，以及各自的分数。

总结：该环境为结果为导向的reward，并不会对agent的探索过程轨迹进行reward。

### RL逻辑

位置：RAGEN/ragen/trainer/agent_trainer.py中的RayAgentTrainer的fit

### Agent Rollout逻辑

**es_manager**

作用：根据大模型的输出给出工具的调用结果。

1. `_init_envs`函数为配置文件中定义的每一种环境类型(tag)，创建n_group个组。每个组都包含 group_size个环境实例。
2. `step`函数通过 `env_input['env_id']` 从 `self.envs` 列表中找到对应的环境实例 `entry`。调用 `_extract_map_valid_actions` 函数检查 LLM 给出的动作（`env_input['actions']`）是否是当前环境允许的合法动作。如果 LLM 生成了无效或非法的动作，会给这个环境的 `penalty` 加上一个惩罚分数。然后调用内部的`_execute_actions`辅助函数循环执行上一步验证过的合法动作。然后一个接一个地调用 `env.step(action)`。如果在中途某一步环境返回了 `done=True`（表示任务成功或失败），循环会提前中止。它会累加这个回合内所有动作产生的奖励 `acc_reward`。在动作执行完毕后，函数会调用 `_log_env_state` 来记录这次交互的结果。这一步会更新环境的状态（如已执行动作数、是否成功 `terminated` 或失败 `truncated`），并通过 `_update_cache_history` 将包含本次动作、奖励、LLM响应等信息的完整回合数据更新到历史记录中。最后，函数会判断该环境是否已结束（`turn_done`），只有那些尚未结束的环境的最新状态才会被加入到返回列表 `env_outputs` 中，以便进行下一轮的决策。

**ctx_manager**

作用：管理大模型与环境之间的输入输出。

1. `get_lm_inputs`函数将env中的history转化为大模型输入的messages。【**因为作者在此处嵌入了很多自己写的prompt，比如回答格式和长度约束等等，所以在使用的时候需要自定义messsage的格式**】
2. `get_env_inputs`函数将大模型的输出转化为env_inputs。

**逻辑**

位置：RAGEN/ragen/llm_agent/agent_proxy.py中的LLMAgentProxy的rollout

核心代码：

```python
for i in range(self.config.agent_proxy.max_turn):
			lm_inputs: DataProto = ctx_manager.get_lm_inputs(env_outputs, prepare_for_update=False)
			lm_inputs.meta_info = dataproto.meta_info # TODO: setup vllm early stop when max length is reached. make sure this can be done
			lm_outputs: DataProto = self.generate_sequences(lm_inputs)
			env_inputs: List[Dict] = ctx_manager.get_env_inputs(lm_outputs)
			env_outputs: List[Dict] = es_manager.step(env_inputs)
			if len(env_outputs) == 0: # all finished
				break
```

### 总结

1. **高度的定制化要求与陡峭的学习曲线**： RAGEN 作为一个通用的 RL Agent 框架，要求用户深度参与三个核心模块的定制：
   - **环境 (Env) 适配**：必须将具体的 Tool-Call 任务（如 API 调用、代码执行）封装成符合 `gymnasium` 接口的 RL 环境，这需要自行定义复杂的状态表示、动作空间映射和奖励函数【此处是能够接受而且必须要做的配置～】。
   - **执行与状态管理 (es_manager)**：需要通过 YAML 精确配置环境的并行数量、分组、种子等，以服务于框架为 RL 实验设计的评估和采样逻辑【此处得重写我应该去存储什么样的环境输出，此处作者每一个轨迹都存储了reward，但其实对tool_call的轨迹来说是不必要的】。
   - **上下文构建 (ctx_manager)**：需要深度定制 LLM 的 Prompt 格式、历史对话的拼接逻辑以及模型响应的解析方式，以适配特定的模型和任务范式（例如 CoT）【作者在此处加入了许多自己的提示词约束，比如格式和长度约束，因此得自己去重写构建大模型输入messages的函数】。
2. **示例与通用 Tool-Call 场景的脱节**： 框架提供的“推箱子”这类网格世界（Grid World）示例，虽然很好地展示了其 RL 核心能力，但与典型的 Tool-Call 任务（涉及 API 调用、文本解析、非结构化状态）差异巨大。
3. **分散的配置管理**：配置项分散在多个 YAML 文件中。对于初次接触的用户来说，追踪和理解完整的配置链路需要花费额外的时间成本。

总体上，这个框架的设计哲学似乎更偏向于为**研究人员**提供一个**灵活、可扩展的底层 RL-Agent 实验平台**，而不是一个让**开发者**能够“开箱即用”的业务框架。

