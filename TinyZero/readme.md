### 核心逻辑

#### 奖励函数【可自定义】

位置：TinyZero/verl/utils/reward_score/countdown.py

说明：在给定format_score=0.1和score=1的情况下，首先通过正则表达式从answer中提取equation并验证其格式是否符合要求，符合则获得格式奖励分数0.1；然后通过eval计算equation的结果，若正确则获得剩余分数0.9，最终总分为1。

#### RL角色定义

位置：TinyZero/verl/workers/megatron_workers.py

角色定义：在Actor、RefPolicy、Critic、RewardModel和Rollout这几个组件中，Rollout是根据prompt生成response的过程，可能由Actor或RefPolicy执行（verl实现是基于Actor进行rollout的on-policy策略）。因此可以将这些类归纳为ActorRolloutRefWorker、CriticWorker和RewardModelWorker三种工作类型。

#### RL逻辑【PPO or GRPO】

位置：TinyZero/verl/trainer/ppo/ray_trainer.py中的RayPPOTrainer中的fit方法

```python
gen_batch = batch.pop(batch_keys=['input_ids', 'attention_mask', 'position_ids'])

# 根据prompt生成对应的response，response个数由self.config.actor_rollout_ref.rollout.n控制
gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
# 通过uid跟踪prompt，方便GRPO中同组计算优势值
batch.non_tensor_batch['uid'] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))],dtype=object)

# 重复n次prompt
# prompt+response 1
# ...
# prompt+response n
batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
batch = batch.union(gen_batch_output)

# 分布式训练的负载均衡
self._balance_batch(batch, metrics=metrics)

# 计算RefPolicy的log_prob
if self.use_reference_policy:
  ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
  batch = batch.union(ref_log_prob)

# 计算baseline分数【PPO】
if self.use_critic:
  values = self.critic_wg.compute_values(batch)
  batch = batch.union(values)

if self.use_rm:
  # reward model
  reward_tensor = self.rm_wg.compute_rm_score(batch)
  batch = batch.union(reward_tensor)
  # rule-based reward
  reward_tensor = self.reward_fn(batch)
  batch.batch['token_level_scores'] = reward_tensor

# 计算KL散度
if not self.config.actor_rollout_ref.actor.use_kl_loss:
  batch, kl_metrics =apply_kl_penalty(batch,kl_ctrl=self.kl_ctrl,kl_penalty=self.config.algorithm.kl_penalty)
	metrics.update(kl_metrics)
else:
  batch.batch['token_level_rewards'] = batch.batch['token_level_scores']

# 计算优势值
batch = compute_advantage(batch,adv_estimator=self.config.algorithm.adv_estimator,
                          gamma=self.config.algorithm.gamma,
                          lam=self.config.algorithm.lam,
                          num_repeat=self.config.actor_rollout_ref.rollout.n)

# 更新Critic模型
if self.use_critic:
  critic_output = self.critic_wg.update_critic(batch)
  critic_output_metrics = reduce_metrics(critic_output.meta_info['metrics'])
  metrics.update(critic_output_metrics)

# 更新Actor模型
if self.config.trainer.critic_warmup <= self.global_steps:
  actor_output = self.actor_rollout_wg.update_actor(batch)
  actor_output_metrics = reduce_metrics(actor_output.meta_info['metrics'])
  metrics.update(actor_output_metrics)
```

#### 优势值计算【GAE or GRPO】

位置：TinyZero/verl/trainer/ppo/core_algos.py



### 数据准备

**原始数据**：https://huggingface.co/datasets/Jiayi-Pan/Countdown-Tasks-3to4/viewer/default/train?p=4903&views%5B%5D=train

**任务描述**：给定一组数字（每个数字仅能使用一次）和四种基本运算（加、减、乘、除），通过合理的运算顺序组合，最终计算结果必须等于目标值 `target`（例如经典的24点游戏中 `target = 24`）。

**命令**：

```bash
python ./examples/data_preprocess/countdown.py --local_dir ./data/countdown
```

**对话构造**：

```python
A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer.
User: Using the numbers {numbers}, create an equation that equals {target}. You can use basic arithmetic operations (+, -, *, /) and each number can only be used once. Show your work in <think> </think> tags. And return the final answer in <answer> </answer> tags, for example <answer> (1 + 2) / 3 </answer>.
Assistant: Let me solve this step by step.
<think>
```

**样例展示**：

```python
target: 50
nums: [30 50 53 82]
data_source: countdown
prompt: [{'content': 'A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer.\nUser: Using the numbers [30, 50, 53, 82], create an equation that equals 50. You can use basic arithmetic operations (+, -, *, /) and each number can only be used once. Show your work in <think> </think> tags. And return the final answer in <answer> </answer> tags, for example <answer> (1 + 2) / 3 </answer>.\nAssistant: Let me solve this step by step.\n<think>', 'role': 'user'}]
ability: math
reward_model: {'ground_truth': {'numbers': array([30, 50, 53, 82]), 'target': 50}, 'style': 'rule'}
extra_info: {'index': 36680, 'split': 'train'}
```

### 模型准备

**模型**：https://huggingface.co/Qwen/Qwen2.5-3B

**命令**：

```bash
huggingface-cli download --resume-download Qwen/Qwen2.5-3B --local-dir ./models
```

### 训练代码
