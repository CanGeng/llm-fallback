# llm-fallback

[English](README.md) | 中文

在 DeepSeek Harness 中为同名模型提供多级 provider 兜底（fallback）。

一个 pool 把一个模型 id 映射到一个**有序**的 provider 列表：第一个是主链路，后面依次是兜底。插件让每一步都优先走主链路（请求前缀稳定 → KV cache 复用），只有当前一级彻底用尽时才切换。

## 行为

- `agent/request` 为每一步选择 pool 的层级。新的一步从主链路（第 0 级）开始，**除非显式请求的 provider 本身就是 pool 成员**——那就从它开始、向后兜底。主链路恢复后，下一步会立即回到它。它还会用 `resolveCallConfig` 探测每个候选，跳过（并记录）那些在流式之前就会失败的——缺失路由（`NO_ADAPTER`）、未知模型、或不支持的推理强度（`UNSUPPORTED_REASONING_EFFORT`）。
- `agent/request-error` 记录每一级的失败，然后返回 `{ kind: 'retry' }` 推进到下一级。当所有层级用尽时，抛出一个聚合错误，列出每一级的失败（`provider: code — message`）。

同 provider 的重试先发生（经由 `dsh-llm-retry`，它位于本插件之外），所以 `RATE_LIMIT` 这类暂时性错误不会触发切换、也不会打断前缀。只有终止性错误（`QUOTA`、`AUTH` 等）或重试预算用尽才会兜底切换。

## `retryPolicy.mode: always` 不兼容

`always` 会无限重试同一个 provider，且无法从监听器侧停止，因此与「最终必须收敛」的兜底链冲突：

- `alwaysPolicy: reject`（默认）——拒绝该 provider，并点名它和模型。加载时已注册的 provider 在插件加载时即被拒绝；之后经 settings 激活的路由（休眠路由）会在 pool 请求首次用到它时被拒绝。
- `alwaysPolicy: degrade`——警告并把该 provider 限制为最多 `alwaysMaxRetries` 次同 provider 重试，然后兜底切换。

两种模式下，`always` provider 都绝不能是 pool 的**最后一级**（那永远无法收敛）——它总会被拒绝。

## 配置

见 `cordis.patch.yml`。`pools` 列表把一个模型 id 映射到有序的 provider 链。配置在插件加载时校验：未知的 `alwaysPolicy`、缺非空 `model` 的 pool、或 `providers` 不是数组的 pool，都会立即抛错，而不是之后静默失败。

```yaml
config:
  alwaysPolicy: reject
  alwaysMaxRetries: 2
  pools:
    - model: deepseek-v4-pro
      providers: [deepseek-official, ark-coding, ark-agent]
```

## 开发

```sh
node --test   # 或：npm test
```

## 安装

```sh
dsh plugin --profile web add link:/path/to/llm-fallback
dsh web   # 重启以加载新的 bundle 层
```
