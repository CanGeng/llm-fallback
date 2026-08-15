# llm-fallback

Multi-level provider fallback for same-named models in the DeepSeek Harness.

A pool maps one model id to an ordered provider list. The first provider is the
primary; each later one is a fallback. The plugin keeps every step on the
primary (stable request prefix → KV cache reuse) and only fails over when the
current level is exhausted.

## Behavior

- `agent/request` picks the pool level for each step. A new step starts at the
  primary (level 0) **unless the explicitly requested provider is itself a pool
  member** — then it starts there and fails over forward. A recovered primary is
  therefore reused immediately on the next step. It also probes each candidate
  with `resolveCallConfig` and skips (recording) any that would fail before
  streaming — a missing route (`NO_ADAPTER`), an unknown model, or an
  unsupported reasoning effort.
- `agent/request-error` records each level's failure, then advances the level by
  returning `{ kind: 'retry' }`. When every level is exhausted it throws one
  aggregated error listing each level's failure (`provider: code — message`).

Same-provider retries happen first (via `dsh-llm-retry`, which sits outside this
plugin), so transient errors like `RATE_LIMIT` never trigger a failover and never
break the prefix. Only terminal errors (`QUOTA`, `AUTH`, …) or an exhausted
retry budget fail over.

## `retryPolicy.mode: always` is incompatible

`always` retries one provider forever and cannot be stopped from a listener, so
it conflicts with a terminating fallback chain:

- `alwaysPolicy: reject` (default) — refuse the provider, naming it and the
  model. Providers registered at load time are rejected when the plugin loads;
  routes activated later through settings (dormant routes) are rejected the
  first time a pool request would use them.
- `alwaysPolicy: degrade` — warn and cap that provider at `alwaysMaxRetries`
  same-provider retries before failing over.

In both modes, an `always` provider may never be the **last** level of a pool
(that can never terminate) — it is always rejected.

## Configuration

See `cordis.patch.yml`. The `pools` list maps a model id to its ordered
provider chain. The config is validated when the plugin loads: an unknown
`alwaysPolicy`, a pool without a non-empty `model`, or a pool whose `providers`
is not an array throws immediately instead of failing silently later.

```yaml
config:
  alwaysPolicy: reject
  alwaysMaxRetries: 2
  pools:
    - model: deepseek-v4-pro
      providers: [deepseek-official, ark-coding, ark-agent]
```

## Development

```sh
node --test   # or: npm test
```

## Install

```sh
dsh plugin --profile web add link:/path/to/llm-fallback
dsh web   # restart to load the new bundle layer
```
