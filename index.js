// Multi-level provider fallback for same-named models in the DeepSeek Harness.
//
// A pool maps one model id to an ordered list of provider routes. The first
// provider is the primary; each later one is a fallback. The plugin stays on
// the primary for every step so request prefixes (and KV cache) stay stable,
// and only fails over when the current level is exhausted:
//
//   - `agent/request`  — pick the pool level for this step (an explicit pool
//                        member starts there), and skip candidates that would
//                        fail before streaming (missing route, unknown model,
//                        unsupported reasoning effort).
//   - `agent/request-error` — record a streaming failure, then advance the level.
//                        When a level still has a fallback, return
//                        `{ kind: 'retry' }` so the loop re-issues the same step
//                        on the next level. When every level is exhausted, throw
//                        one aggregated error listing each level's failure.
//
// `retryPolicy.mode: 'always'` is structurally incompatible with a terminating
// fallback chain: it retries one provider forever and can never be stopped from
// a listener. The plugin therefore rejects it by default (fail loud), or — with
// `alwaysPolicy: 'degrade'` — caps an always provider at `alwaysMaxRetries`
// same-provider retries before failing over. In both modes, an `always` provider
// may never be the LAST level (that can never terminate).

export const name = 'llm-fallback'
export const inject = ['llm']

export function apply(ctx, config) {
  const alwaysPolicy = config?.alwaysPolicy ?? 'reject' // 'reject' | 'degrade'
  const alwaysMaxRetries = config?.alwaysMaxRetries ?? 2
  if (alwaysPolicy !== 'reject' && alwaysPolicy !== 'degrade') {
    throw new Error(`llm-fallback: invalid alwaysPolicy "${alwaysPolicy}" — expected "reject" or "degrade"`)
  }
  const pools = (config?.pools ?? []).map((p) => {
    // Fail loud on malformed pools: a missing model would register under
    // `undefined`, and a non-array providers (e.g. a string) would spread into
    // nonsense routes instead of being caught.
    if (typeof p?.model !== 'string' || p.model.length === 0) {
      throw new Error('llm-fallback: each pool needs a non-empty "model" string')
    }
    if (!Array.isArray(p.providers)) {
      throw new Error(`llm-fallback: pool "${p.model}" needs a "providers" array`)
    }
    return { model: p.model, providers: [...p.providers] }
  })
  const byModel = new Map(pools.map((p) => [p.model, p.providers]))

  // Load-time validation for providers that are already registered. Providers
  // activated later through settings (dormant pi-ai routes) are re-checked at
  // runtime in the `agent/request` listener.
  for (const { model, providers } of pools) {
    providers.forEach((provider, index) => {
      let policy
      try {
        policy = ctx.llm.providerRetryPolicy(provider)
      } catch (error) {
        if (error?.code === 'NO_ADAPTER') return // not registered yet; runtime fallback covers it
        throw error
      }
      if (policy?.mode !== 'always') return
      const isLast = index === providers.length - 1
      if (isLast || alwaysPolicy === 'reject') {
        throw new Error(
          `llm-fallback: provider "${provider}" (model "${model}") uses retryPolicy.mode=always, ` +
            (isLast
              ? 'which can never terminate as the last fallback level'
              : 'which is incompatible with fallback — set alwaysPolicy: "degrade" to allow it with a finite retry cap'),
        )
      }
      ctx.logger.warn(
        'llm-fallback: provider "%s" (model "%s") uses always; degrading to at most %d same-provider retries before failing over',
        provider,
        model,
        alwaysMaxRetries,
      )
    })
  }

  const state = new WeakMap() // agent -> { stepKey, model, providers, cursor, retries, failures, warned }
  const keyOf = (turn, step) => `${turn}:${step}`

  ctx.on('agent/request', async (payload, next) => {
    const proposal = await next()
    const providers = byModel.get(proposal?.model)
    if (!providers?.length) return proposal

    const stepKey = keyOf(payload.turn, payload.step)
    let s = state.get(payload.agent)
    if (!s || s.stepKey !== stepKey) {
      // New step: honor an explicit provider that is a pool member — start there
      // and fail over forward; anything else starts at the primary (index 0).
      const explicitIdx = providers.indexOf(proposal.provider)
      s = { stepKey, model: proposal.model, providers, cursor: explicitIdx >= 0 ? explicitIdx : 0, retries: 0, failures: [], warned: new Set() }
      state.set(payload.agent, s)
    }
    // Skip candidates that would fail before streaming (the prepareCall stage):
    // an unregistered route (NO_ADAPTER), an unknown model, or an unsupported
    // reasoning effort. Record each skip so the final error shows the full chain.
    while (s.cursor < s.providers.length) {
      const candidate = s.providers[s.cursor]
      try {
        await ctx.llm.resolveCallConfig({ ...proposal, provider: candidate }, payload.signal)
        break
      } catch (error) {
        // A turn abort during the probe must surface as an abort, not be
        // mistaken for a candidate failure (which would skip every provider
        // and throw a misleading aggregated error).
        if (payload.signal?.aborted) throw error
        const code = error?.code ?? 'UNKNOWN'
        const message = error?.message ?? ''
        ctx.logger.warn('llm-fallback: skipping provider "%s" (model "%s"): %s — %s', candidate, s.model, code, message)
        s.failures.push({ provider: candidate, failure: { code, message } })
        s.cursor += 1
      }
    }

    if (s.cursor >= s.providers.length) {
      const chain = s.failures.map((f) => `  - ${f.provider}: ${f.failure.code ?? 'UNKNOWN'} — ${f.failure.message ?? ''}`).join('\n')
      throw new Error(`llm-fallback: no provider in pool can serve model "${s.model}":\n${chain}`)
    }

    const provider = s.providers[s.cursor]

    // Re-check the chosen provider's retry policy at runtime — routes activated
    // after load (dormant pi-ai routes enabled through settings) were not seen
    // by the load-time validation. An `always` provider may never be the last
    // level (it can never terminate), and under alwaysPolicy: 'reject' it may
    // not serve a pool at all. In 'degrade' mode it serves here and is capped
    // in the agent/request-error listener.
    let policy
    try {
      policy = ctx.llm.providerRetryPolicy(provider)
    } catch (error) {
      if (error?.code !== 'NO_ADAPTER') throw error
    }
    if (policy?.mode === 'always') {
      if (s.cursor === s.providers.length - 1) {
        throw new Error(
          `llm-fallback: provider "${provider}" is the last fallback for model "${s.model}" with retryPolicy.mode=always; it can never terminate`,
        )
      }
      if (alwaysPolicy === 'reject') {
        throw new Error(
          `llm-fallback: provider "${provider}" (model "${s.model}") uses retryPolicy.mode=always, ` +
            'which is incompatible with fallback — set alwaysPolicy: "degrade" to allow it with a finite retry cap',
        )
      }
    }

    return provider === proposal.provider ? proposal : { ...proposal, provider }
  })

  ctx.on('agent/request-error', (payload, next) => {
    const s = state.get(payload.agent)
    if (!s || s.stepKey !== keyOf(payload.turn, payload.step)) return next()
    const cur = s.providers[s.cursor]
    const failure = payload.failure ?? {}

    // Record this level's failure (refresh, not append, while the same provider retries).
    const last = s.failures[s.failures.length - 1]
    if (last && last.provider === cur) last.failure = failure
    else s.failures.push({ provider: cur, failure })

    // An `always` provider is asked on every failed attempt. Let it retry
    // naturally (return next()) but keep our own cap; once exceeded, fail over.
    if (payload.retryPolicy?.mode === 'always') {
      if (!s.warned.has(cur)) {
        s.warned.add(cur)
        ctx.logger.warn(
          'llm-fallback: provider "%s" (model "%s") uses always; capping at %d same-provider retries',
          cur,
          s.model,
          alwaysMaxRetries,
        )
      }
      s.retries += 1
      if (s.retries <= alwaysMaxRetries) return next()
    }

    s.retries = 0
    s.cursor += 1
    if (s.cursor < s.providers.length) {
      ctx.logger.warn(
        'llm-fallback: model "%s" provider "%s" failed (%s) -> failing over to "%s"',
        s.model,
        cur,
        failure.code ?? 'UNKNOWN',
        s.providers[s.cursor],
      )
      return { kind: 'retry' }
    }

    const chain = s.failures
      .map((f) => `  - ${f.provider}: ${f.failure.code ?? 'UNKNOWN'} — ${f.failure.message ?? ''}`)
      .join('\n')
    throw new Error(`llm-fallback: all providers exhausted for model "${s.model}":\n${chain}`)
  })
}
