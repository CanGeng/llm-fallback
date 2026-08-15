// Tests for llm-fallback. Runs with `node --test` — no external dependencies.
// The cordis context and the `llm` service are stubbed; the two listeners the
// plugin registers (`agent/request`, `agent/request-error`) are captured and
// invoked directly, mirroring how dsh-agent-loop dispatches them.

import { test } from 'node:test'
import assert from 'node:assert/strict'
import { apply, name, inject } from './index.js'

const noAdapter = (provider) => Object.assign(new Error(`no adapter registered for provider "${provider}"`), { code: 'NO_ADAPTER' })
const NORMAL = { mode: 'normal', maxRetries: 2, retryableCodes: ['RATE_LIMIT'] }
const ALWAYS = { mode: 'always' }

// policies: provider -> retry policy (missing key = unregistered route, throws NO_ADAPTER)
// probeFailures: provider -> error thrown by resolveCallConfig (prepareCall-stage failure)
function makeCtx({ policies = {}, probeFailures = {} } = {}) {
  const listeners = {}
  const warns = []
  const probeCalls = []
  const ctx = {
    llm: {
      providerRetryPolicy(provider) {
        if (!(provider in policies)) throw noAdapter(provider)
        return policies[provider]
      },
      async resolveCallConfig(config) {
        probeCalls.push(config.provider)
        const failure = probeFailures[config.provider]
        if (failure) throw failure
        return config
      },
    },
    logger: { warn: (...args) => warns.push(args) },
    on(event, listener) {
      listeners[event] = listener
    },
  }
  return { ctx, listeners, warns, probeCalls, policies, probeFailures }
}

function request(listeners, { agent, turn = 1, step = 1, proposal, signal }) {
  return listeners['agent/request'](
    { agent, turn, step, signal: signal ?? new AbortController().signal },
    async () => proposal,
  )
}

// The request-error listener is synchronous, so it throws rather than returns a
// rejected promise; wrapping in an async helper normalizes both to rejections.
async function requestError(listeners, { agent, turn = 1, step = 1, provider, failure = {}, retryPolicy, next }) {
  return listeners['agent/request-error'](
    { agent, turn, step, provider, failure, retryPolicy, signal: new AbortController().signal },
    next ?? (async () => undefined),
  )
}

const POOL = { model: 'deepseek-v4-pro', providers: ['deepseek-official', 'ark-coding', 'ark-agent'] }

// ---------------------------------------------------------------- module shape

test('exports cordis plugin name and inject', () => {
  assert.equal(name, 'llm-fallback')
  assert.deepEqual(inject, ['llm'])
})

// ------------------------------------------------------- apply-time validation

test('applies with defaults when config is omitted', async () => {
  const { ctx, listeners } = makeCtx()
  apply(ctx)
  const agent = {}
  const proposal = { provider: 'anything', model: 'unlisted-model' }
  const result = await request(listeners, { agent, proposal })
  assert.equal(result, proposal) // no pool -> untouched, same reference
})

test('reject policy (default) refuses to load a registered always provider', () => {
  const { ctx } = makeCtx({ policies: { 'deepseek-official': ALWAYS, 'ark-coding': NORMAL } })
  assert.throws(
    () => apply(ctx, { pools: [{ model: 'm', providers: ['deepseek-official', 'ark-coding'] }] }),
    (err) => {
      assert.match(err.message, /"deepseek-official"/)
      assert.match(err.message, /"m"/)
      assert.match(err.message, /incompatible with fallback/)
      return true
    },
  )
})

test('an always provider as the last level is refused even under degrade', () => {
  const { ctx } = makeCtx({ policies: { a: NORMAL, b: ALWAYS } })
  assert.throws(
    () => apply(ctx, { alwaysPolicy: 'degrade', pools: [{ model: 'm', providers: ['a', 'b'] }] }),
    /never terminate/,
  )
})

test('unregistered provider at load time is skipped (NO_ADAPTER), not an error', () => {
  const { ctx } = makeCtx({ policies: { a: NORMAL } }) // b not registered
  apply(ctx, { pools: [{ model: 'm', providers: ['a', 'b'] }] })
})

test('degrade policy loads an always provider with a warning', () => {
  const { ctx, warns } = makeCtx({ policies: { a: ALWAYS, b: NORMAL } })
  apply(ctx, { alwaysPolicy: 'degrade', alwaysMaxRetries: 5, pools: [{ model: 'm', providers: ['a', 'b'] }] })
  assert.equal(warns.length, 1)
  assert.match(warns[0][0], /degrading to at most %d same-provider retries/)
  assert.deepEqual(warns[0].slice(1), ['a', 'm', 5])
})

test('a pool of normal providers loads silently', () => {
  const { ctx, warns } = makeCtx({ policies: { a: NORMAL, b: NORMAL } })
  apply(ctx, { pools: [{ model: 'm', providers: ['a', 'b'] }] })
  assert.equal(warns.length, 0)
})

test('invalid alwaysPolicy value throws instead of silently degrading', () => {
  const { ctx } = makeCtx()
  assert.throws(() => apply(ctx, { alwaysPolicy: 'rejct' }), /invalid alwaysPolicy "rejct"/)
})

test('malformed pools fail loud', () => {
  const { ctx } = makeCtx()
  assert.throws(() => apply(ctx, { pools: [{ providers: ['a'] }] }), /non-empty "model" string/)
  assert.throws(() => apply(ctx, { pools: [{ model: '', providers: ['a'] }] }), /non-empty "model" string/)
  assert.throws(() => apply(ctx, { pools: [{ model: 'm' }] }), /"providers" array/)
  assert.throws(() => apply(ctx, { pools: [{ model: 'm', providers: 'ark' }] }), /"providers" array/)
})

// ---------------------------------------------------------------- agent/request

test('an explicit provider that is a pool member starts there (proposal returned as-is)', async () => {
  const { ctx, listeners } = makeCtx()
  apply(ctx, { pools: [POOL] })
  const proposal = { provider: 'ark-coding', model: POOL.model }
  const result = await request(listeners, { agent: {}, proposal })
  assert.equal(result, proposal)
})

test('a non-member or absent provider starts at the primary, preserving other fields', async () => {
  const { ctx, listeners } = makeCtx()
  apply(ctx, { pools: [POOL] })
  const proposal = { provider: '', model: POOL.model, reasoningEffort: 'high' }
  const result = await request(listeners, { agent: {}, proposal })
  assert.notEqual(result, proposal)
  assert.deepEqual(result, { provider: 'deepseek-official', model: POOL.model, reasoningEffort: 'high' })
})

test('a pool with an empty providers array leaves the proposal untouched', async () => {
  const { ctx, listeners } = makeCtx()
  apply(ctx, { pools: [{ model: 'm', providers: [] }] })
  const proposal = { provider: 'x', model: 'm' }
  assert.equal(await request(listeners, { agent: {}, proposal }), proposal)
})

test('an undefined proposal (no downstream provider/model yet) passes through', async () => {
  const { ctx, listeners } = makeCtx()
  apply(ctx, { pools: [POOL] })
  assert.equal(await request(listeners, { agent: {}, proposal: undefined }), undefined)
})

test('prepareCall-stage failures are skipped and the next candidate serves', async () => {
  const { ctx, listeners, warns } = makeCtx({ probeFailures: { 'deepseek-official': noAdapter('deepseek-official') } })
  apply(ctx, { pools: [POOL] })
  const result = await request(listeners, { agent: {}, proposal: { provider: 'deepseek-official', model: POOL.model } })
  assert.equal(result.provider, 'ark-coding')
  assert.equal(warns.length, 1)
  assert.match(warns[0][0], /skipping provider "%s"/)
  assert.deepEqual(warns[0].slice(1), ['deepseek-official', POOL.model, 'NO_ADAPTER', 'no adapter registered for provider "deepseek-official"'])
})

test('when every candidate fails the probe, one aggregated error lists the full chain', async () => {
  const { ctx, listeners } = makeCtx({
    probeFailures: {
      'deepseek-official': noAdapter('deepseek-official'),
      'ark-coding': Object.assign(new Error('unknown model'), { code: 'UNKNOWN_MODEL' }),
      'ark-agent': Object.assign(new Error('effort unsupported'), { code: 'UNSUPPORTED_EFFORT' }),
    },
  })
  apply(ctx, { pools: [POOL] })
  await assert.rejects(
    request(listeners, { agent: {}, proposal: { provider: 'deepseek-official', model: POOL.model } }),
    (err) => {
      assert.match(err.message, /no provider in pool can serve model "deepseek-v4-pro"/)
      assert.match(err.message, /- deepseek-official: NO_ADAPTER — /)
      assert.match(err.message, /- ark-coding: UNKNOWN_MODEL — unknown model/)
      assert.match(err.message, /- ark-agent: UNSUPPORTED_EFFORT — effort unsupported/)
      return true
    },
  )
})

test('a turn abort during the probe is rethrown, not mistaken for candidate failure', async () => {
  const abortError = Object.assign(new Error('The operation was aborted'), { name: 'AbortError' })
  const { ctx, listeners } = makeCtx({ probeFailures: { 'deepseek-official': abortError } })
  apply(ctx, { pools: [POOL] })
  const controller = new AbortController()
  controller.abort()
  await assert.rejects(
    request(listeners, {
      agent: {},
      proposal: { provider: 'deepseek-official', model: POOL.model },
      signal: controller.signal,
    }),
    (err) => err === abortError,
  )
})

test('runtime: a late-registered always provider as the last level is refused', async () => {
  // b is dormant at load time, then activated later through settings.
  const { ctx, listeners, policies } = makeCtx({
    policies: { a: NORMAL },
    probeFailures: { a: noAdapter('a') },
  })
  apply(ctx, { pools: [{ model: 'm', providers: ['a', 'b'] }] })
  policies.b = ALWAYS
  await assert.rejects(
    request(listeners, { agent: {}, proposal: { provider: 'a', model: 'm' } }),
    /last fallback for model "m".*never terminate/,
  )
})

test('runtime: reject policy is enforced for a late-registered always provider', async () => {
  const { ctx, listeners, policies } = makeCtx() // nothing registered at load
  apply(ctx, { pools: [{ model: 'm', providers: ['a', 'b'] }] })
  policies.a = ALWAYS
  policies.b = NORMAL
  await assert.rejects(
    request(listeners, { agent: {}, proposal: { provider: 'a', model: 'm' } }),
    (err) => {
      assert.match(err.message, /"a" \(model "m"\)/)
      assert.match(err.message, /incompatible with fallback/)
      return true
    },
  )
})

test('re-dispatch within one step keeps the cursor; a new step resets to the primary', async () => {
  const { ctx, listeners, probeCalls } = makeCtx({ probeFailures: { 'deepseek-official': noAdapter('deepseek-official') } })
  apply(ctx, { pools: [POOL] })
  const agent = {}
  const first = await request(listeners, { agent, proposal: { provider: 'deepseek-official', model: POOL.model } })
  assert.equal(first.provider, 'ark-coding')
  // Same step re-dispatched (e.g. after a retry): cursor kept, no re-probe of the skipped provider.
  const again = await request(listeners, { agent, proposal: { provider: 'deepseek-official', model: POOL.model } })
  assert.equal(again.provider, 'ark-coding')
  assert.deepEqual(probeCalls.filter((p) => p === 'deepseek-official').length, 1)
  // Next step starts over at the primary (and probes it again).
  const nextStep = await request(listeners, { agent, step: 2, proposal: { provider: 'deepseek-official', model: POOL.model } })
  assert.equal(nextStep.provider, 'ark-coding')
  assert.deepEqual(probeCalls.filter((p) => p === 'deepseek-official').length, 2)
})

// ----------------------------------------------------------- agent/request-error

test('request-error without matching state delegates to next()', async () => {
  const { ctx, listeners } = makeCtx()
  apply(ctx, { pools: [POOL] })
  const agent = {}
  // No request yet for this agent.
  assert.equal(await requestError(listeners, { agent, provider: 'x', next: async () => 'downstream' }), 'downstream')
  // Request on step 1, then an error event for a different step.
  await request(listeners, { agent, proposal: { provider: 'deepseek-official', model: POOL.model } })
  assert.equal(await requestError(listeners, { agent, step: 99, provider: 'deepseek-official', next: async () => 'downstream' }), 'downstream')
})

test('a streaming failure fails over to the next level with { kind: "retry" }', async () => {
  const { ctx, listeners, warns } = makeCtx()
  apply(ctx, { pools: [POOL] })
  const agent = {}
  await request(listeners, { agent, proposal: { provider: 'deepseek-official', model: POOL.model } })
  let nextCalled = false
  const action = await requestError(listeners, {
    agent,
    provider: 'deepseek-official',
    failure: { code: 'QUOTA', message: 'quota exhausted' },
    retryPolicy: NORMAL,
    next: async () => {
      nextCalled = true
    },
  })
  assert.deepEqual(action, { kind: 'retry' })
  assert.equal(nextCalled, false) // failover short-circuits downstream listeners
  assert.equal(warns.length, 1)
  assert.match(warns[0][0], /failing over to "%s"/)
  assert.deepEqual(warns[0].slice(1), [POOL.model, 'deepseek-official', 'QUOTA', 'ark-coding'])
  // The loop re-issues the step; the plugin now serves the next level.
  const result = await request(listeners, { agent, proposal: { provider: 'deepseek-official', model: POOL.model } })
  assert.equal(result.provider, 'ark-coding')
})

test('when every level is exhausted, one aggregated error lists each failure', async () => {
  const { ctx, listeners } = makeCtx()
  apply(ctx, { pools: [{ model: 'm', providers: ['a', 'b'] }] })
  const agent = {}
  await request(listeners, { agent, proposal: { provider: 'a', model: 'm' } })
  assert.deepEqual(await requestError(listeners, { agent, provider: 'a', failure: { code: 'QUOTA', message: 'quota' } }), { kind: 'retry' })
  await request(listeners, { agent, proposal: { provider: 'a', model: 'm' } })
  await assert.rejects(
    Promise.resolve(requestError(listeners, { agent, provider: 'b', failure: { code: 'AUTH', message: 'bad key' } })),
    (err) => {
      assert.match(err.message, /all providers exhausted for model "m"/)
      assert.match(err.message, /- a: QUOTA — quota/)
      assert.match(err.message, /- b: AUTH — bad key/)
      return true
    },
  )
})

test('an always provider retries downstream up to alwaysMaxRetries, then fails over', async () => {
  const { ctx, listeners, warns } = makeCtx({ policies: { a: ALWAYS, b: NORMAL } })
  apply(ctx, { alwaysPolicy: 'degrade', alwaysMaxRetries: 2, pools: [{ model: 'm', providers: ['a', 'b'] }] })
  const agent = {}
  await request(listeners, { agent, proposal: { provider: 'a', model: 'm' } })

  const sentinel = { kind: 'retry', from: 'downstream' }
  let nextCalls = 0
  const next = async () => {
    nextCalls += 1
    return sentinel
  }
  const fail = () => requestError(listeners, { agent, provider: 'a', failure: { code: 'RATE_LIMIT' }, retryPolicy: ALWAYS, next })

  assert.equal(await fail(), sentinel) // retry 1 within the cap
  assert.equal(await fail(), sentinel) // retry 2 within the cap
  const third = await fail() // cap exceeded -> failover
  assert.deepEqual(third, { kind: 'retry' })
  assert.equal(nextCalls, 2)
  // The "capping" warning is logged once per provider, not per failure.
  assert.equal(warns.filter((w) => /capping at %d same-provider retries/.test(w[0])).length, 1)
  // Failover landed on b.
  const result = await request(listeners, { agent, proposal: { provider: 'a', model: 'm' } })
  assert.equal(result.provider, 'b')
})

test('repeated failures of one provider refresh its chain entry instead of appending', async () => {
  const { ctx, listeners } = makeCtx({ policies: { a: ALWAYS, b: NORMAL } })
  apply(ctx, { alwaysPolicy: 'degrade', alwaysMaxRetries: 1, pools: [{ model: 'm', providers: ['a', 'b'] }] })
  const agent = {}
  await request(listeners, { agent, proposal: { provider: 'a', model: 'm' } })
  await requestError(listeners, { agent, provider: 'a', failure: { code: 'RATE_LIMIT', message: 'first' }, retryPolicy: ALWAYS })
  await requestError(listeners, { agent, provider: 'a', failure: { code: 'RATE_LIMIT', message: 'latest' }, retryPolicy: ALWAYS }) // cap exceeded -> failover
  await request(listeners, { agent, proposal: { provider: 'a', model: 'm' } })
  await assert.rejects(
    Promise.resolve(requestError(listeners, { agent, provider: 'b', failure: { code: 'AUTH', message: 'bad key' } })),
    (err) => {
      assert.equal(err.message.match(/- a:/g).length, 1, 'one chain entry for provider a')
      assert.match(err.message, /- a: RATE_LIMIT — latest/) // refreshed to the latest failure
      assert.match(err.message, /- b: AUTH — bad key/)
      return true
    },
  )
})

test('probe-stage skips and streaming failures combine into one full failure chain', async () => {
  const { ctx, listeners } = makeCtx({ probeFailures: { 'deepseek-official': noAdapter('deepseek-official') } })
  apply(ctx, { pools: [POOL] })
  const agent = {}
  const first = await request(listeners, { agent, proposal: { provider: 'deepseek-official', model: POOL.model } })
  assert.equal(first.provider, 'ark-coding')
  assert.deepEqual(
    await requestError(listeners, { agent, provider: 'ark-coding', failure: { code: 'QUOTA', message: 'quota' } }),
    { kind: 'retry' },
  )
  const second = await request(listeners, { agent, proposal: { provider: 'deepseek-official', model: POOL.model } })
  assert.equal(second.provider, 'ark-agent')
  await assert.rejects(
    Promise.resolve(requestError(listeners, { agent, provider: 'ark-agent', failure: { code: 'AUTH', message: 'bad key' } })),
    (err) => {
      assert.match(err.message, /all providers exhausted for model "deepseek-v4-pro"/)
      assert.match(err.message, /- deepseek-official: NO_ADAPTER — /) // probe-stage skip recorded
      assert.match(err.message, /- ark-coding: QUOTA — quota/)
      assert.match(err.message, /- ark-agent: AUTH — bad key/)
      return true
    },
  )
})

test('state is tracked per agent', async () => {
  const { ctx, listeners } = makeCtx()
  apply(ctx, { pools: [{ model: 'm', providers: ['a', 'b'] }] })
  const agentA = {}
  const agentB = {}
  await request(listeners, { agent: agentA, proposal: { provider: 'a', model: 'm' } })
  await request(listeners, { agent: agentB, proposal: { provider: 'a', model: 'm' } })
  // agentA fails over; agentB must stay on the primary.
  await requestError(listeners, { agent: agentA, provider: 'a', failure: { code: 'QUOTA' } })
  assert.equal((await request(listeners, { agent: agentA, proposal: { provider: 'a', model: 'm' } })).provider, 'b')
  assert.equal((await request(listeners, { agent: agentB, proposal: { provider: 'a', model: 'm' } })).provider, 'a')
})
