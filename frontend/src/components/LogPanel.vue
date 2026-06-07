<script setup lang="ts">
import { ref, watch, nextTick, onUnmounted } from 'vue'

interface LogLine {
  id: number
  ts: string
  level: string
  text: string
}

const props = defineProps<{
  activeCase: string | null
  isRunning: boolean
}>()

const emit = defineEmits<{
  (e: 'stream-end'): void
}>()

const lines = ref<LogLine[]>([])
let counter = 0
let es: EventSource | null = null
let autoScroll = ref(true)
const logEl = ref<HTMLElement | null>(null)
const caseIndex = ref<{ idx: number; total: number; name: string } | null>(null)
const heartbeat = ref<{ elapsed_ms: number; tick: number } | null>(null)

const LEVEL_CLASS: Record<string, string> = {
  INFO: 'lvl-info',
  OK:   'lvl-ok',
  FAIL: 'lvl-fail',
  WARN: 'lvl-warn',
  HR:   'lvl-hr',
  HEAD: 'lvl-head',
  TBL:  'lvl-tbl',
  SAMP: 'lvl-samp',
  BADP: 'lvl-badp',
  BADF: 'lvl-badf',
  BLNK: 'lvl-blank',
}

function start(caseId: string | null) {
  // 关闭旧连接
  if (es) {
    es.close()
    es = null
  }
  lines.value = []
  counter = 0
  caseIndex.value = null
  heartbeat.value = null

  const url = caseId
    ? `/api/run?case=${encodeURIComponent(caseId)}`
    : '/api/run'
  es = new EventSource(url)

  es.onmessage = (ev) => {
    try {
      const payload = JSON.parse(ev.data)
      handle(payload)
    } catch (e) {
      console.error('bad SSE payload:', ev.data, e)
    }
  }

  es.onerror = (ev) => {
    // 浏览器会自动重连;但我们结束后希望它停
    // 所以用 'stream_end' / 'done' 显式关闭
    console.warn('SSE error', ev)
  }
}

function handle(payload: any) {
  if (payload.type === 'line') {
    lines.value.push({
      id: counter++,
      ts: payload.ts,
      level: payload.level,
      text: payload.text,
    })
    if (autoScroll.value) {
      nextTick(() => {
        if (logEl.value) {
          logEl.value.scrollTop = logEl.value.scrollHeight
        }
      })
    }
  } else if (payload.type === 'case_start') {
    caseIndex.value = { idx: payload.idx, total: payload.total, name: payload.case }
  } else if (payload.type === 'case_end') {
    // 可以加 divider
  } else if (payload.type === 'heartbeat') {
    heartbeat.value = { elapsed_ms: payload.elapsed_ms, tick: payload.tick }
  } else if (payload.type === 'done' || payload.type === 'stream_end' || payload.type === 'error') {
    if (es) {
      es.close()
      es = null
    }
    heartbeat.value = null
    emit('stream-end')
  }
}

function clear() {
  lines.value = []
  counter = 0
}

function toggleAutoScroll() {
  autoScroll.value = !autoScroll.value
}

function onScroll() {
  if (!logEl.value) return
  const el = logEl.value
  const atBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 40
  autoScroll.value = atBottom
}

watch(
  () => props.activeCase,
  (newVal, oldVal) => {
    if (newVal !== oldVal || props.isRunning) {
      start(newVal)
    }
  }
)

onUnmounted(() => {
  if (es) {
    es.close()
    es = null
  }
})
</script>

<template>
  <section class="log-panel">
    <div class="log-toolbar">
      <div class="left">
        <span v-if="caseIndex" class="case-pill">
          CASE {{ caseIndex.idx }} / {{ caseIndex.total }}
        </span>
        <span class="case-pill-name">{{ caseIndex?.name ?? '—' }}</span>
      </div>
      <div class="right">
        <span v-if="heartbeat" class="heartbeat">
          ⏱ {{ heartbeat.elapsed_ms }} ms
        </span>
        <button @click="toggleAutoScroll" :class="['autoscroll', { on: autoScroll }]">
          {{ autoScroll ? '⤓ auto-scroll' : '⤓ paused' }}
        </button>
        <button @click="clear" :disabled="lines.length === 0">clear</button>
      </div>
    </div>

    <div class="log-window" ref="logEl" @scroll="onScroll">
      <div v-if="lines.length === 0" class="empty">
        Click <b>Run All</b> in the sidebar, or pick a single case.
        <br /><br />
        Backend: <code>/api/run?case=tiny|small|low_willingness|scarce|large</code>
        <br />
        All cases: <code>/api/run</code>
      </div>
      <div
        v-for="line in lines"
        :key="line.id"
        :class="['line', LEVEL_CLASS[line.level] || 'lvl-info']"
      >
        <span class="ts">[{{ line.ts }}]</span>
        <span class="lvl">[{{ line.level.padEnd(4) }}]</span>
        <span class="text">{{ line.text || ' ' }}</span>
      </div>
    </div>
  </section>
</template>

<style scoped>
.log-panel {
  flex: 1;
  display: flex;
  flex-direction: column;
  min-width: 0;
  background: var(--bg);
}
.log-toolbar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 8px 14px;
  border-bottom: 1px solid var(--border);
  background: var(--bg-elev);
  font-size: 12px;
}
.left, .right {
  display: flex;
  align-items: center;
  gap: 10px;
}
.case-pill {
  background: var(--accent);
  color: white;
  padding: 2px 8px;
  border-radius: 10px;
  font-weight: 600;
  font-size: 11px;
}
.case-pill-name {
  color: var(--text-dim);
}
.heartbeat {
  color: var(--badge);
  font-variant-numeric: tabular-nums;
  font-size: 11px;
}
.autoscroll {
  font-size: 11px;
  padding: 4px 8px;
}
.autoscroll.on {
  border-color: var(--ok);
  color: var(--ok);
}
.log-window {
  flex: 1;
  overflow-y: auto;
  padding: 10px 14px 30px;
  font-size: 13px;
  line-height: 1.5;
  scroll-behavior: smooth;
}
.empty {
  color: var(--text-dim);
  padding: 40px;
  text-align: center;
  font-size: 13px;
}
.empty code {
  background: var(--bg-elev);
  padding: 1px 5px;
  border-radius: 3px;
  font-size: 11px;
}
.line {
  display: flex;
  gap: 8px;
  padding: 1px 0;
  white-space: pre;
  font-variant-numeric: tabular-nums;
}
.line .ts { color: var(--ts); flex-shrink: 0; }
.line .lvl { flex-shrink: 0; width: 70px; }
.line .text { flex: 1; color: var(--text); white-space: pre; }

.lvl-info .lvl { color: var(--info); }
.lvl-info .text { color: var(--text); }

.lvl-ok .lvl   { color: var(--ok); font-weight: 600; }
.lvl-ok .text  { color: var(--ok); }

.lvl-fail .lvl { color: var(--fail); font-weight: 600; }
.lvl-fail .text { color: var(--fail); }

.lvl-warn .lvl { color: var(--warn); }
.lvl-warn .text { color: var(--warn); }

.lvl-hr .lvl   { color: var(--hr); }
.lvl-hr .text  { color: var(--hr); }

.lvl-head .lvl { color: var(--accent); }
.lvl-head .text { color: var(--text); font-weight: 600; }

.lvl-tbl .lvl  { color: var(--text-dim); }
.lvl-tbl .text { color: var(--tbl); }

.lvl-samp .lvl { color: var(--text-dim); }
.lvl-samp .text { color: var(--text); }

.lvl-badp .lvl  { color: var(--badge); }
.lvl-badp .text { color: var(--badge); font-weight: 700; }

.lvl-badf .lvl  { color: var(--fail); }
.lvl-badf .text { color: var(--fail); font-weight: 700; }

.lvl-blank .ts,
.lvl-blank .lvl { visibility: hidden; }
.lvl-blank .text { color: var(--text); }
</style>
