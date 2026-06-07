<script setup lang="ts">
import { ref, onMounted } from 'vue'
import LogPanel from './components/LogPanel.vue'
import CaseSidebar from './components/CaseSidebar.vue'

interface CaseInfo {
  id: string
  name: string
  factory: string
}

const cases = ref<CaseInfo[]>([])
const activeCase = ref<string | null>(null)
const isRunning = ref(false)
const backendOk = ref(false)

async function fetchCases() {
  try {
    const resp = await fetch('/api/cases')
    cases.value = await resp.json()
    backendOk.value = true
  } catch (e) {
    backendOk.value = false
    console.error('failed to fetch cases:', e)
  }
}

onMounted(fetchCases)

function startRun(caseId: string | null) {
  activeCase.value = caseId
  isRunning.value = true
}

function onStreamEnd() {
  isRunning.value = false
}
</script>

<template>
  <div class="layout">
    <header class="topbar">
      <div class="brand">
        <span class="logo">▣</span>
        <span class="title">Courier Dispatch Solver — Demo</span>
      </div>
      <div class="status">
        <span :class="['dot', backendOk ? 'on' : 'off']"></span>
        <span>{{ backendOk ? 'backend connected' : 'backend offline' }}</span>
      </div>
    </header>

    <main class="main">
      <CaseSidebar
        :cases="cases"
        :active-case="activeCase"
        :is-running="isRunning"
        @run-all="startRun(null)"
        @run-case="(id) => startRun(id)"
      />
      <LogPanel
        :active-case="activeCase"
        :is-running="isRunning"
        @stream-end="onStreamEnd"
      />
    </main>
  </div>
</template>

<style scoped>
.layout {
  display: flex;
  flex-direction: column;
  height: 100vh;
}
.topbar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 10px 18px;
  border-bottom: 1px solid var(--border);
  background: var(--bg-elev);
}
.brand {
  display: flex;
  align-items: center;
  gap: 10px;
}
.logo {
  color: var(--accent);
  font-size: 18px;
}
.title {
  font-weight: 600;
  letter-spacing: 0.2px;
}
.status {
  display: flex;
  align-items: center;
  gap: 8px;
  color: var(--text-dim);
  font-size: 12px;
}
.dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  display: inline-block;
}
.dot.on { background: var(--ok); }
.dot.off { background: var(--fail); }
.main {
  display: flex;
  flex: 1;
  min-height: 0;
}
</style>
