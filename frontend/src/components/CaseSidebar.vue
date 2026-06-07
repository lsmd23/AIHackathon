<script setup lang="ts">
interface CaseInfo {
  id: string
  name: string
}

defineProps<{
  cases: CaseInfo[]
  activeCase: string | null
  isRunning: boolean
}>()

defineEmits<{
  (e: 'run-all'): void
  (e: 'run-case', id: string): void
}>()
</script>

<template>
  <aside class="sidebar">
    <div class="section">
      <button
        class="primary run-all"
        :disabled="isRunning"
        @click="$emit('run-all')"
      >
        ▶ Run All (5 cases)
      </button>
    </div>

    <div class="section">
      <div class="section-title">Cases</div>
      <ul class="case-list">
        <li
          v-for="(c, i) in cases"
          :key="c.id"
          :class="['case-item', { active: activeCase === c.id, running: activeCase === c.id }]"
        >
          <button
            class="case-btn"
            :disabled="isRunning"
            @click="$emit('run-case', c.id)"
          >
            <span class="case-num">{{ i + 1 }}</span>
            <span class="case-name">{{ c.name }}</span>
          </button>
        </li>
      </ul>
    </div>

    <div class="footer">
      <div>Online score: <strong>727.85</strong></div>
      <div>10/10 case · 100% coverage</div>
    </div>
  </aside>
</template>

<style scoped>
.sidebar {
  width: 280px;
  background: var(--bg-elev);
  border-right: 1px solid var(--border);
  display: flex;
  flex-direction: column;
  overflow-y: auto;
}
.section {
  padding: 12px;
  border-bottom: 1px solid var(--border);
}
.run-all {
  width: 100%;
  font-weight: 600;
}
.section-title {
  font-size: 11px;
  text-transform: uppercase;
  color: var(--text-dim);
  letter-spacing: 1px;
  margin-bottom: 8px;
}
.case-list {
  list-style: none;
  padding: 0;
  margin: 0;
  display: flex;
  flex-direction: column;
  gap: 4px;
}
.case-item {
  border-radius: 4px;
}
.case-item.active {
  background: rgba(91, 140, 255, 0.12);
  border-left: 2px solid var(--accent);
}
.case-btn {
  width: 100%;
  text-align: left;
  background: transparent;
  border: 1px solid transparent;
  padding: 8px 10px;
  display: flex;
  gap: 10px;
  align-items: flex-start;
  color: var(--text);
}
.case-btn:hover:not(:disabled) {
  background: #1c2540;
  border-color: var(--border);
}
.case-item.active .case-btn {
  background: transparent;
  border-color: transparent;
}
.case-num {
  color: var(--text-dim);
  font-size: 11px;
  width: 16px;
  text-align: right;
  flex-shrink: 0;
  padding-top: 1px;
}
.case-item.active .case-num {
  color: var(--accent);
  font-weight: 600;
}
.case-name {
  flex: 1;
  font-size: 12px;
  line-height: 1.4;
}
.footer {
  margin-top: auto;
  padding: 12px;
  border-top: 1px solid var(--border);
  font-size: 11px;
  color: var(--text-dim);
  display: flex;
  flex-direction: column;
  gap: 4px;
}
.footer strong {
  color: var(--ok);
  font-size: 14px;
}
</style>
