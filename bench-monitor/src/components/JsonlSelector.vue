<script setup lang="ts">
import { ref, computed } from 'vue';
import type { LeaderboardEntry } from '../types/message';

const props = defineProps<{
  entries: LeaderboardEntry[];
  isLoading: boolean;
  selectedPath?: string;
}>();

const emit = defineEmits<{
  select: [path: string];
  refresh: [];
}>();

const isExpanded = ref(true);

const groupedEntries = computed(() => {
  const groups: Record<string, LeaderboardEntry[]> = {};
  for (const entry of props.entries) {
    const modelName = entry.name.replace(/ \(Turn \d+\)$/, '');
    if (!groups[modelName]) {
      groups[modelName] = [];
    }
    groups[modelName].push(entry);
  }
  return groups;
});

function selectFile(path: string) {
  emit('select', path);
}
</script>

<template>
  <div class="selector">
    <div class="selector-header" @click="isExpanded = !isExpanded">
      <span class="expand-icon">{{ isExpanded ? '▼' : '▶' }}</span>
      <span class="title">Leaderboard</span>
      <button class="refresh-btn" @click.stop="emit('refresh')" :disabled="isLoading">
        {{ isLoading ? '...' : '↻' }}
      </button>
    </div>
    
    <div v-if="isExpanded" class="selector-content">
      <div v-if="entries.length === 0 && !isLoading" class="empty">
        No files found
      </div>
      
      <div v-for="(group, modelName) in groupedEntries" :key="modelName" class="model-group">
        <div class="model-name">{{ modelName }}</div>
        <div 
          v-for="entry in group" 
          :key="entry.path"
          class="file-item"
          :class="{ selected: selectedPath === entry.path }"
          @click="selectFile(entry.path)"
        >
          Turn {{ entry.turn }}
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.selector {
  background: #1e1e1e;
  border-right: 1px solid #333;
  height: 100%;
  overflow-y: auto;
}

.selector-header {
  display: flex;
  align-items: center;
  padding: 12px 16px;
  cursor: pointer;
  border-bottom: 1px solid #333;
  background: #252526;
}

.expand-icon {
  font-size: 10px;
  margin-right: 8px;
  color: #888;
}

.title {
  flex: 1;
  font-weight: 600;
  font-size: 13px;
  color: #ccc;
}

.refresh-btn {
  background: none;
  border: none;
  color: #888;
  cursor: pointer;
  padding: 4px 8px;
  font-size: 14px;
  border-radius: 4px;
}

.refresh-btn:hover:not(:disabled) {
  background: #333;
  color: #fff;
}

.refresh-btn:disabled {
  opacity: 0.5;
}

.selector-content {
  padding: 8px 0;
}

.empty {
  padding: 16px;
  color: #666;
  font-size: 12px;
  text-align: center;
}

.model-group {
  margin-bottom: 8px;
}

.model-name {
  padding: 6px 16px;
  font-size: 11px;
  font-weight: 600;
  color: #666;
  text-transform: uppercase;
  letter-spacing: 0.5px;
}

.file-item {
  padding: 8px 16px 8px 24px;
  cursor: pointer;
  font-size: 13px;
  color: #aaa;
  transition: all 0.15s;
}

.file-item:hover {
  background: #2a2d2e;
  color: #fff;
}

.file-item.selected {
  background: #094771;
  color: #fff;
}
</style>
