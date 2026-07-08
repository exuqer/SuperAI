import { defineStore } from 'pinia';
import { ref } from 'vue';
import { api } from '@/shared/api/client';
import type { AnalyzeResponse, GraphPayload, Job, SemanticResult } from '@/shared/api/types';

export const useRuntimeStore = defineStore('runtime', () => {
  const config = ref<Record<string, unknown> | null>(null);
  const lastAnalysis = ref<AnalyzeResponse | null>(null);
  const lastResult = ref<SemanticResult | null>(null);
  const graph = ref<GraphPayload | null>(null);
  const jobs = ref<Job[]>([]);
  const loading = ref(false);
  const error = ref<string | null>(null);

  async function loadConfig() {
    config.value = await api.getConfig();
  }

  async function analyze(payload: Record<string, unknown>) {
    loading.value = true;
    error.value = null;
    try {
      const response = await api.analyze(payload);
      lastAnalysis.value = response;
      lastResult.value = response.result;
      graph.value = response.graph;
      return response;
    } catch (err) {
      error.value = err instanceof Error ? err.message : String(err);
      throw err;
    } finally {
      loading.value = false;
    }
  }

  async function chat(payload: Record<string, unknown>) {
    loading.value = true;
    error.value = null;
    try {
      const response = await api.chatMessage(payload);
      lastAnalysis.value = response;
      lastResult.value = response.result;
      graph.value = response.graph;
      return response;
    } catch (err) {
      error.value = err instanceof Error ? err.message : String(err);
      throw err;
    } finally {
      loading.value = false;
    }
  }

  async function resonanceChat(payload: Record<string, unknown>) {
    loading.value = true;
    error.value = null;
    try {
      const response = await api.resonanceGenerate(payload);
      lastAnalysis.value = response;
      lastResult.value = response.result;
      graph.value = response.graph;
      return response;
    } catch (err) {
      error.value = err instanceof Error ? err.message : String(err);
      throw err;
    } finally {
      loading.value = false;
    }
  }

  async function loadGraph(params: Record<string, unknown> = {}) {
    graph.value = await api.getGraph(params);
    return graph.value;
  }

  async function refreshJobs() {
    jobs.value = await api.getJobs();
  }

  async function trackJob(job: Job) {
    jobs.value = [job, ...jobs.value.filter((item) => item.job_id !== job.job_id)];
    const timer = window.setInterval(async () => {
      const updated = await api.getJob(job.job_id);
      jobs.value = [updated, ...jobs.value.filter((item) => item.job_id !== job.job_id)];
      if (updated.status === 'completed' || updated.status === 'failed') {
        window.clearInterval(timer);
      }
    }, 800);
  }

  return {
    config,
    lastAnalysis,
    lastResult,
    graph,
    jobs,
    loading,
    error,
    loadConfig,
    analyze,
    chat,
    resonanceChat,
    loadGraph,
    refreshJobs,
    trackJob,
  };
});
