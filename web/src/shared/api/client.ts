/** Shared HTTP client with normalized error handling. */

export interface ApiError extends Error {
  status: number;
  detail?: string;
  data?: unknown;
}

export interface RequestOptions extends RequestInit {
  params?: Record<string, string | number | boolean | undefined>;
}

function buildUrl(path: string, params?: Record<string, string | number | boolean | undefined>): string {
  const url = new URL(path, window.location.origin);
  if (params) {
    Object.entries(params).forEach(([key, value]) => {
      if (value !== undefined && value !== null) {
        url.searchParams.append(key, String(value));
      }
    });
  }
  return url.toString();
}

async function handleResponse<T>(response: Response): Promise<T> {
  if (!response.ok) {
    let detail: string | undefined;
    let data: unknown;
    try {
      const body = await response.json();
      detail = body.detail || body.error?.message || body.error?.detail?.message;
      data = body;
    } catch {
      detail = await response.text().catch(() => 'Unknown error');
    }
    const error: ApiError = new Error(detail || `Request failed: ${response.status}`) as ApiError;
    error.status = response.status;
    error.detail = detail;
    error.data = data;
    throw error;
  }
  return response.json() as Promise<T>;
}

export async function apiClient<T>(path: string, options: RequestOptions = {}): Promise<T> {
  const { params, headers, ...init } = options;
  const url = buildUrl(path, params);

  const response = await fetch(url, {
    ...init,
    headers: {
      'Content-Type': 'application/json',
      ...headers,
    },
  });

  return handleResponse<T>(response);
}

export const api = {
  get<T>(path: string, params?: Record<string, string | number | boolean | undefined>): Promise<T> {
    return apiClient<T>(path, { method: 'GET', params });
  },

  post<T>(path: string, body?: unknown, params?: Record<string, string | number | boolean | undefined>): Promise<T> {
    return apiClient<T>(path, {
      method: 'POST',
      body: body === undefined ? undefined : JSON.stringify(body),
      params,
    });
  },

  put<T>(path: string, body: unknown, params?: Record<string, string | number | boolean | undefined>): Promise<T> {
    return apiClient<T>(path, {
      method: 'PUT',
      body: JSON.stringify(body),
      params,
    });
  },

  patch<T>(path: string, body: unknown, params?: Record<string, string | number | boolean | undefined>): Promise<T> {
    return apiClient<T>(path, {
      method: 'PATCH',
      body: JSON.stringify(body),
      params,
    });
  },

  delete<T>(path: string, params?: Record<string, string | number | boolean | undefined>): Promise<T> {
    return apiClient<T>(path, { method: 'DELETE', params });
  },
};
