import { useCallback, useEffect, useRef, useState } from 'react';

/**
 * Load data from the API when a component mounts (and on reload()).
 * Returns { data, error, loading, reload }. Ignores responses that arrive
 * after the component unmounted or after a newer request started.
 */
export function useApiResource(fetcher, deps = [], { enabled = true } = {}) {
  const [state, setState] = useState({ data: null, error: null, loading: enabled });
  const requestId = useRef(0);

  const load = useCallback(async () => {
    if (!enabled) return;
    const id = ++requestId.current;
    setState((s) => ({ ...s, loading: true, error: null }));
    try {
      const data = await fetcher();
      if (id === requestId.current) setState({ data, error: null, loading: false });
    } catch (error) {
      if (id === requestId.current) setState({ data: null, error, loading: false });
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [enabled, ...deps]);

  useEffect(() => {
    load();
    return () => { requestId.current += 1; };
  }, [load]);

  return { ...state, reload: load };
}
