import crypto from 'crypto';

export interface CacheLike {
    get<T>(key: string): Promise<T | undefined | null>;
    set(key: string, value: any, ttl?: number): Promise<any>;
}

export type FetcherOptions = RequestInit & {
    /** Optional TTL override for this specific request, in milliseconds. */
    ttl?: number;
};

export type Fetcher = (
    url: string | URL | Request,
    options?: FetcherOptions
) => Promise<Response>;

export interface CreateFetcherDependencies {
    /** The cache instance (e.g., from cache-manager). */
    cache?: CacheLike;
    /** A prefix for all cache keys to avoid collisions. Defaults to 'http-cache'. */
    prefix?: string;
    /** Time-to-live for cache entries, in milliseconds. */
    ttl?: number;
    /** Request timeout in milliseconds. If not provided, no timeout is applied. */
    timeout?: number;
    /** User-Agent string for requests. */
    userAgent?: string;
    /** Optional custom fetch implementation. Defaults to global fetch. */
    fetch?: (url: string | URL | Request, init?: RequestInit) => Promise<Response>;
    /** 
     * Optional callback to determine if a response should be cached. 
     * It receives a cloned response that can be read (e.g. .json()).
     * If it returns false, the response is not cached.
     */
    shouldCache?: (response: Response) => Promise<boolean> | boolean;
}

interface CacheData {
    bodyBase64: string;
    headers: Record<string, string>;
    status: number;
    finalUrl: string;
}

export class CachedResponse extends Response {
    #finalUrl: string;

    constructor(body: BodyInit | null, init: ResponseInit, finalUrl: string) {
        super(body, init);
        this.#finalUrl = finalUrl;
    }

    get url() {
        return this.#finalUrl;
    }
}

/**
 * Creates a deterministic hash of headers for cache key generation.
 * Headers are sorted alphabetically to ensure consistency.
 */
function hashHeaders(headers?: HeadersInit): string {
    if (!headers) return '';
    
    let headerEntries: [string, string][];
    
    if (headers instanceof Headers) {
        headerEntries = Array.from(headers.entries());
    } else if (Array.isArray(headers)) {
        headerEntries = headers as [string, string][];
    } else {
        headerEntries = Object.entries(headers);
    }
    
    if (headerEntries.length === 0) return '';
    
    // Sort alphabetically by key for deterministic ordering
    headerEntries.sort((a, b) => a[0].localeCompare(b[0]));
    
    const headerString = headerEntries
        .map(([key, value]) => `${key}:${value}`)
        .join('|');
    
    return crypto.createHash('md5').update(headerString).digest('hex');
}

/**
 * Factory function that creates a `fetch` replacement with a caching layer.
 * @param deps - Dependencies including the cache instance, prefix, TTL, and timeout.
 * @returns A function with the same signature as native `fetch`.
 */
export function createCachedFetcher(deps: CreateFetcherDependencies): Fetcher {
    const { cache, prefix = 'http-cache', ttl, timeout, userAgent, fetch: customFetch, shouldCache } = deps;

    const fetchImpl = customFetch ?? fetch;

    const fetchWithTimeout = async (url: string | URL | Request, options?: RequestInit): Promise<Response> => {
        const headers = new Headers(options?.headers);
        if (userAgent) {
            headers.set('User-Agent', userAgent);
        }

        const finalOptions: RequestInit = {
            ...options,
            headers,
        };

        if (!timeout) {
            try {
                return await fetchImpl(url, finalOptions);
            } catch(error: any) {
                throw error;
            }
        }

        const controller = new AbortController();
        const timeoutId = setTimeout(() => {
            const urlString = typeof url === 'string' ? url : url.toString();
            console.log(`[Fetch Timeout] Request timed out after ${timeout}ms for: ${urlString}`);
            controller.abort();
        }, timeout);

        finalOptions.signal = controller.signal;

        try {
            const response = await fetchImpl(url, finalOptions);
            return response;
        } catch (error) {
            if (error instanceof Error && error.name === 'AbortError') {
                const urlString = typeof url === 'string' ? url : url.toString();
                throw new Error(`Request to ${urlString} timed out after ${timeout}ms`);
            }
            throw error;
        } finally {
            clearTimeout(timeoutId);
        }
    };

    return async (url: string | URL | Request, options?: FetcherOptions): Promise<Response> => {
        let method = 'GET';
        if (options?.method) {
            method = options.method;
        } else if (url instanceof Request) {
            method = url.method;
        }

        const urlString = typeof url === 'string' ? url : url.toString();

        if (!cache) {
            console.log(`[Cache SKIP] Cache not configured for request to: ${urlString}`);
            return fetchWithTimeout(url, options);
        }

        let cacheKey = `${prefix}:${urlString}`;

        // Hash body for POST requests
        if (method.toUpperCase() === 'POST' && options?.body) {
            let bodyStr = '';
            if (typeof options.body === 'string') {
                bodyStr = options.body;
            } else if (options.body instanceof URLSearchParams) {
                bodyStr = options.body.toString();
            } else {
                try {
                    bodyStr = JSON.stringify(options.body);
                } catch (e) {
                    bodyStr = 'unserializable';
                }
            }

            const bodyHash = crypto.createHash('md5').update(bodyStr).digest('hex');
            cacheKey += `:body:${bodyHash}`;
        }

        // Hash all request headers into cache key
        const headersHash = hashHeaders(options?.headers);
        if (headersHash) {
            cacheKey += `:headers:${headersHash}`;
        }

        // 1. Check the cache
        const cachedItem = await cache.get<CacheData>(cacheKey);
        if (cachedItem) {
            const body = Buffer.from(cachedItem.bodyBase64, 'base64');
            return new CachedResponse(
                body,
                {
                    status: cachedItem.status,
                    headers: cachedItem.headers,
                },
                cachedItem.finalUrl
            );
        }

        // 2. Perform the actual fetch if not in cache
        const fetchAndCache = async () => {
            const response = await fetchWithTimeout(url, options);

            // 3. Store in cache on success
            if (response.ok) {
                let isCacheable = true;

                if (shouldCache) {
                    const checkClone = response.clone();
                    try {
                        isCacheable = await shouldCache(checkClone);
                    } catch (e) {
                        console.warn('[Cache Check Error] shouldCache threw an error, skipping cache', e);
                        isCacheable = false;
                    }
                } else {
                    const contentType = response.headers.get('content-type');
                    if (contentType && contentType.includes('application/json')) {
                        const checkClone = response.clone();
                        try {
                            const body = await checkClone.json();
                            if (body && typeof body === 'object' && 'error' in body) {
                                console.log(`[Cache SKIP] JSON response contains .error property for: ${urlString}`);
                                isCacheable = false;
                            }
                        } catch (e) {
                            // Ignore JSON parse errors, assume cacheable if status is OK
                        }
                    }
                }

                if (isCacheable) {
                    const responseClone = response.clone();
                    const bodyBuffer = await responseClone.arrayBuffer();
                    const bodyBase64 = Buffer.from(bodyBuffer).toString('base64');
                    const headers = Object.fromEntries(response.headers.entries());

                    const itemToCache: CacheData = {
                        bodyBase64,
                        headers,
                        status: response.status,
                        finalUrl: response.url,
                    };

                    await cache.set(cacheKey, itemToCache, options?.ttl ?? ttl);
                    console.log(`[Cache SET] for: ${cacheKey}`);
                }
            }

            // 4. Return the original response
            return response;
        };

        return fetchAndCache();
    };
}
