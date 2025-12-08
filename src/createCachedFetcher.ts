// src/createCachedFetcher.ts

import crypto from 'crypto';

// Define a minimal interface for the cache to avoid tight coupling with cache-manager versions
// and to support the new v7 API which might not export 'Cache' in the same way.
export interface CacheLike {
    get<T>(key: string): Promise<T | undefined | null>;
    set(key: string, value: any, ttl?: number): Promise<any>;
}

// Define a custom options type that extends RequestInit with our custom `ttl` property.
export type FetcherOptions = RequestInit & {
    /** Optional TTL override for this specific request, in milliseconds. */
    ttl?: number;
};

// Define the shape of the function we are creating and exporting.
// It must match the native fetch signature, but with our custom options.
export type Fetcher = (
    url: string | URL | Request,
    options?: FetcherOptions
) => Promise<Response>;

// Define the dependencies needed to create our cached fetcher.
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
}

// The data we store in the cache. Kept internal to this module.
// The body is stored as a base64 string to ensure proper serialization in Redis.
interface CacheData {
    bodyBase64: string;
    headers: Record<string, string>;
    status: number;
    finalUrl: string; // Crucial for resolving relative URLs on cache HITs
}

// A custom Response class to correctly handle the `.url` property on cache HITs.
// This is an implementation detail and doesn't need to be exported.
export class CachedResponse extends Response {
    #finalUrl: string;

    constructor(body: BodyInit | null, init: ResponseInit, finalUrl: string) {
        super(body, init);
        this.#finalUrl = finalUrl;
    }

    // Override the read-only `url` property
    get url() {
        return this.#finalUrl;
    }
}

/**
 * Factory function that creates a `fetch` replacement with a caching layer.
 * @param deps - Dependencies including the cache instance, prefix, TTL, and timeout.
 * @returns A function with the same signature as native `fetch`.
 */
export function createCachedFetcher(deps: CreateFetcherDependencies): Fetcher {
    const { cache, prefix = 'http-cache', ttl, timeout, userAgent, fetch: customFetch } = deps;

    const fetchImpl = customFetch ?? fetch;

    const fetchWithTimeout = async (url: string | URL | Request, options?: RequestInit): Promise<Response> => {
        // Correctly merge headers using Headers API to handle various input formats (plain object, Headers instance, array)
        // and avoid issues with spreading Headers objects which can lead to lost headers or Symbol errors.
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

    // This is the actual fetcher implementation, returned by the factory.
    // It "closes over" the dependencies provided to the factory.
    return async (url: string | URL | Request, options?: FetcherOptions): Promise<Response> => {
        // Determine the request method. Default to GET for fetch.
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

        // If POST (or others with body), append hash of body to cache key
        if (method.toUpperCase() === 'POST' && options?.body) {
            let bodyStr = '';
            if (typeof options.body === 'string') {
                bodyStr = options.body;
            } else if (options.body instanceof URLSearchParams) {
                bodyStr = options.body.toString();
            } else {
                // Fallback for other types, though mostly we expect string/JSON here
                try {
                    bodyStr = JSON.stringify(options.body);
                } catch (e) {
                    bodyStr = 'unserializable';
                }
            }

            const hash = crypto.createHash('md5').update(bodyStr).digest('hex');
            cacheKey += `:${hash}`;
        }

        // 1. Check the cache
        const cachedItem = await cache.get<CacheData>(cacheKey);
        if (cachedItem) {
            // Decode the base64 body back into a Buffer.
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
                const responseClone = response.clone();
                const bodyBuffer = await responseClone.arrayBuffer();
                // Convert ArrayBuffer to a base64 string for safe JSON serialization.
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

            // 4. Return the original response
            return response;
        };

        return fetchAndCache();
    };
}
