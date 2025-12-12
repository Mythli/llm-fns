import { describe, it, expect } from 'vitest';
import { normalizeOptions, mergeRequestOptions } from './createLlmClient.js';

describe('normalizeOptions', () => {
    it('should normalize a simple string prompt', () => {
        const result = normalizeOptions('Hello world');
        expect(result).toEqual({
            messages: [{ role: 'user', content: 'Hello world' }]
        });
    });

    it('should normalize a string prompt with options', () => {
        const result = normalizeOptions('Hello world', { temperature: 0.5 });
        expect(result).toEqual({
            messages: [{ role: 'user', content: 'Hello world' }],
            temperature: 0.5
        });
    });

    it('should normalize an options object with string messages', () => {
        const result = normalizeOptions({
            messages: 'Hello world',
            temperature: 0.7
        });
        expect(result).toEqual({
            messages: [{ role: 'user', content: 'Hello world' }],
            temperature: 0.7
        });
    });

    it('should pass through an options object with array messages', () => {
        const messages = [{ role: 'user', content: 'Hello' }] as any;
        const result = normalizeOptions({
            messages,
            temperature: 0.7
        });
        expect(result).toEqual({
            messages,
            temperature: 0.7
        });
    });

    it('should include requestOptions when provided', () => {
        const result = normalizeOptions('Hello world', { 
            temperature: 0.5,
            requestOptions: {
                headers: { 'X-Custom': 'value' },
                timeout: 5000
            }
        });
        expect(result).toEqual({
            messages: [{ role: 'user', content: 'Hello world' }],
            temperature: 0.5,
            requestOptions: {
                headers: { 'X-Custom': 'value' },
                timeout: 5000
            }
        });
    });
});

describe('mergeRequestOptions', () => {
    it('should return undefined when both are undefined', () => {
        const result = mergeRequestOptions(undefined, undefined);
        expect(result).toBeUndefined();
    });

    it('should return override when base is undefined', () => {
        const override = { timeout: 5000 };
        const result = mergeRequestOptions(undefined, override);
        expect(result).toBe(override);
    });

    it('should return base when override is undefined', () => {
        const base = { timeout: 5000 };
        const result = mergeRequestOptions(base, undefined);
        expect(result).toBe(base);
    });

    it('should merge headers from both', () => {
        const base = { 
            headers: { 'X-Base': 'base-value' },
            timeout: 5000
        };
        const override = { 
            headers: { 'X-Override': 'override-value' }
        };
        const result = mergeRequestOptions(base, override);
        expect(result).toEqual({
            headers: {
                'X-Base': 'base-value',
                'X-Override': 'override-value'
            },
            timeout: 5000
        });
    });

    it('should override scalar properties', () => {
        const base = { timeout: 5000 };
        const override = { timeout: 10000 };
        const result = mergeRequestOptions(base, override);
        expect(result).toEqual({ timeout: 10000, headers: {} });
    });

    it('should override conflicting headers', () => {
        const base = { 
            headers: { 'X-Shared': 'base-value', 'X-Base': 'base' }
        };
        const override = { 
            headers: { 'X-Shared': 'override-value', 'X-Override': 'override' }
        };
        const result = mergeRequestOptions(base, override);
        expect(result).toEqual({
            headers: {
                'X-Shared': 'override-value',
                'X-Base': 'base',
                'X-Override': 'override'
            }
        });
    });
});
