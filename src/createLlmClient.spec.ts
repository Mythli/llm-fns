import { describe, it, expect } from 'vitest';
import { normalizeOptions } from './createLlmClient.js';

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
});
