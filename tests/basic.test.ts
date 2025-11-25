import { describe, it, expect } from 'vitest';
import { createTestLlm } from './setup.js';
import crypto from 'crypto';

describe('Basic LLM Integration', () => {
    it('should generate text from a simple prompt (Power User Interface)', async () => {
        const { llm } = await createTestLlm();
        
        const response = await llm.promptText({
            messages: [{ role: 'user', content: 'Say "Hello Integration Test"' }],
            temperature: 0,
        });

        expect(response).toBeTruthy();
        expect(response).toContain('Hello Integration Test');
    });

    it('should cache responses', async () => {
        const { llm } = await createTestLlm();
        const uniqueId = crypto.randomUUID();
        const promptOptions = {
            messages: [{ role: 'user', content: `Repeat this ID: ${uniqueId}` }],
            temperature: 0,
            ttl: 5000, // 5 seconds
        };

        // First call - should hit API
        const start1 = Date.now();
        const response1 = await llm.promptText(promptOptions);
        
        expect(response1).toContain(uniqueId);

        // Check if cached
        const isCached = await llm.isPromptCached(promptOptions);
        expect(isCached).toBe(true);

        // Second call - should be fast (cached)
        const start2 = Date.now();
        const response2 = await llm.promptText(promptOptions);
        const duration2 = Date.now() - start2;

        expect(response2).toBe(response1);
        // API calls usually take > 200ms. Cache hits are usually < 10ms.
        // We'll be generous and say second call should be significantly faster or under 100ms.
        expect(duration2).toBeLessThan(100); 
    });
});
