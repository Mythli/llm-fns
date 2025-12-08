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
});
