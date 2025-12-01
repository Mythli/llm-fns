import { describe, it, expect, vi } from 'vitest';
import { createTestLlm } from './setup.js';
import { createJsonSchemaLlmClient } from '../src/createJsonSchemaLlmClient.js';

// Helper to create a mock prompt function
function createMockPrompt(responses: string[]) {
    let callCount = 0;
    return vi.fn(async (...args: any[]) => {
        const content = responses[callCount] || responses[responses.length - 1];
        callCount++;
        
        return {
            id: 'mock-id',
            object: 'chat.completion',
            created: Date.now(),
            model: 'mock-model',
            choices: [{
                message: { role: 'assistant', content },
                finish_reason: 'stop',
                index: 0
            }],
            usage: { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 }
        } as any;
    });
}

describe('JSON Schema Structured Output Integration', () => {
    it('should extract structured data matching a raw JSON schema', async () => {
        const { llm } = await createTestLlm();

        const schema = {
            type: "object",
            properties: {
                sentiment: { type: "string", enum: ["positive", "negative"] },
                score: { type: "number" }
            },
            required: ["sentiment", "score"],
            additionalProperties: false
        };

        const text = "I absolutely love this product! It's amazing.";

        const result = await llm.promptJson(
            [{ role: 'user', content: `Analyze sentiment: ${text}` }],
            schema
        );

        expect(result).toBeDefined();
        expect(result.sentiment).toBe('positive');
        expect(typeof result.score).toBe('number');
    });

    describe('Retry Mechanisms (Mocked)', () => {
        const schema = {
            type: "object",
            properties: {
                age: { type: "number" }
            },
            required: ["age"]
        };

        it('should fix invalid JSON syntax using the internal fixer', async () => {
            const mockPrompt = createMockPrompt([
                '{"age": 20', // Missing closing brace
                '{"age": 20}' // Fixed
            ]);

            const client = createJsonSchemaLlmClient({
                prompt: mockPrompt,
                isPromptCached: async () => false,
            });

            const result = await client.promptJson(
                [{ role: 'user', content: "test" }],
                schema
            );
            
            expect(result.age).toBe(20);
            expect(mockPrompt).toHaveBeenCalledTimes(2);
            
            // The second call should be the fixer prompt
            const secondCallArgs = mockPrompt.mock.calls[1][0] as any;
            expect(secondCallArgs.messages[1].content).toContain('BROKEN RESPONSE');
        });

        it('should fix validation errors using the internal fixer', async () => {
            const mockPrompt = createMockPrompt([
                '{"age": "twenty"}', // String instead of number
                '{"age": 20}'        // Fixed
            ]);

            const client = createJsonSchemaLlmClient({
                prompt: mockPrompt,
                isPromptCached: async () => false,
            });

            const result = await client.promptJson(
                [{ role: 'user', content: "test" }],
                schema
            );
            
            expect(result.age).toBe(20);
            expect(mockPrompt).toHaveBeenCalledTimes(2);
            
            const secondCallArgs = mockPrompt.mock.calls[1][0] as any;
            expect(secondCallArgs.messages[1].content).toContain('Schema Validation Error');
        });
    });
});
