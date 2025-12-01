import { describe, it, expect, vi } from 'vitest';
import { z } from 'zod';
import { createTestLlm } from './setup.js';
import { createZodLlmClient } from '../src/createZodLlmClient.js';
import { createJsonSchemaLlmClient } from '../src/createJsonSchemaLlmClient.js';
import { LlmRetryExhaustedError } from '../src/createLlmRetryClient.js';

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

describe('Zod Structured Output Integration', () => {
    it('should extract structured data matching a schema', async () => {
        const { llm } = await createTestLlm();

        const InventorySchema = z.object({
            items: z.array(z.object({
                name: z.string(),
                quantity: z.number(),
                color: z.string().optional(),
            }))
        });

        const text = "I have 3 red apples and 2 yellow bananas in my basket.";

        const result = await llm.promptZod(
            "Extract the inventory items from the user text.",
            [{ type: 'text', text }],
            InventorySchema
        );

        expect(result).toBeDefined();
        expect(result.items).toHaveLength(2);
        
        const apple = result.items.find(i => i.name.toLowerCase().includes('apple'));
        expect(apple).toBeDefined();
        expect(apple?.quantity).toBe(3);
        expect(apple?.color).toBe('red');

        const banana = result.items.find(i => i.name.toLowerCase().includes('banana'));
        expect(banana).toBeDefined();
        expect(banana?.quantity).toBe(2);
        expect(banana?.color).toBe('yellow');
    });

    describe('Retry Mechanisms (Mocked)', () => {
        const Schema = z.object({
            age: z.number()
        });

        it('should fix invalid JSON syntax using the internal fixer', async () => {
            const mockPrompt = createMockPrompt([
                '{"age": 20', // Missing closing brace
                '{"age": 20}' // Fixed
            ]);

            const jsonSchemaClient = createJsonSchemaLlmClient({
                prompt: mockPrompt,
                isPromptCached: async () => false,
            });

            const client = createZodLlmClient({
                jsonSchemaClient
            });

            const result = await client.promptZod("test", "test", Schema);
            
            expect(result.age).toBe(20);
            expect(mockPrompt).toHaveBeenCalledTimes(2);
            
            // The second call should be the fixer prompt
            // args[0] is options object
            const secondCallArgs = mockPrompt.mock.calls[1][0] as any;
            expect(secondCallArgs.messages[1].content).toContain('BROKEN RESPONSE');
        });

        it('should fix schema validation errors using the internal fixer', async () => {
            const mockPrompt = createMockPrompt([
                '{"age": "twenty"}', // String instead of number
                '{"age": 20}'        // Fixed
            ]);

            const jsonSchemaClient = createJsonSchemaLlmClient({
                prompt: mockPrompt,
                isPromptCached: async () => false,
            });

            const client = createZodLlmClient({
                jsonSchemaClient
            });

            const result = await client.promptZod("test", "test", Schema);
            
            expect(result.age).toBe(20);
            expect(mockPrompt).toHaveBeenCalledTimes(2);
            
            const secondCallArgs = mockPrompt.mock.calls[1][0] as any;
            expect(secondCallArgs.messages[1].content).toContain('Schema Validation Error');
        });

        it('should use the main retry loop when internal fixer is disabled', async () => {
            const mockPrompt = createMockPrompt([
                '{"age": "wrong"}', // Initial wrong response
                '{"age": 20}'       // Corrected in next turn
            ]);

            const jsonSchemaClient = createJsonSchemaLlmClient({
                prompt: mockPrompt,
                isPromptCached: async () => false,
                disableJsonFixer: true // Force main loop
            });

            const client = createZodLlmClient({
                jsonSchemaClient
            });

            const result = await client.promptZod("test", "test", Schema);
            
            expect(result.age).toBe(20);
            expect(mockPrompt).toHaveBeenCalledTimes(2);
            
            // In the main loop, the history is preserved and error is added as user message
            const secondCallArgs = mockPrompt.mock.calls[1][0] as any;
            const messages = secondCallArgs.messages;
            // 0: System, 1: User, 2: Assistant (wrong), 3: User (Error feedback)
            expect(messages.length).toBe(4);
            expect(messages[3].role).toBe('user');
            expect(messages[3].content).toContain('SCHEMA_VALIDATION_ERROR');
        });

        it('should switch to fallback prompt on retry if provided', async () => {
            const mockMainPrompt = createMockPrompt([
                '{"age": "wrong"}' // Fails schema
            ]);
            
            const mockFallbackPrompt = createMockPrompt([
                '{"age": 20}' // Succeeds
            ]);

            const jsonSchemaClient = createJsonSchemaLlmClient({
                prompt: mockMainPrompt,
                fallbackPrompt: mockFallbackPrompt,
                isPromptCached: async () => false,
                disableJsonFixer: true // Force error to retry loop immediately
            });

            const client = createZodLlmClient({
                jsonSchemaClient
            });

            const result = await client.promptZod("test", "test", Schema);
            
            expect(result.age).toBe(20);
            expect(mockMainPrompt).toHaveBeenCalledTimes(1);
            expect(mockFallbackPrompt).toHaveBeenCalledTimes(1);
        });

        it('should throw LlmRetryExhaustedError when retries are exhausted', async () => {
            const mockPrompt = createMockPrompt([
                '{"age": "wrong"}',
                '{"age": "still wrong"}',
                '{"age": "forever wrong"}'
            ]);

            const jsonSchemaClient = createJsonSchemaLlmClient({
                prompt: mockPrompt,
                isPromptCached: async () => false,
                disableJsonFixer: true
            });

            const client = createZodLlmClient({
                jsonSchemaClient
            });

            await expect(client.promptZod("test", "test", Schema, { maxRetries: 1 }))
                .rejects.toThrow(LlmRetryExhaustedError);
            
            expect(mockPrompt).toHaveBeenCalledTimes(2); // Initial + 1 retry
        });
    });
});
