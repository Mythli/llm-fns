import { describe, it, expect, beforeEach, vi } from 'vitest';
import type { MockedFunction } from 'vitest';
import { createZodLlmClient } from './createZodLlmClient.js';
import { z } from 'zod';
import { PromptFunction } from './createLlmClient.js';

// Define a simple schema for testing
const PersonSchema = z.object({
    firstName: z.string().describe("The person's first name"),
    lastName: z.string().describe("The person's last name"),
});

function createMockCompletion(content: string): any {
    return {
        choices: [
            {
                message: {
                    role: 'assistant',
                    content: content
                }
            }
        ]
    };
}

describe('ZodLlmClient', () => {
    let client: ReturnType<typeof createZodLlmClient>;
    let mockPrompt: MockedFunction<PromptFunction>;

    const mainInstruction = 'Extract person details.';
    const userMessagePayload = [{ type: 'text' as const, text: 'The person is John Doe.' }];

    beforeEach(() => {
        // Create mock functions for each test
        mockPrompt = vi.fn();
    });

    it('should return valid data on the first attempt (happy path)', async () => {
        // Arrange
        const validResponse = { firstName: 'John', lastName: 'Doe' };
        // The mock needs to return a promise that resolves to the response object
        mockPrompt.mockResolvedValue(createMockCompletion(JSON.stringify(validResponse)));
        client = createZodLlmClient({ prompt: mockPrompt });

        // Act
        const result = await client.promptZod(mainInstruction, userMessagePayload, PersonSchema);

        // Assert
        expect(result).toEqual(validResponse);
        expect(mockPrompt).toHaveBeenCalledTimes(1);
    });

    it('should use regular retry after a failed fixer attempt for schema error', async () => {
        // Arrange
        const invalidResponse = JSON.stringify({ firstName: 'John' }); // Missing lastName
        const validResponse = { firstName: 'John', lastName: 'Doe' };

        mockPrompt
            .mockResolvedValueOnce(createMockCompletion(invalidResponse)) // 1. Initial call, schema error
            .mockResolvedValueOnce(createMockCompletion("CANNOT_FIX"))    // 2. Fixer call, fails
            .mockResolvedValueOnce(createMockCompletion(JSON.stringify(validResponse)));  // 3. Retry call, succeeds

        client = createZodLlmClient({ prompt: mockPrompt });

        // Act
        const result = await client.promptZod(mainInstruction, userMessagePayload, PersonSchema, { maxRetries: 1 });

        // Assert
        expect(result).toEqual(validResponse);
        expect(mockPrompt).toHaveBeenCalledTimes(3);

        // Check the third call (the retry call)
        const retryCallArgs = mockPrompt.mock.calls[2][0];
        const messages = retryCallArgs.messages;
        expect(messages).toHaveLength(4); // system, user, assistant, user (retry)
        expect(messages[3].role).toBe('user');
        expect(messages[3].content).toContain('Your previous response resulted in an error');
        expect(messages[3].content).toContain('SCHEMA_VALIDATION_ERROR');
    });

    it('should use one-time fixer for JSON parse error and succeed', async () => {
        // Arrange
        const invalidJsonResponse = `{ "firstName": "John", "lastName": "Doe"`; // Malformed JSON
        const fixedResponse = { firstName: 'John', lastName: 'Doe' };
        const fixedResponseString = JSON.stringify(fixedResponse);

        mockPrompt
            .mockResolvedValueOnce(createMockCompletion(invalidJsonResponse)) // Initial call returns broken JSON
            .mockResolvedValueOnce(createMockCompletion(fixedResponseString)); // Fixer call returns valid JSON

        client = createZodLlmClient({ prompt: mockPrompt });

        // Act
        const result = await client.promptZod(mainInstruction, userMessagePayload, PersonSchema);

        // Assert
        expect(result).toEqual(fixedResponse);
        expect(mockPrompt).toHaveBeenCalledTimes(2);

        // Check the second call (the fixer call)
        const fixerCallArgs = mockPrompt.mock.calls[1][0];
        const fixerMessages = fixerCallArgs.messages;
        expect(fixerMessages[0].role).toBe('system');
        expect(fixerMessages[0].content).toContain('You are an expert at fixing malformed JSON');
        expect(fixerMessages[1].role).toBe('user');
        expect(fixerMessages[1].content).toContain('BROKEN RESPONSE');
        expect(fixerMessages[1].content).toContain(invalidJsonResponse);
    });

    it('should proceed to regular retry if one-time fixer fails', async () => {
        // Arrange
        const invalidJsonResponse = `{ "firstName": "John", "lastName": "Doe"`; // Malformed JSON
        const validResponse = { firstName: 'John', lastName: 'Doe' };

        mockPrompt
            .mockResolvedValueOnce(createMockCompletion(invalidJsonResponse)) // 1. Initial call, broken JSON
            .mockResolvedValueOnce(createMockCompletion("CANNOT_FIX"))       // 2. Fixer call, fails to fix
            .mockResolvedValueOnce(createMockCompletion(JSON.stringify(validResponse))); // 3. Regular retry call, succeeds

        client = createZodLlmClient({ prompt: mockPrompt });

        // Act
        const result = await client.promptZod(mainInstruction, userMessagePayload, PersonSchema, { maxRetries: 1 });

        // Assert
        expect(result).toEqual(validResponse);
        expect(mockPrompt).toHaveBeenCalledTimes(3);

        // Check the third call (the retry call)
        const retryCallArgs = mockPrompt.mock.calls[2][0];
        const retryMessages = retryCallArgs.messages;
        expect(retryMessages).toHaveLength(4); // system, user, assistant, user (retry)
        expect(retryMessages[3].content).toContain('Your previous response resulted in an error');
        expect(retryMessages[3].content).toContain('JSON_PARSE_ERROR');
    });
});
