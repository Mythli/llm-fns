import { describe, it, expect } from 'vitest';
import { z } from 'zod';
import { normalizeZodArgs } from './createZodLlmClient.js';

describe('normalizeZodArgs', () => {
    const TestSchema = z.object({
        foo: z.string()
    });

    it('should normalize Schema Only (Case 1)', () => {
        const result = normalizeZodArgs(TestSchema);
        
        expect(result.mainInstruction).toContain('Generate a valid JSON');
        expect(result.userMessagePayload).toBe('Generate the data.');
        expect(result.dataExtractionSchema).toBe(TestSchema);
        expect(result.options).toBeUndefined();
    });

    it('should normalize Schema Only with Options (Case 1)', () => {
        const options = { temperature: 0.5 };
        const result = normalizeZodArgs(TestSchema, options);
        
        expect(result.mainInstruction).toContain('Generate a valid JSON');
        expect(result.userMessagePayload).toBe('Generate the data.');
        expect(result.dataExtractionSchema).toBe(TestSchema);
        expect(result.options).toBe(options);
    });

    it('should normalize Prompt + Schema (Case 2)', () => {
        const prompt = "Extract data";
        const result = normalizeZodArgs(prompt, TestSchema);
        
        expect(result.mainInstruction).toContain('You are a helpful assistant');
        expect(result.userMessagePayload).toBe(prompt);
        expect(result.dataExtractionSchema).toBe(TestSchema);
        expect(result.options).toBeUndefined();
    });

    it('should normalize Prompt + Schema with Options (Case 2)', () => {
        const prompt = "Extract data";
        const options = { temperature: 0.5 };
        const result = normalizeZodArgs(prompt, TestSchema, options);
        
        expect(result.mainInstruction).toContain('You are a helpful assistant');
        expect(result.userMessagePayload).toBe(prompt);
        expect(result.dataExtractionSchema).toBe(TestSchema);
        expect(result.options).toBe(options);
    });

    it('should normalize System + User + Schema (Case 3)', () => {
        const system = "System prompt";
        const user = "User prompt";
        const result = normalizeZodArgs(system, user, TestSchema);
        
        expect(result.mainInstruction).toBe(system);
        expect(result.userMessagePayload).toBe(user);
        expect(result.dataExtractionSchema).toBe(TestSchema);
        expect(result.options).toBeUndefined();
    });

    it('should normalize System + User + Schema with Options (Case 3)', () => {
        const system = "System prompt";
        const user = "User prompt";
        const options = { temperature: 0.5 };
        const result = normalizeZodArgs(system, user, TestSchema, options);
        
        expect(result.mainInstruction).toBe(system);
        expect(result.userMessagePayload).toBe(user);
        expect(result.dataExtractionSchema).toBe(TestSchema);
        expect(result.options).toBe(options);
    });

    it('should throw error for invalid arguments', () => {
        expect(() => normalizeZodArgs({} as any)).toThrow('Invalid arguments');
    });
});
