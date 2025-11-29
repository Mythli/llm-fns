"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
const vitest_1 = require("vitest");
const zod_1 = require("zod");
const createZodLlmClient_js_1 = require("./createZodLlmClient.js");
(0, vitest_1.describe)('normalizeZodArgs', () => {
    const TestSchema = zod_1.z.object({
        foo: zod_1.z.string()
    });
    (0, vitest_1.it)('should normalize Schema Only (Case 1)', () => {
        const result = (0, createZodLlmClient_js_1.normalizeZodArgs)(TestSchema);
        (0, vitest_1.expect)(result.mainInstruction).toContain('Generate a valid JSON');
        (0, vitest_1.expect)(result.userMessagePayload).toBe('Generate the data.');
        (0, vitest_1.expect)(result.dataExtractionSchema).toBe(TestSchema);
        (0, vitest_1.expect)(result.options).toBeUndefined();
    });
    (0, vitest_1.it)('should normalize Schema Only with Options (Case 1)', () => {
        const options = { temperature: 0.5 };
        const result = (0, createZodLlmClient_js_1.normalizeZodArgs)(TestSchema, options);
        (0, vitest_1.expect)(result.mainInstruction).toContain('Generate a valid JSON');
        (0, vitest_1.expect)(result.userMessagePayload).toBe('Generate the data.');
        (0, vitest_1.expect)(result.dataExtractionSchema).toBe(TestSchema);
        (0, vitest_1.expect)(result.options).toBe(options);
    });
    (0, vitest_1.it)('should normalize Prompt + Schema (Case 2)', () => {
        const prompt = "Extract data";
        const result = (0, createZodLlmClient_js_1.normalizeZodArgs)(prompt, TestSchema);
        (0, vitest_1.expect)(result.mainInstruction).toContain('You are a helpful assistant');
        (0, vitest_1.expect)(result.userMessagePayload).toBe(prompt);
        (0, vitest_1.expect)(result.dataExtractionSchema).toBe(TestSchema);
        (0, vitest_1.expect)(result.options).toBeUndefined();
    });
    (0, vitest_1.it)('should normalize Prompt + Schema with Options (Case 2)', () => {
        const prompt = "Extract data";
        const options = { temperature: 0.5 };
        const result = (0, createZodLlmClient_js_1.normalizeZodArgs)(prompt, TestSchema, options);
        (0, vitest_1.expect)(result.mainInstruction).toContain('You are a helpful assistant');
        (0, vitest_1.expect)(result.userMessagePayload).toBe(prompt);
        (0, vitest_1.expect)(result.dataExtractionSchema).toBe(TestSchema);
        (0, vitest_1.expect)(result.options).toBe(options);
    });
    (0, vitest_1.it)('should normalize System + User + Schema (Case 3)', () => {
        const system = "System prompt";
        const user = "User prompt";
        const result = (0, createZodLlmClient_js_1.normalizeZodArgs)(system, user, TestSchema);
        (0, vitest_1.expect)(result.mainInstruction).toBe(system);
        (0, vitest_1.expect)(result.userMessagePayload).toBe(user);
        (0, vitest_1.expect)(result.dataExtractionSchema).toBe(TestSchema);
        (0, vitest_1.expect)(result.options).toBeUndefined();
    });
    (0, vitest_1.it)('should normalize System + User + Schema with Options (Case 3)', () => {
        const system = "System prompt";
        const user = "User prompt";
        const options = { temperature: 0.5 };
        const result = (0, createZodLlmClient_js_1.normalizeZodArgs)(system, user, TestSchema, options);
        (0, vitest_1.expect)(result.mainInstruction).toBe(system);
        (0, vitest_1.expect)(result.userMessagePayload).toBe(user);
        (0, vitest_1.expect)(result.dataExtractionSchema).toBe(TestSchema);
        (0, vitest_1.expect)(result.options).toBe(options);
    });
    (0, vitest_1.it)('should throw error for invalid arguments', () => {
        (0, vitest_1.expect)(() => (0, createZodLlmClient_js_1.normalizeZodArgs)({})).toThrow('Invalid arguments');
    });
});
