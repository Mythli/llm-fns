"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
const vitest_1 = require("vitest");
const createLlmClient_js_1 = require("./createLlmClient.js");
(0, vitest_1.describe)('normalizeOptions', () => {
    (0, vitest_1.it)('should normalize a simple string prompt', () => {
        const result = (0, createLlmClient_js_1.normalizeOptions)('Hello world');
        (0, vitest_1.expect)(result).toEqual({
            messages: [{ role: 'user', content: 'Hello world' }]
        });
    });
    (0, vitest_1.it)('should normalize a string prompt with options', () => {
        const result = (0, createLlmClient_js_1.normalizeOptions)('Hello world', { temperature: 0.5 });
        (0, vitest_1.expect)(result).toEqual({
            messages: [{ role: 'user', content: 'Hello world' }],
            temperature: 0.5
        });
    });
    (0, vitest_1.it)('should normalize an options object with string messages', () => {
        const result = (0, createLlmClient_js_1.normalizeOptions)({
            messages: 'Hello world',
            temperature: 0.7
        });
        (0, vitest_1.expect)(result).toEqual({
            messages: [{ role: 'user', content: 'Hello world' }],
            temperature: 0.7
        });
    });
    (0, vitest_1.it)('should pass through an options object with array messages', () => {
        const messages = [{ role: 'user', content: 'Hello' }];
        const result = (0, createLlmClient_js_1.normalizeOptions)({
            messages,
            temperature: 0.7
        });
        (0, vitest_1.expect)(result).toEqual({
            messages,
            temperature: 0.7
        });
    });
});
