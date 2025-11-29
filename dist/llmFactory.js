"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.createLlm = createLlm;
const createLlmClient_js_1 = require("./createLlmClient.js");
const createLlmRetryClient_js_1 = require("./createLlmRetryClient.js");
const createZodLlmClient_js_1 = require("./createZodLlmClient.js");
function createLlm(params) {
    const baseClient = (0, createLlmClient_js_1.createLlmClient)(params);
    const retryClient = (0, createLlmRetryClient_js_1.createLlmRetryClient)({
        prompt: baseClient.prompt
    });
    const zodClient = (0, createZodLlmClient_js_1.createZodLlmClient)({
        prompt: baseClient.prompt,
        isPromptCached: baseClient.isPromptCached
    });
    return {
        ...baseClient,
        ...retryClient,
        ...zodClient
    };
}
