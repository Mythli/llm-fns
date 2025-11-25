import { createLlmClient, CreateLlmClientParams } from "./createLlmClient.js";
import { createLlmRetryClient } from "./createLlmRetryClient.js";
import { createZodLlmClient } from "./createZodLlmClient.js";

export interface CreateLlmFactoryParams extends CreateLlmClientParams {
    // Optional overrides for specific sub-clients if needed, but usually just base params
}

export function createLlm(params: CreateLlmFactoryParams) {
    const baseClient = createLlmClient(params);
    
    const retryClient = createLlmRetryClient({
        prompt: baseClient.prompt
    });

    const zodClient = createZodLlmClient({
        prompt: baseClient.prompt
    });

    return {
        ...baseClient,
        ...retryClient,
        ...zodClient
    };
}
