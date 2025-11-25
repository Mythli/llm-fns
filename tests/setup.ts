import OpenAI from 'openai';
import { createCache } from 'cache-manager';
import { createLlm } from '../src/llmFactory.js';
import { env } from './env.js';

export async function createTestLlm() {
    const openai = new OpenAI({
        apiKey: env.OPENAI_API_KEY,
        baseURL: env.OPENAI_BASE_URL,
    });

    // Create a memory cache for testing
    const cache = createCache();

    const llm = createLlm({
        openai,
        cache,
        defaultModel: env.TEST_MODEL,
    });

    return { llm, cache };
}
