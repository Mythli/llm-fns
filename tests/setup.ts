import OpenAI from 'openai';
import { createCache } from 'cache-manager';
import KeyvSqlite from '@keyv/sqlite';
import { createLlm } from '../src/llmFactory.js';
import { env } from './env.js';

export async function createTestLlm() {
    const openai = new OpenAI({
        apiKey: env.OPENAI_API_KEY,
        baseURL: env.OPENAI_BASE_URL,
    });

    // Create a SQLite cache for testing
    const sqliteStore = new KeyvSqlite('sqlite://test-cache.sqlite');
    const cache = createCache({ stores: [sqliteStore as any] });

    const llm = createLlm({
        openai,
        cache,
        defaultModel: env.TEST_MODEL,
    });

    return { llm, cache };
}
