import OpenAI from 'openai';
import { createCache } from 'cache-manager';
import KeyvSqlite from '@keyv/sqlite';
import PQueue from 'p-queue';
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

    const queue = new PQueue({ concurrency: 4 });

    queue.on('active', () => {
        console.log(`[Queue] Active. Pending: ${queue.pending}, Size: ${queue.size}`);
    });

    queue.on('add', () => {
        console.log(`[Queue] Task added. Pending: ${queue.pending}, Size: ${queue.size}`);
    });

    queue.on('next', () => {
        console.log(`[Queue] Task completed. Pending: ${queue.pending}, Size: ${queue.size}`);
    });

    queue.on('idle', () => {
        console.log(`[Queue] Idle.`);
    });

    const llm = createLlm({
        openai,
        cache,
        defaultModel: env.TEST_MODEL,
        queue,
    });

    return { llm, cache, queue };
}
