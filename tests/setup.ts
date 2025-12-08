import OpenAI from 'openai';
import PQueue from 'p-queue';
import { createLlm } from '../src/llmFactory.js';
import { env } from './env.js';

export async function createTestLlm() {
    const openai = new OpenAI({
        apiKey: env.TEST_API_KEY,
        baseURL: env.TEST_BASE_URL,
    });

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
        defaultModel: env.TEST_MODEL,
        queue,
    });

    return { llm, queue };
}
