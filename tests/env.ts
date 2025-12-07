import { z } from 'zod';
import dotenv from 'dotenv';

// Load .env.test explicitly if available, falling back to .env
dotenv.config({ path: '.env.test' });

const envSchema = z.object({
    TEST_API_KEY: z.string().min(1, "OPENAI_API_KEY is required"),
    // Optional override for base URL (e.g. for OpenRouter or local proxies)
    TEST_BASE_URL: z.string().url().optional(),
    // Model to use for testing. Defaults to a cheaper model.
    TEST_MODEL: z.string().default("openai/gpt-oss-120b"),
});

export const env = envSchema.parse(process.env);
