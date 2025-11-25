import { z } from 'zod';
import dotenv from 'dotenv';

dotenv.config();

const envSchema = z.object({
    OPENAI_API_KEY: z.string().min(1, "OPENAI_API_KEY is required"),
    // Optional override for base URL (e.g. for OpenRouter or local proxies)
    OPENAI_BASE_URL: z.string().url().optional(),
    // Model to use for testing. Defaults to a cheaper model.
    TEST_MODEL: z.string().default("gpt-4o-mini"),
});

export const env = envSchema.parse(process.env);
