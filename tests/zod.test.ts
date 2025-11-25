import { describe, it, expect } from 'vitest';
import { z } from 'zod';
import { createTestLlm } from './setup.js';

describe('Zod Structured Output Integration', () => {
    it('should extract structured data matching a schema', async () => {
        const { llm } = await createTestLlm();

        const InventorySchema = z.object({
            items: z.array(z.object({
                name: z.string(),
                quantity: z.number(),
                color: z.string().optional(),
            }))
        });

        const text = "I have 3 red apples and 2 yellow bananas in my basket.";

        const result = await llm.promptZod(
            "Extract the inventory items from the user text.",
            [{ type: 'text', text }],
            InventorySchema
        );

        expect(result).toBeDefined();
        expect(result.items).toHaveLength(2);
        
        const apple = result.items.find(i => i.name.toLowerCase().includes('apple'));
        expect(apple).toBeDefined();
        expect(apple?.quantity).toBe(3);
        expect(apple?.color).toBe('red');

        const banana = result.items.find(i => i.name.toLowerCase().includes('banana'));
        expect(banana).toBeDefined();
        expect(banana?.quantity).toBe(2);
        expect(banana?.color).toBe('yellow');
    });

    it('should handle validation retries (simulated by strict schema)', async () => {
        const { llm } = await createTestLlm();

        // A schema that requires specific formatting that might be missed initially
        // forcing the retry/fixer logic to kick in if the model is lazy.
        const MathSchema = z.object({
            result: z.number(),
            explanation: z.string().max(50), // Short constraint
        });

        const result = await llm.promptZod(
            "Calculate 25 * 4 and explain briefly.",
            [{ type: 'text', text: "Do the math." }],
            MathSchema
        );

        expect(result.result).toBe(100);
        expect(result.explanation.length).toBeLessThanOrEqual(50);
    });
});
