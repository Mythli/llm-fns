# LLM Client Wrapper for OpenAI

A generic, type-safe wrapper around the OpenAI API. It abstracts away the boilerplate (parsing, retries, caching, logging) while allowing raw access when needed. 

Designed for power users who need to switch between simple string prompts and complex, resilient agentic workflows.

## Installation

```bash
npm install openai zod cache-manager p-queue
```

## Quick Start (Factory)

The `createLlm` factory bundles all functionality (Basic, Retry, Zod) into a single client.

```typescript
import OpenAI from 'openai';
import { createLlm } from './src'; 

const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

const llm = createLlm({
    openai,
    defaultModel: 'google/gemini-3-pro-preview',
    // optional: 
    // cache: Cache instance (cache-manager)
    // queue: PQueue instance for concurrency control
    // maxConversationChars: number (auto-truncation)
});
```

---

# Use Case 1: Text & Chat (`llm.prompt` / `llm.promptText`)

### Level 1: The Easy Way (String Output)
Use `promptText` when you just want the answer as a string.

**Return Type:** `Promise<string>`

```typescript
// 1. Simple User Question
const ans1 = await llm.promptText("Why is the sky blue?");

// 2. System Instruction + User Question
const ans2 = await llm.promptText("You are a poet", "Describe the sea");

// 3. Conversation History (Chat Bots)
const ans3 = await llm.promptText([ 
    { role: "user", content: "Hi" },
    { role: "assistant", content: "Ho" } 
]);
```

### Level 2: The Raw Object (Shortcuts)
Use `prompt` when you need the **Full OpenAI Response** (`usage`, `id`, `choices`, `finish_reason`) but want to use the **Simple Inputs** from Level 1.

**Return Type:** `Promise<OpenAI.Chat.Completions.ChatCompletion>`

```typescript
// Shortcut A: Single String -> User Message
const res1 = await llm.prompt("Why is the sky blue?");
console.log(res1.usage.total_tokens); // Access generic OpenAI properties

// Shortcut B: Two Strings -> System + User
const res2 = await llm.prompt(
    "You are a SQL Expert.",       // System
    "Write a query for users."     // User
);
```

### Level 3: Full Control (Config Object)
Use the **Config Object** overload for absolute control. This allows you to mix Standard OpenAI flags with Library flags.

**Input Type:** `LlmPromptOptions`

```typescript
const res = await llm.prompt({
    // Standard OpenAI params
    messages: [{ role: "user", content: "Hello" }],
    temperature: 1.5,
    frequency_penalty: 0.2,
    max_tokens: 100,
    
    // Library Extensions
    model: "gpt-4o",    // Override default model for this call
    ttl: 5000,          // Cache this specific call for 5s (in ms)
    retries: 5,         // Retry network errors 5 times
});
```

---

# Use Case 2: Images (`llm.promptImage`)

Generates an image and returns it as a `Buffer`. This handles the fetching of the URL or Base64 decoding automatically.

**Return Type:** `Promise<Buffer>`

```typescript
// 1. Simple Generation
const buffer1 = await llm.promptImage("A cyberpunk cat");

// 2. Advanced Configuration (Model & Aspect Ratio)
const buffer2 = await llm.promptImage({
    messages: "A cyberpunk cat",
    model: "dall-e-3",          // Override default model
    size: "1024x1024",          // OpenAI specific params pass through
    quality: "hd"
});

// fs.writeFileSync('cat.png', buffer2);
```

---

# Use Case 3: Structured Data (`llm.promptJson` & `llm.promptZod`)

This is a high-level wrapper that employs a **Re-asking Loop**. If the LLM outputs invalid JSON or data that fails the schema validation, the client automatically feeds the error back to the LLM and asks it to fix it (up to `maxRetries`).

**Return Type:** `Promise<T>`

### Level 1: Raw JSON Schema (`promptJson`)
Use this if you have a standard JSON Schema object (e.g. from another library or API) and don't want to use Zod.

```typescript
const MySchema = {
    type: "object",
    properties: {
        sentiment: { type: "string", enum: ["positive", "negative"] },
        score: { type: "number" }
    },
    required: ["sentiment", "score"],
    additionalProperties: false
};

const result = await llm.promptJson(
    [{ role: "user", content: "I love this!" }],
    MySchema,
    (data) => data // Optional validator function
);
```

### Level 2: Zod Wrapper (`promptZod`)
This is syntactic sugar over `promptJson`. It converts your Zod schema to JSON Schema and automatically sets up the validator to throw formatted Zod errors for the retry loop.

**Return Type:** `Promise<z.infer<typeof Schema>>`

```typescript
import { z } from 'zod';
const UserSchema = z.object({ name: z.string(), age: z.number() });

// 1. Schema Only (Hallucinate data)
const user = await llm.promptZod(UserSchema);

// 2. Extraction (Context + Schema)
const email = "Meeting at 2 PM with Bob.";
const event = await llm.promptZod(email, z.object({ time: z.string(), who: z.string() }));

// 3. Full Control (History + Schema + Options)
const history = [
    { role: "user", content: "I cast Fireball." },
    { role: "assistant", content: "It misses." }
];

const gameState = await llm.promptZod(
    history,             // Arg 1: Context
    GameStateSchema,     // Arg 2: Schema
    {                    // Arg 3: Options Override
        model: "google/gemini-flash-1.5", 
        disableJsonFixer: true, // Turn off the automatic JSON repair agent
        maxRetries: 0,          // Fail immediately on error
        ttl: 60000              // Cache result
    }
);
```

### Level 3: Hooks & Pre-processing
Sometimes LLMs output data that is *almost* correct (e.g., strings for numbers). You can sanitize data before validation runs.

```typescript
const result = await llm.promptZod(MySchema, {
    // Transform JSON before validation runs
    beforeValidation: (data) => {
        if (data.price && typeof data.price === 'string') {
            return { ...data, price: parseFloat(data.price) };
        }
        return data;
    },
    
    // Toggle usage of 'response_format: { type: "json_object" }'
    useResponseFormat: false 
});
```

---

# Use Case 4: Agentic Retry Loops (`llm.promptTextRetry`)

The library exposes the "Conversational Retry" engine used internally by `promptZod`. You can provide a `validate` function. If it throws a `LlmRetryError`, the error message is fed back to the LLM, and it tries again.

**Return Type:** `Promise<string>` (or generic `<T>`)

```typescript
import { LlmRetryError } from './src';

const poem = await llm.promptTextRetry({
    messages: "Write a haiku about coding.",
    maxRetries: 3,
    validate: async (text, info) => {
        // 'info' contains history and attempt number
        // info: { attemptNumber: number, conversation: [...], mode: 'main'|'fallback' }
        
        if (!text.toLowerCase().includes("bug")) {
            // This message goes back to the LLM:
            // User: "Please include the word 'bug'."
            throw new LlmRetryError("Please include the word 'bug'.", 'CUSTOM_ERROR');
        }
        return text;
    }
});
```

---

# Use Case 5: Architecture & Composition

How to build the client manually to enable **Fallback Chains** and **Smart Routing**.

### Level 1: The Base Client (`createLlmClient`)
This creates the underlying engine that generates Text and Images. It handles Caching and Queuing but *not* Zod or Retry Loops.

```typescript
import { createLlmClient } from './src';

// 1. Define a CHEAP model
const cheapClient = createLlmClient({ 
    openai, 
    defaultModel: 'google/gemini-flash-1.5' 
});

// 2. Define a STRONG model
const strongClient = createLlmClient({ 
    openai, 
    defaultModel: 'google/gemini-3-pro-preview' 
});
```

### Level 2: The Zod Client (`createZodLlmClient`)
This wraps a Base Client with the "Fixer" logic. You inject the `prompt` function you want it to use.

```typescript
import { createZodLlmClient } from './src';

// A standard Zod client using only the strong model
const zodClient = createZodLlmClient({
    prompt: strongClient.prompt, 
    isPromptCached: strongClient.isPromptCached
});
```

### Level 3: The Fallback Chain (Smart Routing)
Link two clients together. If the `prompt` function of the first client fails (retries exhausted, refusal, or unfixable JSON), it switches to the `fallbackPrompt`.

```typescript
const smartClient = createZodLlmClient({
    // Primary Strategy: Try Cheap/Fast
    prompt: cheapClient.prompt,
    isPromptCached: cheapClient.isPromptCached,

    // Fallback Strategy: Switch to Strong/Expensive
    // This is triggered if the Primary Strategy exhausts its retries or validation fails
    fallbackPrompt: strongClient.prompt,
});

// Usage acts exactly like the standard client
await smartClient.promptZod(MySchema);
```

---

# Utilities: Cache Inspection

Check if a specific prompt is already cached without making an API call (or partial cache check for Zod calls).

**Return Type:** `Promise<boolean>`

```typescript
const options = { messages: "Compare 5000 files..." };

// 1. Check Standard Call
if (await llm.isPromptCached(options)) {
    console.log("Zero latency result available!");
}

// 2. Check Zod Call (checks exact schema + prompt combo)
if (await llm.isPromptZodCached(options, MySchema)) {
    // ...
}
```
