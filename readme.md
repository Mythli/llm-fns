# LLM Client Wrapper for OpenAI

A generic, type-safe wrapper around the OpenAI API. It abstracts away the boilerplate (parsing, retries, caching) while allowing raw access when needed.

## Installation

```bash
npm install openai zod cache-manager p-queue
```

## Quick Start (Factory)

```typescript
import OpenAI from 'openai';
import { createLlm } from './src'; 

const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

const llm = createLlm({
    openai,
    defaultModel: 'google/gemini-3-pro-preview',
    // optional: cache: ..., queue: ...
});
```

---

# Use Case 1: Text & Chat (`llm.prompt` / `llm.promptText`)

### Level 1: The Easy Way (String Output)
Use `promptText` when you just want the answer as a string.

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
Use `prompt` when you need the **Full OpenAI Response** (`usage`, `id`, `choices`...) but want to use the **Simple Inputs** from Level 1.

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

```typescript
const res = await llm.prompt({
    // Standard OpenAI
    messages: [{ role: "user", content: "Hello" }],
    temperature: 1.5,
    frequency_penalty: 0.2,
    
    // Library Extensions
    ttl: 5000,          // Cache this specific call for 5s
    retries: 5,         // Retry network errors 5 times
    timeout: 10000,     // Kill request after 10s
});
```

---

# Use Case 2: Images (`llm.promptImage`)

Generates an image and returns it as a `Buffer`. This handles the fetching of the URL or Base64 decoding automatically.

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

# Use Case 3: Structured Data (`llm.promptZod`)

### Level 1: Generation (Schema Only)
The client "hallucinates" data matching the shape.

```typescript
import { z } from 'zod';
const UserSchema = z.object({ name: z.string(), age: z.number() });

// Input: Schema only
const user = await llm.promptZod(UserSchema);
// Output: { name: "Alice", age: 32 }
```

### Level 2: Extraction (Injection Shortcuts)
Pass context alongside the schema. This automates the "System Prompt JSON Injection".

```typescript
// 1. Extract from String
const email = "Meeting at 2 PM with Bob.";
const event = await llm.promptZod(email, z.object({ time: z.string(), who: z.string() }));

// 2. Strict Separation (System, User, Schema)
// Useful for auditing code or translations where instructions must not bleed into data.
const analysis = await llm.promptZod(
    "You are a security auditor.", // Arg 1: System
    "function dangerous() {}",     // Arg 2: User Data
    SecuritySchema                 // Arg 3: Schema
);
```

### Level 3: State & Options (History + Config)
Process full chat history into state, and use the **Options Object (4th Argument)** to control the internals (Models, Retries, Caching).

```typescript
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
        retries: 0              // Fail immediately on error
    }
);
```

### Level 4: Hooks & Pre-processing
Sometimes LLMs output data that is *almost* correct (e.g., strings for numbers). You can sanitize data before Zod validation runs.

```typescript
const result = await llm.promptZod(MySchema, {
    // Transform JSON before Zod validation runs
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

```typescript
import { LlmRetryError } from './src';

const poem = await llm.promptTextRetry({
    messages: "Write a haiku about coding.",
    maxRetries: 3,
    validate: async (text, info) => {
        // 'info' contains history and attempt number
        
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
    fallbackPrompt: strongClient.prompt,
    
    // Optional: Hook to log when a fallback happens
    onFallback: (error) => console.warn("Upgrading model due to:", error)
});

// Usage acts exactly like the standard client
await smartClient.promptZod(MySchema);
```

---

# Utilities: Cache Inspection

Check if a specific prompt is already cached without making an API call (or partial cache check for Zod calls).

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
