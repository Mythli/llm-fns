# LLM Client Wrapper for OpenAI

A generic, type-safe wrapper around the OpenAI API. It abstracts away the boilerplate (parsing, retries, caching, logging) while allowing raw access when needed. 

Designed for power users who need to switch between simple string prompts and complex, resilient agentic workflows.

## Installation

```bash
npm install openai zod cache-manager p-queue ajv
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
    // defaultRequestOptions: { headers, timeout, signal }
});
```

---

## Model Presets (Temperature & Thinking Level)

The `defaultModel` parameter accepts either a simple string or a **configuration object** that bundles the model name with default parameters like `temperature` or `reasoning_effort`.

### Setting a Default Temperature

```typescript
// Create a "creative" client with high temperature
const creativeWriter = createLlm({
    openai,
    defaultModel: {
        model: 'gpt-4o',
        temperature: 1.2,
        frequency_penalty: 0.5
    }
});

// All calls will use these defaults
await creativeWriter.promptText("Write a poem about the ocean");

// Override for a specific call
await creativeWriter.promptText("Summarize this document", {
    model: { temperature: 0.2 }  // Override just temperature, keeps model
});
```

### Configuring Thinking/Reasoning Models

For models that support extended thinking (like `o1`, `o3`, or Claude with thinking), use `reasoning_effort` or model-specific parameters:

```typescript
// Create a "deep thinker" client for complex reasoning tasks
const reasoner = createLlm({
    openai,
    defaultModel: {
        model: 'o3',
        reasoning_effort: 'high'  // 'low' | 'medium' | 'high'
    }
});

// All calls will use extended thinking
const analysis = await reasoner.promptText("Analyze this complex problem...");

// Create a fast reasoning client for simpler tasks
const quickReasoner = createLlm({
    openai,
    defaultModel: {
        model: 'o3-mini',
        reasoning_effort: 'low'
    }
});
```

### Multiple Preset Clients

A common pattern is to create multiple clients with different presets:

```typescript
// Deterministic client for structured data extraction
const extractorLlm = createLlm({
    openai,
    defaultModel: {
        model: 'gpt-4o-mini',
        temperature: 0
    }
});

// Creative client for content generation
const writerLlm = createLlm({
    openai,
    defaultModel: {
        model: 'gpt-4o',
        temperature: 1.0,
        top_p: 0.95
    }
});

// Reasoning client for complex analysis
const analyzerLlm = createLlm({
    openai,
    defaultModel: {
        model: 'o3',
        reasoning_effort: 'medium'
    }
});

// Use the appropriate client for each task
const data = await extractorLlm.promptZod(DataSchema);
const story = await writerLlm.promptText("Write a short story");
const solution = await analyzerLlm.promptText("Solve this logic puzzle...");
```

### Per-Call Overrides

Any preset can be overridden on individual calls:

```typescript
const llm = createLlm({
    openai,
    defaultModel: {
        model: 'gpt-4o',
        temperature: 0.7
    }
});

// Use defaults
await llm.promptText("Hello");

// Override model entirely
await llm.promptText("Complex task", {
    model: {
        model: 'o3',
        reasoning_effort: 'high'
    }
});

// Override just temperature (keeps default model)
await llm.promptText("Be more creative", {
    temperature: 1.5
});

// Or use short form to switch models
await llm.promptText("Quick task", {
    model: 'gpt-4o-mini'
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
    retries: 5,         // Retry network errors 5 times
    
    // Request-level options (headers, timeout, abort signal)
    requestOptions: {
        headers: { 'X-Cache-Salt': 'v2' },  // Affects cache key
        timeout: 60000,
        signal: abortController.signal
    }
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
Use this if you have a standard JSON Schema object (e.g. from another library or API) and don't want to use Zod. It uses **AJV** internally to validate the response against the schema.

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
    MySchema
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

### Level 4: Retryable Errors in Zod Transforms

You can throw `SchemaValidationError` inside Zod `.transform()` or `.refine()` to trigger the retry loop. This is useful for complex validation logic that can't be expressed in the schema itself.

```typescript
import { z } from 'zod';
import { SchemaValidationError } from './src';

const ProductSchema = z.object({
    name: z.string(),
    price: z.number(),
    currency: z.string()
}).transform((data) => {
    // Custom validation that triggers retry
    if (data.price < 0) {
        throw new SchemaValidationError(
            `Price cannot be negative. Got: ${data.price}. Please provide a valid positive price.`
        );
    }
    
    // Normalize currency
    const validCurrencies = ['USD', 'EUR', 'GBP'];
    if (!validCurrencies.includes(data.currency.toUpperCase())) {
        throw new SchemaValidationError(
            `Invalid currency "${data.currency}". Must be one of: ${validCurrencies.join(', ')}`
        );
    }
    
    return {
        ...data,
        currency: data.currency.toUpperCase()
    };
});

// If the LLM returns { price: -10, ... }, the error message is sent back
// and the LLM gets another chance to fix it
const product = await llm.promptZod("Extract product info from: ...", ProductSchema);
```

**Important:** Only `SchemaValidationError` triggers the retry loop. Other errors (like `TypeError`, database errors, etc.) will bubble up immediately without retry. This prevents infinite loops when there's a bug in your transform logic.

```typescript
const SafeSchema = z.object({
    userId: z.string()
}).transform(async (data) => {
    // This error WILL trigger retry (user can fix the input)
    if (!data.userId.match(/^[a-z0-9]+$/)) {
        throw new SchemaValidationError(
            `Invalid userId format "${data.userId}". Must be lowercase alphanumeric.`
        );
    }
    
    // This error will NOT trigger retry (it's a system error)
    const user = await db.findUser(data.userId);
    if (!user) {
        throw new Error(`User not found: ${data.userId}`); // Bubbles up immediately
    }
    
    return { ...data, user };
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

# Error Handling

The library provides a structured error hierarchy that preserves the full context of failures across retry attempts.

## Error Types

### `LlmRetryError`
Thrown to signal that the current attempt failed but can be retried. The error message is sent back to the LLM.

```typescript
import { LlmRetryError } from './src';

throw new LlmRetryError(
    "The response must include a title field.",  // Message sent to LLM
    'CUSTOM_ERROR',                               // Type: 'JSON_PARSE_ERROR' | 'CUSTOM_ERROR'
    { field: 'title' },                           // Optional details
    '{"name": "test"}'                            // Optional raw response
);
```

### `SchemaValidationError`
A specialized error for schema validation failures. Use this in Zod transforms to trigger retries.

```typescript
import { SchemaValidationError } from './src';

throw new SchemaValidationError("Age must be a positive number");
```

### `LlmRetryAttemptError`
Wraps each failed attempt with full context. These are chained together via `.cause`.

```typescript
interface LlmRetryAttemptError {
    message: string;
    mode: 'main' | 'fallback';           // Which prompt was used
    conversation: ChatCompletionMessageParam[];  // Full message history
    attemptNumber: number;               // 0-indexed attempt number
    error: Error;                        // The original error (LlmRetryError, etc.)
    rawResponse?: string | null;         // The raw LLM response
    cause?: LlmRetryAttemptError;        // Previous attempt's error (chain)
}
```

### `LlmRetryExhaustedError`
Thrown when all retry attempts have been exhausted. Contains the full chain of attempt errors.

```typescript
interface LlmRetryExhaustedError {
    message: string;
    cause: LlmRetryAttemptError;  // The last attempt error (with chain to previous)
}
```

### `LlmFatalError`
Thrown for unrecoverable errors (e.g., 401 Unauthorized, 403 Forbidden). These bypass the retry loop entirely.

```typescript
interface LlmFatalError {
    message: string;
    cause?: any;                         // Original error
    messages?: ChatCompletionMessageParam[];  // The messages that caused the error
    rawResponse?: string | null;         // Raw response if available
}
```

## Error Chain Structure

When retries are exhausted, the error chain looks like this:

```
LlmRetryExhaustedError
  └── cause: LlmRetryAttemptError (Attempt 3)
        ├── error: LlmRetryError (the validation error)
        ├── conversation: [...] (full message history)
        ├── rawResponse: '{"age": "wrong3"}'
        └── cause: LlmRetryAttemptError (Attempt 2)
              ├── error: LlmRetryError
              ├── conversation: [...]
              ├── rawResponse: '{"age": "wrong2"}'
              └── cause: LlmRetryAttemptError (Attempt 1)
                    ├── error: LlmRetryError
                    ├── conversation: [...]
                    ├── rawResponse: '{"age": "wrong1"}'
                    └── cause: undefined
```

## Handling Errors

```typescript
import { 
    LlmRetryExhaustedError, 
    LlmRetryAttemptError,
    LlmFatalError 
} from './src';

try {
    const result = await llm.promptZod(MySchema);
} catch (error) {
    if (error instanceof LlmRetryExhaustedError) {
        console.log('All retries failed');
        
        // Walk the error chain
        let attempt = error.cause;
        while (attempt) {
            console.log(`Attempt ${attempt.attemptNumber + 1}:`);
            console.log(`  Mode: ${attempt.mode}`);
            console.log(`  Error: ${attempt.error.message}`);
            console.log(`  Raw Response: ${attempt.rawResponse}`);
            console.log(`  Conversation length: ${attempt.conversation.length}`);
            
            attempt = attempt.cause as LlmRetryAttemptError | undefined;
        }
    }
    
    if (error instanceof LlmFatalError) {
        console.log('Fatal error (no retry):', error.message);
        console.log('Original messages:', error.messages);
    }
}
```

## Extracting the Last Response

A common pattern is to extract the last LLM response from a failed operation:

```typescript
function getLastResponse(error: LlmRetryExhaustedError): string | null {
    return error.cause?.rawResponse ?? null;
}

function getAllResponses(error: LlmRetryExhaustedError): string[] {
    const responses: string[] = [];
    let attempt = error.cause;
    while (attempt) {
        if (attempt.rawResponse) {
            responses.unshift(attempt.rawResponse); // Add to front (chronological order)
        }
        attempt = attempt.cause as LlmRetryAttemptError | undefined;
    }
    return responses;
}
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
