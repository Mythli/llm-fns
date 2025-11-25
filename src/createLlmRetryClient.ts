import OpenAI from 'openai';
import { PromptFunction, LlmPromptOptions } from "./createLlmClient.js";

// Custom error for the querier to handle, allowing retries with structured feedback.
export class LlmRetryError extends Error {
    constructor(
        public readonly message: string,
        public readonly type: 'JSON_PARSE_ERROR' | 'CUSTOM_ERROR',
        public readonly details?: any,
        public readonly rawResponse?: string | null,
    ) {
        super(message);
        this.name = 'LlmRetryError';
    }
}

export class LlmRetryExhaustedError extends Error {
    constructor(
        public readonly message: string,
        options?: ErrorOptions
    ) {
        super(message, options);
        this.name = 'LlmRetryExhaustedError';
    }
}

// This error is thrown by LlmRetryClient for each failed attempt.
// It wraps the underlying error (from API call or validation) and adds context.
export class LlmRetryAttemptError extends Error {
    constructor(
        public readonly message: string,
        public readonly mode: 'main' | 'fallback',
        public readonly conversation: OpenAI.Chat.Completions.ChatCompletionMessageParam[],
        public readonly attemptNumber: number,
        options?: ErrorOptions
    ) {
        super(message, options);
        this.name = 'LlmRetryAttemptError';
    }
}

export type LlmRetryOptions = Omit<LlmPromptOptions, 'messages'> & {
    maxRetries?: number;
};

export interface LlmRetryResponseInfo {
    mode: 'main' | 'fallback';
    conversation: OpenAI.Chat.Completions.ChatCompletionMessageParam[];
    attemptNumber: number;
}

export interface CreateLlmRetryClientParams {
    prompt: PromptFunction;
    fallbackPrompt?: PromptFunction;
}

function constructLlmMessages(
    mainInstruction: string,
    userMessagePayload: string | OpenAI.Chat.Completions.ChatCompletionContentPart[],
    attemptNumber: number,
    previousError?: LlmRetryAttemptError
): OpenAI.Chat.Completions.ChatCompletionMessageParam[] {
    if (attemptNumber === 0) {
        // First attempt
        return [
            { role: "system", content: mainInstruction },
            { role: "user", content: userMessagePayload }
        ];
    }

    if (!previousError) {
        // Should not happen for attempt > 0, but as a safeguard...
        throw new Error("Invariant violation: previousError is missing for a retry attempt.");
    }
    const cause = previousError.cause;

    if (!(cause instanceof LlmRetryError)) {
        throw Error('cause must be an instanceof LlmRetryError')
    }

    const messages: OpenAI.Chat.Completions.ChatCompletionMessageParam[] = [...previousError.conversation];

    messages.push({ role: "user", content: cause.message });

    return messages;
}

export function createLlmRetryClient(params: CreateLlmRetryClientParams) {
    const { prompt, fallbackPrompt } = params;

    async function runPromptLoop<T>(
        mainInstruction: string,
        userMessagePayload: string | OpenAI.Chat.Completions.ChatCompletionContentPart[],
        processResponse: (response: any, info: LlmRetryResponseInfo) => Promise<T>,
        options: LlmRetryOptions | undefined,
        responseType: 'raw' | 'text' | 'image'
    ): Promise<T> {
        const maxRetries = options?.maxRetries || 3;
        let lastError: LlmRetryAttemptError | undefined;

        for (let attempt = 0; attempt <= maxRetries; attempt++) {
            const useFallback = !!fallbackPrompt && attempt > 0;
            const currentPrompt = useFallback ? fallbackPrompt! : prompt;
            const mode = useFallback ? 'fallback' : 'main';

            const messages = constructLlmMessages(
                mainInstruction,
                userMessagePayload,
                attempt,
                lastError
            );

            const { maxRetries: _maxRetries, ...restOptions } = options || {};

            try {
                const completion = await currentPrompt({
                    messages: messages,
                    ...restOptions,
                });

                const assistantMessage = completion.choices[0]?.message;
                let dataToProcess: any = completion;
                
                if (responseType === 'text') {
                    const content = assistantMessage?.content;
                    if (content === null || content === undefined) {
                        throw new LlmRetryError("LLM returned no text content.", 'CUSTOM_ERROR', undefined, JSON.stringify(completion));
                    }
                    dataToProcess = content;
                } else if (responseType === 'image') {
                    const messageAny = assistantMessage as any;
                    if (messageAny.images && Array.isArray(messageAny.images) && messageAny.images.length > 0) {
                        const imageUrl = messageAny.images[0].image_url.url;
                        if (typeof imageUrl === 'string') {
                            if (imageUrl.startsWith('http')) {
                                const imgRes = await fetch(imageUrl);
                                const arrayBuffer = await imgRes.arrayBuffer();
                                dataToProcess = Buffer.from(arrayBuffer);
                            } else {
                                const base64Data = imageUrl.replace(/^data:image\/\w+;base64,/, "");
                                dataToProcess = Buffer.from(base64Data, 'base64');
                            }
                        } else {
                            throw new LlmRetryError("LLM returned invalid image URL.", 'CUSTOM_ERROR', undefined, JSON.stringify(completion));
                        }
                    } else {
                        throw new LlmRetryError("LLM returned no image.", 'CUSTOM_ERROR', undefined, JSON.stringify(completion));
                    }
                }

                // Construct conversation history for success or potential error reporting
                const finalConversation = [...messages];
                if (assistantMessage) {
                    finalConversation.push(assistantMessage);
                }

                const info: LlmRetryResponseInfo = {
                    mode,
                    conversation: finalConversation,
                    attemptNumber: attempt,
                };

                // processResponse is expected to throw LlmRetryError for validation failures.
                const result = await processResponse(dataToProcess, info);
                return result; // Success

            } catch (error: any) {
                if (error instanceof LlmRetryError) {
                    // This is a recoverable error, so we'll create a detailed attempt error and continue the loop.
                    const conversationForError = [...messages];
                    
                    // If the error contains the raw response (e.g. the invalid text), add it to history
                    // so the LLM knows what it generated previously.
                    if (error.rawResponse) {
                        conversationForError.push({ role: 'assistant', content: error.rawResponse });
                    } else if (responseType === 'raw' && error.details) {
                        // For raw mode, if we have details, maybe we can infer something, but usually rawResponse is key.
                    }

                    lastError = new LlmRetryAttemptError(
                        `Attempt ${attempt + 1} failed.`,
                        mode,
                        conversationForError, 
                        attempt,
                        { cause: error }
                    );
                } else {
                    // This is a non-recoverable error (e.g., network, API key), so we re-throw it immediately.
                    throw error;
                }
            }
        }

        throw new LlmRetryExhaustedError(
            `Operation failed after ${maxRetries + 1} attempts.`,
            { cause: lastError }
        );
    }

    async function promptRetry<T>(
        mainInstruction: string,
        userMessagePayload: string | OpenAI.Chat.Completions.ChatCompletionContentPart[],
        processResponse: (response: OpenAI.Chat.Completions.ChatCompletion, info: LlmRetryResponseInfo) => Promise<T>,
        options?: LlmRetryOptions
    ): Promise<T> {
        return runPromptLoop(mainInstruction, userMessagePayload, processResponse, options, 'raw');
    }

    async function promptTextRetry<T>(
        mainInstruction: string,
        userMessagePayload: string | OpenAI.Chat.Completions.ChatCompletionContentPart[],
        processResponse: (response: string, info: LlmRetryResponseInfo) => Promise<T>,
        options?: LlmRetryOptions
    ): Promise<T> {
        return runPromptLoop(mainInstruction, userMessagePayload, processResponse, options, 'text');
    }

    async function promptImageRetry<T>(
        mainInstruction: string,
        userMessagePayload: string | OpenAI.Chat.Completions.ChatCompletionContentPart[],
        processResponse: (response: Buffer, info: LlmRetryResponseInfo) => Promise<T>,
        options?: LlmRetryOptions
    ): Promise<T> {
        return runPromptLoop(mainInstruction, userMessagePayload, processResponse, options, 'image');
    }

    return { promptRetry, promptTextRetry, promptImageRetry };
}
