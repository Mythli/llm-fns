import OpenAI from 'openai';
import { AskGptFunction, GptAskOptions } from "./createCachedGptAsk.js";

// Custom error for the querier to handle, allowing retries with structured feedback.
export class LlmQuerierError extends Error {
    constructor(
        public readonly message: string,
        public readonly type: 'JSON_PARSE_ERROR' | 'CUSTOM_ERROR',
        public readonly details?: any,
        public readonly rawResponse?: string | null,
    ) {
        super(message);
        this.name = 'LlmQuerierError';
    }
}

export class LlmRequeryExhaustedError extends Error {
    constructor(
        public readonly message: string,
        options?: ErrorOptions
    ) {
        super(message, options);
        this.name = 'LlmRequeryExhaustedError';
    }
}

// This error is thrown by LlmReQuerier for each failed attempt.
// It wraps the underlying error (from API call or validation) and adds context.
export class LlmAttemptError extends Error {
    constructor(
        public readonly message: string,
        public readonly mode: 'main' | 'fallback',
        public readonly conversation: OpenAI.Chat.Completions.ChatCompletionMessageParam[],
        public readonly attemptNumber: number,
        options?: ErrorOptions
    ) {
        super(message, options);
        this.name = 'LlmAttemptError';
    }
}

export type LlmReQuerierOptions = Omit<GptAskOptions, 'messages'> & {
    maxRetries?: number;
};

export interface LlmResponseInfo {
    mode: 'main' | 'fallback';
    conversation: OpenAI.Chat.Completions.ChatCompletionMessageParam[];
    attemptNumber: number;
}

export interface CreateLlmReQuerierParams {
    ask: AskGptFunction;
    fallbackAsk?: AskGptFunction;
}

function constructLlmMessages(
    mainInstruction: string,
    userMessagePayload: OpenAI.Chat.Completions.ChatCompletionContentPart[],
    attemptNumber: number,
    previousError?: LlmAttemptError
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

    if (!(cause instanceof LlmQuerierError)) {
        throw Error('cause must be an instanceof LlmQuerierError')
    }

    const messages: OpenAI.Chat.Completions.ChatCompletionMessageParam[] = [...previousError.conversation];

    messages.push({ role: "user", content: cause.message });

    return messages;
}

export function createLlmReQuerier(params: CreateLlmReQuerierParams) {
    const { ask, fallbackAsk } = params;

    async function query<T>(
        mainInstruction: string,
        userMessagePayload: OpenAI.Chat.Completions.ChatCompletionContentPart[],
        processResponse: (response: string, info: LlmResponseInfo) => Promise<T>,
        options?: LlmReQuerierOptions
    ): Promise<T> {
        const maxRetries = options?.maxRetries || 3;
        let lastError: LlmAttemptError | undefined;

        for (let attempt = 0; attempt <= maxRetries; attempt++) {
            const useFallback = !!fallbackAsk && attempt > 0;
            const currentAsk = useFallback ? fallbackAsk! : ask;
            const mode = useFallback ? 'fallback' : 'main';

            const messages = constructLlmMessages(
                mainInstruction,
                userMessagePayload,
                attempt,
                lastError
            );

            const { maxRetries: _maxRetries, ...restOptions } = options || {};

            try {
                const llmResponseString = await currentAsk({
                    messages: messages,
                    ...restOptions,
                });

                if (!llmResponseString) {
                    // This is a validation error, so we throw a custom error to be caught for a retry.
                    throw new LlmQuerierError("LLM returned no response.", 'CUSTOM_ERROR');
                }

                const finalConversation = [...messages, { role: 'assistant', content: llmResponseString }] as OpenAI.Chat.Completions.ChatCompletionMessageParam[];

                const info: LlmResponseInfo = {
                    mode,
                    conversation: finalConversation,
                    attemptNumber: attempt,
                };

                // processResponse is expected to throw LlmQuerierError for validation failures.
                const result = await processResponse(llmResponseString, info);
                return result; // Success

            } catch (error: any) {
                if (error instanceof LlmQuerierError) {
                    // This is a recoverable error, so we'll create a detailed attempt error and continue the loop.
                    const conversationForError = [...messages];
                    if (error.rawResponse) {
                        conversationForError.push({ role: 'assistant', content: error.rawResponse });
                    }
                    lastError = new LlmAttemptError(
                        `Attempt ${attempt + 1} failed.`,
                        mode,
                        conversationForError, // The conversation including the assistant's (failed) reply
                        attempt,
                        { cause: error }
                    );
                } else {
                    // This is a non-recoverable error (e.g., network, API key), so we re-throw it immediately.
                    throw error;
                }
            }
        }

        throw new LlmRequeryExhaustedError(
            `Operation failed after ${maxRetries + 1} attempts.`,
            { cause: lastError }
        );
    }

    return { query };
}
