import OpenAI from 'openai';
import { 
    PromptFunction, 
    LlmCommonOptions, 
    LlmPromptOptions, 
    LlmPromptParams,
    normalizeOptions,
    LlmFatalError
} from "./createLlmClient.js";

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

export class LlmRetryAttemptError extends Error {
    constructor(
        public readonly message: string,
        public readonly mode: 'main' | 'fallback',
        public readonly conversation: OpenAI.Chat.Completions.ChatCompletionMessageParam[],
        public readonly attemptNumber: number,
        public readonly error: Error,
        public readonly rawResponse?: string | null,
        options?: ErrorOptions
    ) {
        super(message, options);
        this.name = 'LlmRetryAttemptError';
    }
}

export interface LlmRetryResponseInfo {
    mode: 'main' | 'fallback';
    conversation: OpenAI.Chat.Completions.ChatCompletionMessageParam[];
    attemptNumber: number;
}

/**
 * Options for retry prompt functions.
 * Extends common options with retry-specific settings.
 */
export interface LlmRetryOptions<T = any> extends LlmCommonOptions {
    maxRetries?: number;
    validate?: (response: any, info: LlmRetryResponseInfo) => Promise<T>;
}

/**
 * Internal params for retry functions - always has messages array.
 */
interface LlmRetryParams<T = any> extends LlmRetryOptions<T> {
    messages: OpenAI.Chat.Completions.ChatCompletionMessageParam[];
}

export interface CreateLlmRetryClientParams {
    prompt: PromptFunction;
    fallbackPrompt?: PromptFunction;
}

function normalizeRetryOptions<T>(
    arg1: string | LlmPromptOptions,
    arg2?: LlmRetryOptions<T>
): LlmRetryParams<T> {
    const baseParams = normalizeOptions(arg1, arg2);
    return {
        ...baseParams,
        ...arg2,
        messages: baseParams.messages
    };
}

function constructLlmMessages(
    initialMessages: OpenAI.Chat.Completions.ChatCompletionMessageParam[],
    attemptNumber: number,
    previousError?: LlmRetryAttemptError
): OpenAI.Chat.Completions.ChatCompletionMessageParam[] {
    if (attemptNumber === 0) {
        return initialMessages;
    }

    if (!previousError) {
        throw new Error("Invariant violation: previousError is missing for a retry attempt.");
    }
    
    const cause = previousError.error;

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
        retryParams: LlmRetryParams<T>,
        responseType: 'raw' | 'text' | 'image'
    ): Promise<T> {
        const { maxRetries = 3, validate, messages: initialMessages, ...restOptions } = retryParams;

        let lastError: LlmRetryAttemptError | undefined;

        for (let attempt = 0; attempt <= maxRetries; attempt++) {
            const useFallback = !!fallbackPrompt && attempt > 0;
            const currentPrompt = useFallback ? fallbackPrompt! : prompt;
            const mode = useFallback ? 'fallback' : 'main';

            const currentMessages = constructLlmMessages(
                initialMessages,
                attempt,
                lastError
            );

            // Capture raw response for error context
            let rawResponseForError: string | null = null;

            try {
                const completion = await currentPrompt({
                    messages: currentMessages,
                    ...restOptions,
                });

                // Extract raw content immediately
                rawResponseForError = completion.choices[0]?.message?.content || null;

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

                const finalConversation = [...currentMessages];
                if (assistantMessage) {
                    finalConversation.push(assistantMessage);
                }

                const info: LlmRetryResponseInfo = {
                    mode,
                    conversation: finalConversation,
                    attemptNumber: attempt,
                };

                if (validate) {
                    const result = await validate(dataToProcess, info);
                    return result;
                }

                return dataToProcess as T;

            } catch (error: any) {
                if (error instanceof LlmRetryError) {
                    const conversationForError = [...currentMessages];
                    
                    if (error.rawResponse) {
                        conversationForError.push({ role: 'assistant', content: error.rawResponse });
                    } else if (rawResponseForError) {
                        conversationForError.push({ role: 'assistant', content: rawResponseForError });
                    }

                    lastError = new LlmRetryAttemptError(
                        `Attempt ${attempt + 1} failed: ${error.message}`,
                        mode,
                        conversationForError, 
                        attempt,
                        error,
                        error.rawResponse || rawResponseForError,
                        { cause: lastError }
                    );
                } else {
                    // For any other error (ZodError, SchemaValidationError that wasn't fixed, network error, etc.)
                    // We wrap it in LlmFatalError to ensure context is preserved.
                    
                    const fatalMessage = error.message || 'An unexpected error occurred during LLM execution';
                    
                    // If it's already a fatal error, use its cause, otherwise use the error itself
                    const cause = error instanceof LlmFatalError ? error.cause : error;
                    
                    // Use the raw response we captured, or if the error has one (e.g. LlmFatalError from lower client)
                    const responseContent = rawResponseForError || (error as any).rawResponse || null;

                    throw new LlmFatalError(
                        fatalMessage,
                        cause,
                        currentMessages, // This contains the full history of retries
                        responseContent
                    );
                }
            }
        }

        throw new LlmRetryExhaustedError(
            `Operation failed after ${maxRetries + 1} attempts.`,
            { cause: lastError }
        );
    }

    async function promptRetry<T = OpenAI.Chat.Completions.ChatCompletion>(
        content: string,
        options?: LlmRetryOptions<T>
    ): Promise<T>;
    async function promptRetry<T = OpenAI.Chat.Completions.ChatCompletion>(
        options: LlmPromptOptions & LlmRetryOptions<T>
    ): Promise<T>;
    async function promptRetry<T = OpenAI.Chat.Completions.ChatCompletion>(
        arg1: string | (LlmPromptOptions & LlmRetryOptions<T>),
        arg2?: LlmRetryOptions<T>
    ): Promise<T> {
        const retryParams = normalizeRetryOptions<T>(arg1, arg2);
        return runPromptLoop(retryParams, 'raw');
    }

    async function promptTextRetry<T = string>(
        content: string,
        options?: LlmRetryOptions<T>
    ): Promise<T>;
    async function promptTextRetry<T = string>(
        options: LlmPromptOptions & LlmRetryOptions<T>
    ): Promise<T>;
    async function promptTextRetry<T = string>(
        arg1: string | (LlmPromptOptions & LlmRetryOptions<T>),
        arg2?: LlmRetryOptions<T>
    ): Promise<T> {
        const retryParams = normalizeRetryOptions<T>(arg1, arg2);
        return runPromptLoop(retryParams, 'text');
    }

    async function promptImageRetry<T = Buffer>(
        content: string,
        options?: LlmRetryOptions<T>
    ): Promise<T>;
    async function promptImageRetry<T = Buffer>(
        options: LlmPromptOptions & LlmRetryOptions<T>
    ): Promise<T>;
    async function promptImageRetry<T = Buffer>(
        arg1: string | (LlmPromptOptions & LlmRetryOptions<T>),
        arg2?: LlmRetryOptions<T>
    ): Promise<T> {
        const retryParams = normalizeRetryOptions<T>(arg1, arg2);
        return runPromptLoop(retryParams, 'image');
    }

    return { promptRetry, promptTextRetry, promptImageRetry };
}

export type LlmRetryClient = ReturnType<typeof createLlmRetryClient>;
