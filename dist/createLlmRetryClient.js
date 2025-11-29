"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.LlmRetryAttemptError = exports.LlmRetryExhaustedError = exports.LlmRetryError = void 0;
exports.createLlmRetryClient = createLlmRetryClient;
const createLlmClient_js_1 = require("./createLlmClient.js");
// Custom error for the querier to handle, allowing retries with structured feedback.
class LlmRetryError extends Error {
    message;
    type;
    details;
    rawResponse;
    constructor(message, type, details, rawResponse) {
        super(message);
        this.message = message;
        this.type = type;
        this.details = details;
        this.rawResponse = rawResponse;
        this.name = 'LlmRetryError';
    }
}
exports.LlmRetryError = LlmRetryError;
class LlmRetryExhaustedError extends Error {
    message;
    constructor(message, options) {
        super(message, options);
        this.message = message;
        this.name = 'LlmRetryExhaustedError';
    }
}
exports.LlmRetryExhaustedError = LlmRetryExhaustedError;
// This error is thrown by LlmRetryClient for each failed attempt.
// It wraps the underlying error (from API call or validation) and adds context.
class LlmRetryAttemptError extends Error {
    message;
    mode;
    conversation;
    attemptNumber;
    constructor(message, mode, conversation, attemptNumber, options) {
        super(message, options);
        this.message = message;
        this.mode = mode;
        this.conversation = conversation;
        this.attemptNumber = attemptNumber;
        this.name = 'LlmRetryAttemptError';
    }
}
exports.LlmRetryAttemptError = LlmRetryAttemptError;
function constructLlmMessages(initialMessages, attemptNumber, previousError) {
    if (attemptNumber === 0) {
        // First attempt
        return initialMessages;
    }
    if (!previousError) {
        // Should not happen for attempt > 0, but as a safeguard...
        throw new Error("Invariant violation: previousError is missing for a retry attempt.");
    }
    const cause = previousError.cause;
    if (!(cause instanceof LlmRetryError)) {
        throw Error('cause must be an instanceof LlmRetryError');
    }
    const messages = [...previousError.conversation];
    messages.push({ role: "user", content: cause.message });
    return messages;
}
function createLlmRetryClient(params) {
    const { prompt, fallbackPrompt } = params;
    async function runPromptLoop(options, responseType) {
        const { maxRetries = 3, validate, messages, ...restOptions } = options;
        // Ensure messages is an array (normalizeOptions ensures this but types might be loose)
        const initialMessages = messages;
        let lastError;
        for (let attempt = 0; attempt <= maxRetries; attempt++) {
            const useFallback = !!fallbackPrompt && attempt > 0;
            const currentPrompt = useFallback ? fallbackPrompt : prompt;
            const mode = useFallback ? 'fallback' : 'main';
            const currentMessages = constructLlmMessages(initialMessages, attempt, lastError);
            try {
                const completion = await currentPrompt({
                    messages: currentMessages,
                    ...restOptions,
                });
                const assistantMessage = completion.choices[0]?.message;
                let dataToProcess = completion;
                if (responseType === 'text') {
                    const content = assistantMessage?.content;
                    if (content === null || content === undefined) {
                        throw new LlmRetryError("LLM returned no text content.", 'CUSTOM_ERROR', undefined, JSON.stringify(completion));
                    }
                    dataToProcess = content;
                }
                else if (responseType === 'image') {
                    const messageAny = assistantMessage;
                    if (messageAny.images && Array.isArray(messageAny.images) && messageAny.images.length > 0) {
                        const imageUrl = messageAny.images[0].image_url.url;
                        if (typeof imageUrl === 'string') {
                            if (imageUrl.startsWith('http')) {
                                const imgRes = await fetch(imageUrl);
                                const arrayBuffer = await imgRes.arrayBuffer();
                                dataToProcess = Buffer.from(arrayBuffer);
                            }
                            else {
                                const base64Data = imageUrl.replace(/^data:image\/\w+;base64,/, "");
                                dataToProcess = Buffer.from(base64Data, 'base64');
                            }
                        }
                        else {
                            throw new LlmRetryError("LLM returned invalid image URL.", 'CUSTOM_ERROR', undefined, JSON.stringify(completion));
                        }
                    }
                    else {
                        throw new LlmRetryError("LLM returned no image.", 'CUSTOM_ERROR', undefined, JSON.stringify(completion));
                    }
                }
                // Construct conversation history for success or potential error reporting
                const finalConversation = [...currentMessages];
                if (assistantMessage) {
                    finalConversation.push(assistantMessage);
                }
                const info = {
                    mode,
                    conversation: finalConversation,
                    attemptNumber: attempt,
                };
                if (validate) {
                    const result = await validate(dataToProcess, info);
                    return result;
                }
                return dataToProcess;
            }
            catch (error) {
                if (error instanceof LlmRetryError) {
                    // This is a recoverable error, so we'll create a detailed attempt error and continue the loop.
                    const conversationForError = [...currentMessages];
                    // If the error contains the raw response (e.g. the invalid text), add it to history
                    // so the LLM knows what it generated previously.
                    if (error.rawResponse) {
                        conversationForError.push({ role: 'assistant', content: error.rawResponse });
                    }
                    else if (responseType === 'raw' && error.details) {
                        // For raw mode, if we have details, maybe we can infer something, but usually rawResponse is key.
                    }
                    lastError = new LlmRetryAttemptError(`Attempt ${attempt + 1} failed.`, mode, conversationForError, attempt, { cause: error });
                }
                else {
                    // This is a non-recoverable error (e.g., network, API key), so we re-throw it immediately.
                    throw error;
                }
            }
        }
        throw new LlmRetryExhaustedError(`Operation failed after ${maxRetries + 1} attempts.`, { cause: lastError });
    }
    async function promptRetry(arg1, arg2) {
        const options = (0, createLlmClient_js_1.normalizeOptions)(arg1, arg2);
        return runPromptLoop(options, 'raw');
    }
    async function promptTextRetry(arg1, arg2) {
        const options = (0, createLlmClient_js_1.normalizeOptions)(arg1, arg2);
        return runPromptLoop(options, 'text');
    }
    async function promptImageRetry(arg1, arg2) {
        const options = (0, createLlmClient_js_1.normalizeOptions)(arg1, arg2);
        return runPromptLoop(options, 'image');
    }
    return { promptRetry, promptTextRetry, promptImageRetry };
}
