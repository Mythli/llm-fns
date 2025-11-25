import crypto from 'crypto';
import OpenAI from "openai";
import { Cache } from 'cache-manager'; // Using Cache from cache-manager
import { EventTracker } from './EventTracker.js';
import PQueue from 'p-queue';
import { executeWithRetry } from './retryUtils.js';

export function countChars(message: OpenAI.Chat.Completions.ChatCompletionMessageParam): number {
    if (!message.content) return 0;
    if (typeof message.content === 'string') {
        return message.content.length;
    }
    if (Array.isArray(message.content)) {
        return message.content.reduce((sum, part) => {
            if (part.type === 'text') {
                return sum + part.text.length;
            }
            if (part.type === 'image_url') {
                return sum + 2500;
            }
            return sum;
        }, 0);
    }
    return 0;
}

export function truncateSingleMessage(message: OpenAI.Chat.Completions.ChatCompletionMessageParam, charLimit: number): OpenAI.Chat.Completions.ChatCompletionMessageParam {
    const TRUNCATION_SUFFIX = '...[truncated]';
    const messageCopy = JSON.parse(JSON.stringify(message));

    if (charLimit <= 0) {
        messageCopy.content = null;
        return messageCopy;
    }

    if (!messageCopy.content || countChars(messageCopy) <= charLimit) {
        return messageCopy;
    }

    if (typeof messageCopy.content === 'string') {
        let newContent = messageCopy.content;
        if (newContent.length > charLimit) {
            if (charLimit > TRUNCATION_SUFFIX.length) {
                newContent = newContent.substring(0, charLimit - TRUNCATION_SUFFIX.length) + TRUNCATION_SUFFIX;
            } else {
                newContent = newContent.substring(0, charLimit);
            }
        }
        messageCopy.content = newContent;
        return messageCopy;
    }

    if (Array.isArray(messageCopy.content)) {
        // Complex case: multipart message.
        // Strategy: consolidate text, remove images if needed, then truncate text.
        const textParts = messageCopy.content.filter((p: any) => p.type === 'text');
        const imageParts = messageCopy.content.filter((p: any) => p.type === 'image_url');
        let combinedText = textParts.map((p: any) => p.text).join('\n');
        let keptImages = [...imageParts];

        while (combinedText.length + (keptImages.length * 2500) > charLimit && keptImages.length > 0) {
            keptImages.pop(); // remove images from the end
        }

        const imageChars = keptImages.length * 2500;
        const textCharLimit = charLimit - imageChars;

        if (combinedText.length > textCharLimit) {
            if (textCharLimit > TRUNCATION_SUFFIX.length) {
                combinedText = combinedText.substring(0, textCharLimit - TRUNCATION_SUFFIX.length) + TRUNCATION_SUFFIX;
            } else if (textCharLimit >= 0) {
                combinedText = combinedText.substring(0, textCharLimit);
            } else {
                combinedText = "";
            }
        }

        const newContent: OpenAI.Chat.Completions.ChatCompletionContentPart[] = [];
        if (combinedText) {
            newContent.push({ type: 'text', text: combinedText });
        }
        newContent.push(...keptImages);
        messageCopy.content = newContent;
    }

    return messageCopy;
}


export function truncateMessages(messages: OpenAI.Chat.Completions.ChatCompletionMessageParam[], limit: number): OpenAI.Chat.Completions.ChatCompletionMessageParam[] {
    const systemMessage = messages.find(m => m.role === 'system');
    const otherMessages = messages.filter(m => m.role !== 'system');

    let totalChars = otherMessages.reduce((sum: number, msg) => sum + countChars(msg), 0);

    if (totalChars <= limit) {
        return messages;
    }

    const mutableOtherMessages: OpenAI.Chat.Completions.ChatCompletionMessageParam[] = JSON.parse(JSON.stringify(otherMessages));
    let excessChars = totalChars - limit;

    // Truncate messages starting from the second one.
    for (let i = 1; i < mutableOtherMessages.length; i++) {
        if (excessChars <= 0) break;

        const message = mutableOtherMessages[i];
        const messageChars = countChars(message);
        const charsToCut = Math.min(excessChars, messageChars);

        const newCharCount = messageChars - charsToCut;
        mutableOtherMessages[i] = truncateSingleMessage(message, newCharCount);

        excessChars -= charsToCut;
    }

    // If still over limit, truncate the first message.
    if (excessChars > 0) {
        const firstMessage = mutableOtherMessages[0];
        const firstMessageChars = countChars(firstMessage);
        const charsToCut = Math.min(excessChars, firstMessageChars);
        const newCharCount = firstMessageChars - charsToCut;
        mutableOtherMessages[0] = truncateSingleMessage(firstMessage, newCharCount);
    }

    // Filter out empty messages (char count is 0)
    const finalMessages = mutableOtherMessages.filter(msg => countChars(msg) > 0);

    return systemMessage ? [systemMessage, ...finalMessages] : finalMessages;
}

function concatMessageText(messages: OpenAI.Chat.Completions.ChatCompletionMessageParam[]): string {
    const textParts: string[] = [];
    for (const message of messages) {
        if (message.content) {
            if (typeof message.content === 'string') {
                textParts.push(message.content);
            } else if (Array.isArray(message.content)) {
                for (const part of message.content) {
                    if (part.type === 'text') {
                        textParts.push(part.text);
                    } else if (part.type === 'image_url') {
                        textParts.push('[IMAGE]');
                    }
                }
            }
        }
    }
    return textParts.join(' ');
}

function getPromptSummary(messages: OpenAI.Chat.Completions.ChatCompletionMessageParam[]): string {
    const fullText = concatMessageText(messages);
    // Replace multiple whitespace chars with a single space and trim.
    const cleanedText = fullText.replace(/\s+/g, ' ').trim();
    // Truncate to a reasonable length.
    const maxLength = 150;
    if (cleanedText.length > maxLength) {
        return cleanedText.substring(0, maxLength) + '...';
    }
    return cleanedText;
}

/**
 * The response format for OpenAI and OpenRouter.
 * OpenRouter extends this with 'json_schema'.
 */
export type ModelConfig = string | ({ model?: string } & Record<string, any>);

export type OpenRouterResponseFormat =
    | { type: 'text' | 'json_object' }
    | {
    type: 'json_schema';
    json_schema: {
        name: string;
        strict?: boolean;
        schema: object;
    };
};


/**
 * Options for the individual "ask" function calls.
 * These can override defaults or add call-specific parameters.
 * 'messages' is a required property, inherited from OpenAI.Chat.Completions.ChatCompletionCreateParamsNonStreaming.
 */
export interface GptAskOptions extends Omit<OpenAI.Chat.Completions.ChatCompletionCreateParamsNonStreaming, 'model' | 'response_format' | 'modalities'> {
    model?: ModelConfig;    // Allow overriding the default model for a specific call
    ttl?: number;      // Cache TTL in *MILLISECONDS* for this specific call, used if cache is enabled
    retries?: number;  // Number of retries for the API call.
    /** @deprecated Use `reasoning` object instead. */
    response_format?: OpenRouterResponseFormat;
    modalities?: string[];
    image_config?: {
        aspect_ratio?: string;
    };
}

/**
 * Options required to create an instance of the gptAsk function.
 * These are the core dependencies.
 */
export interface CreateCachedGptAskParams {
    openai: OpenAI;
    cache?: Cache; // Cache instance is now optional. Expect a cache-manager compatible instance if provided.
    defaultModel: ModelConfig; // The default OpenAI model to use if not overridden in GptAskOptions
    eventTracker?: EventTracker;
    maxConversationChars?: number;
    queue?: PQueue;
}

/**
 * Factory function that creates a GPT "ask" function, with optional caching.
 * @param params - The core dependencies (API key, base URL, default model, and optional cache instance).
 * @returns An async function `gptAsk` ready to make OpenAI calls, with caching if configured.
 */
export function createCachedGptAsk(params: CreateCachedGptAskParams) {
    const { openai, cache: cacheInstance, defaultModel: factoryDefaultModel, eventTracker, maxConversationChars, queue } = params;

    const getCompletionParamsAndCacheKey = (options: GptAskOptions) => {
        const { ttl, model: callSpecificModel, messages, reasoning_effort, retries, ...restApiOptions } = options;

        const finalMessages = maxConversationChars ? truncateMessages(messages, maxConversationChars) : messages;

        const baseConfig = typeof factoryDefaultModel === 'object' && factoryDefaultModel !== null
            ? factoryDefaultModel
            : (typeof factoryDefaultModel === 'string' ? { model: factoryDefaultModel } : {});

        const overrideConfig = typeof callSpecificModel === 'object' && callSpecificModel !== null
            ? callSpecificModel
            : (typeof callSpecificModel === 'string' ? { model: callSpecificModel } : {});

        const modelConfig = { ...baseConfig, ...overrideConfig };

        const { model: modelToUse, ...modelParams } = modelConfig;

        if (typeof modelToUse !== 'string' || !modelToUse) {
            throw new Error('A model must be specified either in the default configuration or in the ask options.');
        }

        const completionParams = {
            ...modelParams,
            model: modelToUse,
            messages: finalMessages,
            ...restApiOptions,
        };

        let cacheKey: string | undefined;
        if (cacheInstance) {
            const cacheKeyString = JSON.stringify(completionParams);
            cacheKey = `gptask:${crypto.createHash('md5').update(cacheKeyString).digest('hex')}`;
        }

        return { completionParams, cacheKey, ttl, modelToUse, finalMessages, retries };
    };

    async function gptAsk(options: GptAskOptions): Promise<OpenAI.Chat.Completions.ChatCompletion> {
        const { completionParams, cacheKey, ttl, modelToUse, finalMessages, retries } = getCompletionParamsAndCacheKey(options);

        if (cacheInstance && cacheKey) {
            try {
                const cachedResponse = await cacheInstance.get<string>(cacheKey);
                if (cachedResponse !== undefined && cachedResponse !== null) {
                    return JSON.parse(cachedResponse);
                }
            } catch (error) {
                console.warn("Cache get error:", error);
            }
        }

        const apiCallAndCache = async (): Promise<OpenAI.Chat.Completions.ChatCompletion> => {
            const task = () => executeWithRetry<OpenAI.Chat.Completions.ChatCompletion, OpenAI.Chat.Completions.ChatCompletion>(
                async () => {
                    return openai.chat.completions.create(completionParams as any);
                },
                async (completion) => {
                    return { isValid: true, data: completion };
                },
                retries ?? 3
            );

            const response = (await (queue ? queue.add(task) : task())) as OpenAI.Chat.Completions.ChatCompletion;

            if (cacheInstance && response && cacheKey) {
                try {
                    await cacheInstance.set(cacheKey, JSON.stringify(response), ttl);
                } catch (error) {
                    console.warn("Cache set error:", error);
                }
            }
            return response;
        };

        if (eventTracker) {
            const promptSummary = getPromptSummary(finalMessages);
            return eventTracker.trackOperation('gpt.ask', { model: modelToUse, prompt: promptSummary }, apiCallAndCache);
        }
        return apiCallAndCache();
    }

    async function isAskCached(options: GptAskOptions): Promise<boolean> {
        const { cacheKey } = getCompletionParamsAndCacheKey(options);
        if (!cacheInstance || !cacheKey) {
            return false;
        }
        try {
            const cachedResponse = await cacheInstance.get<string>(cacheKey);
            return cachedResponse !== undefined && cachedResponse !== null;
        } catch (error) {
            console.warn("Cache get error:", error);
            return false;
        }
    }

    return { ask: gptAsk, isAskCached };
}

export type AskGptFunction = ReturnType<typeof createCachedGptAsk>['ask'];
export type IsAskCachedFunction = ReturnType<typeof createCachedGptAsk>['isAskCached'];
