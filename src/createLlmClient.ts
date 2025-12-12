import crypto from 'crypto';
import OpenAI from "openai";
import type PQueue from 'p-queue';
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
        const textParts = messageCopy.content.filter((p: any) => p.type === 'text');
        const imageParts = messageCopy.content.filter((p: any) => p.type === 'image_url');
        let combinedText = textParts.map((p: any) => p.text).join('\n');
        let keptImages = [...imageParts];

        while (combinedText.length + (keptImages.length * 2500) > charLimit && keptImages.length > 0) {
            keptImages.pop();
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

    for (let i = 1; i < mutableOtherMessages.length; i++) {
        if (excessChars <= 0) break;

        const message = mutableOtherMessages[i];
        const messageChars = countChars(message);
        const charsToCut = Math.min(excessChars, messageChars);

        const newCharCount = messageChars - charsToCut;
        mutableOtherMessages[i] = truncateSingleMessage(message, newCharCount);

        excessChars -= charsToCut;
    }

    if (excessChars > 0) {
        const firstMessage = mutableOtherMessages[0];
        const firstMessageChars = countChars(firstMessage);
        const charsToCut = Math.min(excessChars, firstMessageChars);
        const newCharCount = firstMessageChars - charsToCut;
        mutableOtherMessages[0] = truncateSingleMessage(firstMessage, newCharCount);
    }

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
    const cleanedText = fullText.replace(/\s+/g, ' ').trim();

    if (cleanedText.length <= 50) {
        return cleanedText;
    }

    const partLength = 15;
    const start = cleanedText.substring(0, partLength);
    const end = cleanedText.substring(cleanedText.length - partLength);

    const midIndex = Math.floor(cleanedText.length / 2);
    const midStart = Math.max(partLength, midIndex - Math.ceil(partLength / 2));
    const midEnd = Math.min(cleanedText.length - partLength, midStart + partLength);
    const middle = cleanedText.substring(midStart, midEnd);

    return `${start}...${middle}...${end}`;
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
 * Request-level options passed to the OpenAI SDK.
 * These are separate from the body parameters.
 */
export interface LlmRequestOptions {
    headers?: Record<string, string>;
    signal?: AbortSignal;
    timeout?: number;
}

/**
 * Merges two LlmRequestOptions objects.
 * Headers are merged (override wins on conflict), other properties are replaced.
 */
export function mergeRequestOptions(
    base?: LlmRequestOptions,
    override?: LlmRequestOptions
): LlmRequestOptions | undefined {
    if (!base && !override) return undefined;
    if (!base) return override;
    if (!override) return base;

    return {
        ...base,
        ...override,
        headers: {
            ...base.headers,
            ...override.headers
        }
    };
}

/**
 * Common options shared by all prompt functions.
 * Does NOT include messages - those are handled separately.
 */
export interface LlmCommonOptions {
    model?: ModelConfig;
    retries?: number;
    /** @deprecated Use `reasoning` object instead. */
    response_format?: OpenRouterResponseFormat;
    modalities?: string[];
    image_config?: {
        aspect_ratio?: string;
    };
    requestOptions?: LlmRequestOptions;
    temperature?: number;
    max_tokens?: number;
    top_p?: number;
    frequency_penalty?: number;
    presence_penalty?: number;
    stop?: string | string[];
    reasoning_effort?: 'low' | 'medium' | 'high';
    seed?: number;
    user?: string;
    tools?: OpenAI.Chat.Completions.ChatCompletionTool[];
    tool_choice?: OpenAI.Chat.Completions.ChatCompletionToolChoiceOption;
}

/**
 * Options for the individual "prompt" function calls.
 * Allows messages as string or array for convenience.
 */
export interface LlmPromptOptions extends LlmCommonOptions {
    messages: string | OpenAI.Chat.Completions.ChatCompletionMessageParam[];
}

/**
 * Internal normalized params - messages is always an array.
 */
export interface LlmPromptParams extends LlmCommonOptions {
    messages: OpenAI.Chat.Completions.ChatCompletionMessageParam[];
}

/**
 * Options required to create an instance of the LlmClient.
 * These are the core dependencies.
 */
export interface CreateLlmClientParams {
    openai: OpenAI;
    defaultModel: ModelConfig;
    maxConversationChars?: number;
    queue?: PQueue;
    defaultRequestOptions?: LlmRequestOptions;
}

/**
 * Normalizes input arguments to LlmPromptParams.
 * Handles string shorthand and messages-as-string.
 */
export function normalizeOptions(
    arg1: string | LlmPromptOptions, 
    arg2?: LlmCommonOptions
): LlmPromptParams {
    if (typeof arg1 === 'string') {
        return {
            messages: [{ role: 'user', content: arg1 }],
            ...arg2
        };
    }
    const options = arg1;
    if (typeof options.messages === 'string') {
        return {
            ...options,
            messages: [{ role: 'user', content: options.messages }]
        };
    }
    return options as LlmPromptParams;
}

/**
 * Factory function that creates a GPT "prompt" function.
 * @param params - The core dependencies (API key, base URL, default model).
 * @returns An async function `prompt` ready to make OpenAI calls.
 */
export function createLlmClient(params: CreateLlmClientParams) {
    const { 
        openai, 
        defaultModel: factoryDefaultModel, 
        maxConversationChars, 
        queue,
        defaultRequestOptions 
    } = params;

    const getCompletionParams = (promptParams: LlmPromptParams) => {
        const { 
            model: callSpecificModel, 
            messages, 
            retries, 
            requestOptions,
            ...restApiOptions 
        } = promptParams;

        const finalMessages = maxConversationChars 
            ? truncateMessages(messages, maxConversationChars) 
            : messages;

        const baseConfig = typeof factoryDefaultModel === 'object' && factoryDefaultModel !== null
            ? factoryDefaultModel
            : (typeof factoryDefaultModel === 'string' ? { model: factoryDefaultModel } : {});

        const overrideConfig = typeof callSpecificModel === 'object' && callSpecificModel !== null
            ? callSpecificModel
            : (typeof callSpecificModel === 'string' ? { model: callSpecificModel } : {});

        const modelConfig = { ...baseConfig, ...overrideConfig };

        const { model: modelToUse, ...modelParams } = modelConfig;

        if (typeof modelToUse !== 'string' || !modelToUse) {
            throw new Error('A model must be specified either in the default configuration or in the prompt options.');
        }

        const completionParams = {
            ...modelParams,
            model: modelToUse,
            messages: finalMessages,
            ...restApiOptions,
        };

        const mergedRequestOptions = mergeRequestOptions(defaultRequestOptions, requestOptions);

        return { completionParams, modelToUse, finalMessages, retries, requestOptions: mergedRequestOptions };
    };

    async function prompt(content: string, options?: LlmCommonOptions): Promise<OpenAI.Chat.Completions.ChatCompletion>;
    async function prompt(options: LlmPromptOptions): Promise<OpenAI.Chat.Completions.ChatCompletion>;
    async function prompt(arg1: string | LlmPromptOptions, arg2?: LlmCommonOptions): Promise<OpenAI.Chat.Completions.ChatCompletion> {
        const promptParams = normalizeOptions(arg1, arg2);
        const { completionParams, finalMessages, retries, requestOptions } = getCompletionParams(promptParams);

        const promptSummary = getPromptSummary(finalMessages);

        const apiCall = async (): Promise<OpenAI.Chat.Completions.ChatCompletion> => {
            const task = () => executeWithRetry<OpenAI.Chat.Completions.ChatCompletion, OpenAI.Chat.Completions.ChatCompletion>(
                async () => {
                    return openai.chat.completions.create(
                        completionParams as any,
                        requestOptions
                    );
                },
                async (completion) => {
                    if((completion as any).error) {
                        return {
                            isValid: false,
                        }
                    }

                    return { isValid: true, data: completion };
                },
                retries ?? 3,
                undefined,
                (error: any) => {
                    if (error?.status === 401 || error?.code === 'invalid_api_key') {
                        return false;
                    }
                    return true;
                }
            );

            const response = (await (queue ? queue.add(task, { id: promptSummary, messages: finalMessages } as any) : task())) as OpenAI.Chat.Completions.ChatCompletion;
            return response;
        };

        return apiCall();
    }

    async function promptText(content: string, options?: LlmCommonOptions): Promise<string>;
    async function promptText(options: LlmPromptOptions): Promise<string>;
    async function promptText(arg1: string | LlmPromptOptions, arg2?: LlmCommonOptions): Promise<string> {
        const promptParams = normalizeOptions(arg1, arg2);
        const response = await prompt(promptParams);
        const content = response.choices[0]?.message?.content;
        if (content === null || content === undefined) {
            throw new Error("LLM returned no text content.");
        }
        return content;
    }

    async function promptImage(content: string, options?: LlmCommonOptions): Promise<Buffer>;
    async function promptImage(options: LlmPromptOptions): Promise<Buffer>;
    async function promptImage(arg1: string | LlmPromptOptions, arg2?: LlmCommonOptions): Promise<Buffer> {
        const promptParams = normalizeOptions(arg1, arg2);
        const response = await prompt(promptParams);
        const message = response.choices[0]?.message as any;

        if (message.images && Array.isArray(message.images) && message.images.length > 0) {
            const imageUrl = message.images[0].image_url.url;
            if (typeof imageUrl === 'string') {
                if (imageUrl.startsWith('http')) {
                    const imgRes = await fetch(imageUrl);
                    const arrayBuffer = await imgRes.arrayBuffer();
                    return Buffer.from(arrayBuffer);
                } else {
                    const base64Data = imageUrl.replace(/^data:image\/\w+;base64,/, "");
                    return Buffer.from(base64Data, 'base64');
                }
            }
        }
        throw new Error("LLM returned no image content.");
    }

    return { prompt, promptText, promptImage };
}

export type PromptFunction = ReturnType<typeof createLlmClient>['prompt'];
export type PromptTextFunction = ReturnType<typeof createLlmClient>['promptText'];
export type PromptImageFunction = ReturnType<typeof createLlmClient>['promptImage'];
