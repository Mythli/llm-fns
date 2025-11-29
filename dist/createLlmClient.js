"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.countChars = countChars;
exports.truncateSingleMessage = truncateSingleMessage;
exports.truncateMessages = truncateMessages;
exports.normalizeOptions = normalizeOptions;
exports.createLlmClient = createLlmClient;
const crypto_1 = __importDefault(require("crypto"));
const retryUtils_js_1 = require("./retryUtils.js");
function countChars(message) {
    if (!message.content)
        return 0;
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
function truncateSingleMessage(message, charLimit) {
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
            }
            else {
                newContent = newContent.substring(0, charLimit);
            }
        }
        messageCopy.content = newContent;
        return messageCopy;
    }
    if (Array.isArray(messageCopy.content)) {
        // Complex case: multipart message.
        // Strategy: consolidate text, remove images if needed, then truncate text.
        const textParts = messageCopy.content.filter((p) => p.type === 'text');
        const imageParts = messageCopy.content.filter((p) => p.type === 'image_url');
        let combinedText = textParts.map((p) => p.text).join('\n');
        let keptImages = [...imageParts];
        while (combinedText.length + (keptImages.length * 2500) > charLimit && keptImages.length > 0) {
            keptImages.pop(); // remove images from the end
        }
        const imageChars = keptImages.length * 2500;
        const textCharLimit = charLimit - imageChars;
        if (combinedText.length > textCharLimit) {
            if (textCharLimit > TRUNCATION_SUFFIX.length) {
                combinedText = combinedText.substring(0, textCharLimit - TRUNCATION_SUFFIX.length) + TRUNCATION_SUFFIX;
            }
            else if (textCharLimit >= 0) {
                combinedText = combinedText.substring(0, textCharLimit);
            }
            else {
                combinedText = "";
            }
        }
        const newContent = [];
        if (combinedText) {
            newContent.push({ type: 'text', text: combinedText });
        }
        newContent.push(...keptImages);
        messageCopy.content = newContent;
    }
    return messageCopy;
}
function truncateMessages(messages, limit) {
    const systemMessage = messages.find(m => m.role === 'system');
    const otherMessages = messages.filter(m => m.role !== 'system');
    let totalChars = otherMessages.reduce((sum, msg) => sum + countChars(msg), 0);
    if (totalChars <= limit) {
        return messages;
    }
    const mutableOtherMessages = JSON.parse(JSON.stringify(otherMessages));
    let excessChars = totalChars - limit;
    // Truncate messages starting from the second one.
    for (let i = 1; i < mutableOtherMessages.length; i++) {
        if (excessChars <= 0)
            break;
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
function concatMessageText(messages) {
    const textParts = [];
    for (const message of messages) {
        if (message.content) {
            if (typeof message.content === 'string') {
                textParts.push(message.content);
            }
            else if (Array.isArray(message.content)) {
                for (const part of message.content) {
                    if (part.type === 'text') {
                        textParts.push(part.text);
                    }
                    else if (part.type === 'image_url') {
                        textParts.push('[IMAGE]');
                    }
                }
            }
        }
    }
    return textParts.join(' ');
}
function getPromptSummary(messages) {
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
function normalizeOptions(arg1, arg2) {
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
    return options;
}
/**
 * Factory function that creates a GPT "prompt" function, with optional caching.
 * @param params - The core dependencies (API key, base URL, default model, and optional cache instance).
 * @returns An async function `prompt` ready to make OpenAI calls, with caching if configured.
 */
function createLlmClient(params) {
    const { openai, cache: cacheInstance, defaultModel: factoryDefaultModel, maxConversationChars, queue } = params;
    const getCompletionParamsAndCacheKey = (options) => {
        const { ttl, model: callSpecificModel, messages, reasoning_effort, retries, ...restApiOptions } = options;
        // Ensure messages is an array (it should be if normalized, but for safety/types)
        const messagesArray = typeof messages === 'string'
            ? [{ role: 'user', content: messages }]
            : messages;
        const finalMessages = maxConversationChars ? truncateMessages(messagesArray, maxConversationChars) : messagesArray;
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
        let cacheKey;
        if (cacheInstance) {
            const cacheKeyString = JSON.stringify(completionParams);
            cacheKey = `gptask:${crypto_1.default.createHash('md5').update(cacheKeyString).digest('hex')}`;
        }
        return { completionParams, cacheKey, ttl, modelToUse, finalMessages, retries };
    };
    async function prompt(arg1, arg2) {
        const options = normalizeOptions(arg1, arg2);
        const { completionParams, cacheKey, ttl, modelToUse, finalMessages, retries } = getCompletionParamsAndCacheKey(options);
        if (cacheInstance && cacheKey) {
            try {
                const cachedResponse = await cacheInstance.get(cacheKey);
                if (cachedResponse !== undefined && cachedResponse !== null) {
                    return JSON.parse(cachedResponse);
                }
            }
            catch (error) {
                console.warn("Cache get error:", error);
            }
        }
        const promptSummary = getPromptSummary(finalMessages);
        const apiCallAndCache = async () => {
            const task = () => (0, retryUtils_js_1.executeWithRetry)(async () => {
                return openai.chat.completions.create(completionParams);
            }, async (completion) => {
                return { isValid: true, data: completion };
            }, retries ?? 3, undefined, (error) => {
                // Do not retry if the API key is invalid (401) or if the error code explicitly states it.
                if (error?.status === 401 || error?.code === 'invalid_api_key') {
                    return false;
                }
                return true;
            });
            const response = (await (queue ? queue.add(task, { id: promptSummary }) : task()));
            if (cacheInstance && response && cacheKey) {
                try {
                    await cacheInstance.set(cacheKey, JSON.stringify(response), ttl);
                }
                catch (error) {
                    console.warn("Cache set error:", error);
                }
            }
            return response;
        };
        return apiCallAndCache();
    }
    async function isPromptCached(arg1, arg2) {
        const options = normalizeOptions(arg1, arg2);
        const { cacheKey } = getCompletionParamsAndCacheKey(options);
        if (!cacheInstance || !cacheKey) {
            return false;
        }
        try {
            const cachedResponse = await cacheInstance.get(cacheKey);
            return cachedResponse !== undefined && cachedResponse !== null;
        }
        catch (error) {
            console.warn("Cache get error:", error);
            return false;
        }
    }
    async function promptText(arg1, arg2) {
        const options = normalizeOptions(arg1, arg2);
        const response = await prompt(options);
        return response.choices[0]?.message?.content || null;
    }
    async function promptImage(arg1, arg2) {
        const options = normalizeOptions(arg1, arg2);
        const response = await prompt(options);
        const message = response.choices[0]?.message;
        if (message.images && Array.isArray(message.images) && message.images.length > 0) {
            const imageUrl = message.images[0].image_url.url;
            if (typeof imageUrl === 'string') {
                if (imageUrl.startsWith('http')) {
                    const imgRes = await fetch(imageUrl);
                    const arrayBuffer = await imgRes.arrayBuffer();
                    return Buffer.from(arrayBuffer);
                }
                else {
                    const base64Data = imageUrl.replace(/^data:image\/\w+;base64,/, "");
                    return Buffer.from(base64Data, 'base64');
                }
            }
        }
        return null;
    }
    return { prompt, isPromptCached, promptText, promptImage };
}
